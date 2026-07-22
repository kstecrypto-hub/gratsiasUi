from __future__ import annotations

from typing import Any

import pytest

from app.services.yeastar.cdr import CDRSummary, YeastarCDR
from app.services.yeastar.errors import YeastarOperationError


class LegacyListClient:
    def __init__(self, pages: dict[int, dict[str, object]]) -> None:
        self.pages = pages
        self.calls: list[tuple[str, str, dict[str, object]]] = []

    async def get(
        self,
        endpoint: str,
        *,
        version: str = "v1",
        params: dict[str, Any] | None = None,
    ) -> dict[str, object]:
        values = dict(params or {})
        self.calls.append((endpoint, version, values))
        return self.pages[int(values["page"])]


def legacy_row(
    *,
    uid: str,
    record_id: int,
    timestamp: int,
    call_type: str = "Inbound",
    disposition: str = "ANSWERED",
    call_from_number: str = "+302100000000",
    call_to_number: str = "1001",
    duration: int = 30,
    call_id: str = "provider-call",
    record_file: str = "",
) -> dict[str, object]:
    return {
        "id": record_id,
        "uid": uid,
        "time": "MM/DD/YYYY hh:mm:ss",  # Intentionally unusable display text.
        "timestamp": timestamp,
        "call_type": call_type,
        "call_from": f"Caller<{call_from_number}>",
        "call_to": f"Extension<{call_to_number}>",
        "call_from_number": call_from_number,
        "call_to_number": call_to_number,
        "duration": duration,
        "ring_duration": 4,
        "talk_duration": max(0, duration - 4),
        "disposition": disposition,
        "call_id": call_id,
        "record_file": record_file,
    }


@pytest.mark.asyncio
async def test_v1_page_uses_only_documented_list_parameters_and_normalizes_row() -> None:
    client = LegacyListClient(
        {
            1: {
                "errcode": 0,
                "total_number": 1,
                "data": [
                    legacy_row(
                        uid="legacy-call",
                        record_id=17,
                        timestamp=1_784_118_600,
                        duration=42,
                        record_file="20260715123000-17.wav",
                    )
                ],
            }
        }
    )

    page = await YeastarCDR(client, page_size=50, api_version="v1").legacy_page(
        page=1,
        page_size=50,
    )

    assert client.calls == [
        (
            "cdr/list",
            "v1",
            {"page": 1, "page_size": 50, "sort_by": "time", "order_by": "desc"},
        )
    ]
    assert page.total_number == 1
    assert page.data[0].provider_dict() == {
        "id": 17,
        "uid": "legacy-call",
        "time": 1_784_118_600,
        "timestamp": 1_784_118_600,
        "call_type": "Inbound",
        "call_from": "Caller<+302100000000>",
        "call_to": "Extension<1001>",
        "call_from_number": "+302100000000",
        "call_to_number": "1001",
        "duration": 42,
        "ring_duration": 4,
        "talk_duration": 38,
        "disposition": "ANSWERED",
        "call_id": "provider-call",
        "record_file": "20260715123000-17.wav",
        "last_status": "ANSWERED",
        "call_duration": 42,
        "legacy_cdr": True,
    }


@pytest.mark.asyncio
async def test_v1_search_all_keeps_all_legs_when_one_leg_matches_and_stops_at_old_page() -> None:
    client = LegacyListClient(
        {
            1: {
                "total_number": 9_999,
                "data": [
                    legacy_row(
                        uid="transferred-call",
                        record_id=200,
                        timestamp=500,
                        call_type="Inbound",
                        call_to_number="1001",
                        call_id="transfer-77",
                    ),
                    legacy_row(
                        uid="transferred-call",
                        record_id=199,
                        timestamp=499,
                        call_type="Internal",
                        call_from_number="1001",
                        call_to_number="1002",
                        call_id="transfer-77",
                    ),
                ],
            },
            2: {
                "total_number": 9_999,
                "data": [
                    legacy_row(uid="old-call", record_id=198, timestamp=299),
                    legacy_row(uid="older-call", record_id=197, timestamp=298),
                ],
            },
        }
    )
    adapter = YeastarCDR(client, page_size=2, api_version="v1")

    records = await adapter.search_all(
        time_begin=400,
        time_end=600,
        filters={"call_type": "Inbound"},
    )

    assert [(item.uid, item.id, item.call_type) for item in records] == [
        ("transferred-call", 200, "Inbound"),
        ("transferred-call", 199, "Internal"),
    ]
    assert [call[2]["page"] for call in client.calls] == [1, 2]
    assert all(call[:2] == ("cdr/list", "v1") for call in client.calls)
    assert all(set(call[2]) == {"page", "page_size", "sort_by", "order_by"} for call in client.calls)


@pytest.mark.asyncio
async def test_v1_search_all_paginates_and_applies_record_file_filter_locally() -> None:
    client = LegacyListClient(
        {
            1: {
                "total_number": 3,
                "data": [
                    legacy_row(
                        uid="with-file",
                        record_id=3,
                        timestamp=300,
                        record_file="recording-3.wav",
                    ),
                    legacy_row(uid="without-file", record_id=2, timestamp=299),
                ],
            },
            2: {
                "total_number": 3,
                "data": [legacy_row(uid="with-file-2", record_id=1, timestamp=298, record_file="one.wav")],
            },
        }
    )

    records = await YeastarCDR(client, page_size=2, api_version="v1").search_all(
        time_begin=1,
        time_end=400,
        filters={"recording_type": 1},
    )

    assert [item.uid for item in records] == ["with-file", "with-file-2"]
    assert [call[2]["page"] for call in client.calls] == [1, 2]


@pytest.mark.asyncio
async def test_v1_rejects_unsupported_queue_filter_without_a_phone_system_request() -> None:
    client = LegacyListClient({})
    adapter = YeastarCDR(client, api_version="v1")

    with pytest.raises(YeastarOperationError, match="queue filtering"):
        await adapter.search_all(
            time_begin=1,
            time_end=2,
            filters={"queue_list": "600"},
        )

    assert client.calls == []


@pytest.mark.asyncio
async def test_v1_detail_synthesizes_a_timeline_for_each_legacy_uid_row() -> None:
    first = CDRSummary.from_legacy(
        legacy_row(
            uid="shared-uid",
            record_id=22,
            timestamp=200,
            call_from_number="1001",
            call_to_number="1002",
            call_id="call-transaction",
            record_file="call.wav",
        )
    )
    second = CDRSummary.from_legacy(
        legacy_row(
            uid="shared-uid",
            record_id=21,
            timestamp=199,
            call_type="Internal",
            disposition="NO ANSWER",
            call_from_number="1002",
            call_to_number="1003",
            call_id="call-transaction",
        )
    )
    client = LegacyListClient({})
    adapter = YeastarCDR(client, api_version="v1")

    detail = await adapter.detail("shared-uid", summaries=[first, second])

    assert detail.basic.uid == "shared-uid"
    assert [(item.transaction_id, item.cdr_id, item.status) for item in detail.timeline] == [
        ("call-transaction", "21", "NO ANSWER"),
        ("call-transaction", "22", "ANSWERED"),
    ]
    assert detail.timeline[1].model_dump()["record_file"] == "call.wav"
    assert client.calls == []
