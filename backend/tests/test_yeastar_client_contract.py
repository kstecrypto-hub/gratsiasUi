from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock

import httpx
import pytest

from app.core.config import Settings
from app.services.yeastar.client import YeastarClient
from app.services.yeastar.cdr import CDRSummary
from app.services.yeastar.errors import YeastarConnectionError, YeastarOperationError
from app.services.yeastar.schemas import SystemInformation


class _AsyncLock:
    async def __aenter__(self) -> "_AsyncLock":
        return self

    async def __aexit__(self, *_: object) -> None:
        return None


class _MemoryRedis:
    """Small Redis double scoped to this module; values stay encoded by the client."""

    def __init__(self) -> None:
        self.values: dict[str, str] = {}

    async def get(self, key: str) -> str | None:
        return self.values.get(key)

    async def setex(self, key: str, _ttl: int, value: str) -> None:
        self.values[key] = value

    async def delete(self, key: str) -> int:
        return int(self.values.pop(key, None) is not None)

    def lock(self, *_: object, **__: object) -> _AsyncLock:
        return _AsyncLock()


def _settings(storage_root: Path) -> Settings:
    return Settings(
        APP_ENV="test",
        APP_SECRET_KEY="test-secret-key-that-is-at-least-32-characters",
        APP_TIMEZONE="Europe/Athens",
        STORAGE_ROOT=storage_root,
        YEASTAR_BASE_URL="https://pbx.example.test",
        YEASTAR_CLIENT_ID="client-id",
        YEASTAR_CLIENT_SECRET="client-secret",
    )


@pytest.mark.asyncio
async def test_authenticate_force_delegates_to_shared_refresh_manager(tmp_path: Path) -> None:
    client = YeastarClient(_settings(tmp_path), _MemoryRedis())
    refresh = AsyncMock(return_value="access-new")
    client.token_manager = SimpleNamespace(refresh_access_token=refresh)

    assert await client.authenticate(force=True) == "access-new"
    refresh.assert_awaited_once_with()


@pytest.mark.asyncio
async def test_connection_failure_retries_once_then_raises(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    attempts = 0

    async def handler(_request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        return httpx.Response(503, json={"message": "temporarily unavailable"})

    async def no_sleep(_delay: float) -> None:
        return None

    monkeypatch.setattr("app.services.yeastar.http_client.asyncio.sleep", no_sleep)
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(handler), base_url="https://pbx.example.test"
    ) as http:
        client = YeastarClient(_settings(tmp_path), _MemoryRedis(), http)
        client.token_manager = SimpleNamespace(
            get_access_token=AsyncMock(return_value="safe-token"),
            invalidate_access_token=AsyncMock(),
            refresh_access_token=AsyncMock(),
            open_circuit=AsyncMock(),
        )
        with pytest.raises(YeastarConnectionError, match="temporarily unavailable"):
            await client.system_information()

    assert attempts == 2


def test_cdr_api_path_uses_the_matching_v1_or_v2_prefix(tmp_path: Path) -> None:
    client = YeastarClient(_settings(tmp_path), _MemoryRedis())

    assert client._api_path("cdr/list", "v1") == "/openapi/v1.0/cdr/list"
    assert client._api_path("cdr/search", "v2") == "/openapi/v2.0/cdr/search"


@pytest.mark.asyncio
async def test_search_cdrs_forwards_each_page_and_maps_filters_and_local_dates(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    client = YeastarClient(_settings(tmp_path), _MemoryRedis())
    seen: list[tuple[str, str, dict[str, Any]]] = []

    async def authorized_get(
        endpoint: str, *, version: str = "v1", params: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        assert params is not None
        seen.append((endpoint, version, dict(params)))
        page = int(params["page"])
        return {"data": [{"uid": f"call-{page}"}], "total_number": 1001}

    monkeypatch.setattr(client, "_authorized_get", authorized_get)
    date_from = datetime(2026, 7, 1, 0, 0, tzinfo=UTC)
    date_to = datetime(2026, 7, 1, 9, 30, tzinfo=UTC)
    filters = {
        "status": "ANSWERED",
        "queue": "17",
        "recording_type": 1,
        "ignored": "must-not-leak",
    }

    first = await client.search_cdrs(date_from, date_to, filters, page=1)
    second = await client.search_cdrs(date_from, date_to, filters, page=2)

    assert first == {"data": [{"uid": "call-1"}], "total_number": 1001}
    assert second == {"data": [{"uid": "call-2"}], "total_number": 1001}
    assert [params["page"] for _, _, params in seen] == [1, 2]
    for endpoint, version, params in seen:
        assert (endpoint, version) == ("cdr/search", "v2")
        assert params["time_begin"] == "07/01/2026 03:00:00"
        assert params["time_end"] == "07/01/2026 12:30:00"
        assert params["last_status"] == "ANSWERED"
        assert params["queue_list"] == "17"
        assert params["recording_type"] == 1
        assert params["page_size"] == 500
        assert "status" not in params
        assert "queue" not in params
        assert "ignored" not in params


@pytest.mark.asyncio
async def test_legacy_cdr_search_uses_epoch_bounds_and_preserves_duplicate_uid_legs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    client = YeastarClient(
        _settings(tmp_path),
        _MemoryRedis(),
        cdr_api_version="v1",
        # This legacy display format is invalid by design: CDR V1 must not
        # read or format it when using timestamp bounds.
        system_date_format="MM/DD/YYYY",
        system_time_format="hh:mm:ss",
    )
    date_from = datetime(2026, 7, 1, 0, 0, tzinfo=UTC)
    date_to = datetime(2026, 7, 1, 9, 30, tzinfo=UTC)
    rows = [
        CDRSummary(uid="legacy-call", id="leg-a"),
        CDRSummary(uid="legacy-call", id="leg-b"),
    ]
    seen: dict[str, object] = {}

    class LegacyCDR:
        async def search_all(self, **kwargs: object) -> list[CDRSummary]:
            seen["search"] = kwargs
            return rows

        def detail_from_summaries(self, summaries: list[CDRSummary]):
            seen["detail"] = summaries
            return SimpleNamespace(
                provider_dict=lambda: {"timeline": [item.id for item in summaries]}
            )

    legacy = LegacyCDR()
    monkeypatch.setattr(client, "_cdr", lambda **_kwargs: legacy)

    results = await client.search_all_cdrs(date_from, date_to)
    detail = await client.get_cdr_detail(
        "legacy-call", summary=results[0].provider_dict()
    )

    assert seen["search"] == {
        "time_begin": int(date_from.timestamp()),
        "time_end": int(date_to.timestamp()),
        "filters": {},
    }
    assert [item.id for item in seen["detail"]] == ["leg-a", "leg-b"]
    assert detail == {"timeline": ["leg-a", "leg-b"]}
    with pytest.raises(YeastarOperationError, match="bounded call search"):
        await client.search_cdrs(date_from, date_to, None, page=1)


@pytest.mark.asyncio
async def test_legacy_appliance_connection_uses_v1_and_ignores_display_only_time_format(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    client = YeastarClient(_settings(tmp_path), _MemoryRedis())
    information = SystemInformation(
        model_name="Yeastar P560",
        firmware_version="37.20.0.78",
        system_date_format="MM/DD/YYYY",
        system_time_format="hh:mm:ss",
    )
    seen: list[str] = []

    class Extensions:
        def __init__(self, *_args: object, **_kwargs: object) -> None:
            return None

        async def verify_read_access(self) -> None:
            seen.append("extensions")

    class LegacyCDR:
        def __init__(
            self, *_args: object, api_version: str, **_kwargs: object
        ) -> None:
            seen.append(f"cdr:{api_version}")

        async def verify_read_access(self) -> None:
            seen.append("cdr-access")

    async def system_information() -> SystemInformation:
        return information

    monkeypatch.setattr(client, "system_information", system_information)
    monkeypatch.setattr("app.services.yeastar.client.YeastarExtensions", Extensions)
    monkeypatch.setattr("app.services.yeastar.client.YeastarCDR", LegacyCDR)

    _, capabilities = await client.inspect_connection()

    assert capabilities.state == "supported"
    assert capabilities.cdr_v2 is False
    assert capabilities.cdr_api_version == "v1"
    assert client.cdr_api_version == "v1"
    assert seen == ["extensions", "cdr:v1", "cdr-access"]


@pytest.mark.asyncio
async def test_ai_cdr_context_follows_offsets_and_merges_leg_context(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    client = YeastarClient(_settings(tmp_path), _MemoryRedis())
    offsets: list[int] = []

    async def authorized_get(
        endpoint: str, *, version: str = "v1", params: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        assert endpoint == "cdr/getaicontext"
        assert version == "v2"
        assert params is not None
        assert params["cdr_ids"] == "leg-a,leg-b"
        offsets.append(int(params["offset"]))
        if params["offset"] == 1:
            return {
                "data": {"leg-a": {"context": [{"text": "first"}], "language": "el"}},
                "offset": 2,
            }
        return {
            "data": {
                "leg-a": {"context": [{"text": "second"}]},
                "leg-b": {"context": [{"text": "other"}]},
            },
            "offset": -1,
        }

    monkeypatch.setattr(client, "_authorized_get", authorized_get)

    result = await client.get_ai_transcript(["leg-a", "leg-b"])

    assert offsets == [1, 2]
    assert result["leg-a"] == {
        "context": [{"text": "first"}, {"text": "second"}],
        "language": "el",
    }
    assert result["leg-b"] == {"context": [{"text": "other"}]}
