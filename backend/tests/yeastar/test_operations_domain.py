from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import httpx
import pytest

from app.core.config import Settings
from app.services.yeastar.client import YeastarClient
from app.services.yeastar.cdr import CDRTimelineEntry, YeastarCDR
from app.services.yeastar.errors import (
    YeastarRecordingDownloadLimitError,
    YeastarSecurityError,
)
from app.services.yeastar.extensions import YeastarExtensions
from app.services.yeastar.recordings import (
    RecordingDownload,
    YeastarRecordings,
    validate_download_resource_path,
)


class PagingClient:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str, dict[str, object]]] = []

    async def get(self, endpoint: str, *, version: str = "v1", params=None):
        values = dict(params or {})
        self.calls.append((endpoint, version, values))
        page = int(values["page"])
        if endpoint == "extension/list":
            return {
                "errcode": 0,
                "data": [
                    {"id": str(page), "number": str(1000 + page)}
                ],
                "total_number": 3,
            }
        if endpoint == "cdr/search":
            return {
                "errcode": 0,
                "data": [{"uid": f"cdr-{page}"}],
                "total_number": 3,
            }
        raise AssertionError(endpoint)


@pytest.mark.asyncio
async def test_extension_and_cdr_pagination_use_configured_page_size() -> None:
    client = PagingClient()
    extensions = await YeastarExtensions(client, page_size=500).list_all()
    cdrs = await YeastarCDR(client, page_size=500).search_all(
        time_begin="07/01/2026 00:00:00",
        time_end="07/02/2026 00:00:00",
        filters={"recording_type": 1},
    )

    assert [item.id for item in extensions] == ["1", "2", "3"]
    assert [item.uid for item in cdrs] == ["cdr-1", "cdr-2", "cdr-3"]
    assert len(client.calls) == 6
    assert all(call[2]["page_size"] == 500 for call in client.calls)


def test_cdr_timeline_persists_transaction_and_cdr_ids_separately() -> None:
    mapping = CDRTimelineEntry(
        leg=2,
        transaction_id="transaction-17",
        cdr_id="cdr-leg-91",
        call_from="Alice",
        call_to="Queue 3",
        call_from_number="1001",
        call_to_number="6300",
        call_from_ext_id="ext-1",
        call_to_ext_id="ext-63",
        ring_duration=4,
        talk_duration=22,
        hold_duration=3,
        event_list=[{"event": "ANSWER"}],
    ).persistence_mapping(1)

    assert mapping["yeastar_leg_id"] == "transaction-17:cdr-leg-91"
    assert mapping["transaction_id"] == "transaction-17"
    assert mapping["yeastar_cdr_id"] == "cdr-leg-91"
    assert mapping["call_from"] == "Alice"
    assert mapping["call_to"] == "Queue 3"
    assert mapping["event_list"] == [{"event": "ANSWER"}]


@pytest.mark.parametrize(
    "value",
    [
        "https://pbx.example.test/api/download/file.mp3",
        "//attacker.example/api/download/file.mp3",
        "/not-api/download/file.mp3",
        "/api/download/../secret",
        "/api/download/%2e%2e/secret",
        "/api/download/%252e%252e/secret",
        "/api/download/file.mp3?access_token=secret",
        "/api/opaque-resource%3Faccess_token=secret",
    ],
)
def test_download_resource_path_rejects_absolute_and_traversal(value: str) -> None:
    with pytest.raises(YeastarSecurityError):
        validate_download_resource_path(value)


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("/api/download/file.mp3", "/api/download/file.mp3"),
        ("/api/temporary-resource/recording", "/api/temporary-resource/recording"),
        ("api/download/file.mp3", "/api/download/file.mp3"),
    ],
)
def test_download_resource_path_accepts_safe_pbx_temporary_api_paths(
    value: str, expected: str
) -> None:
    assert validate_download_resource_path(value) == expected


class _Lock:
    async def __aenter__(self):
        return self

    async def __aexit__(self, *_args: object) -> None:
        return None


class _Redis:
    def lock(self, *_args: object, **_kwargs: object) -> _Lock:
        return _Lock()


class DownloadLimitClient:
    def __init__(self) -> None:
        self.resource_calls = 0
        self.stream_calls = 0

    async def get(self, *_args: object, **_kwargs: object):
        raise AssertionError("search was not expected")

    async def request_recording_download_resource(self, recording_id: str):
        assert recording_id == "recording-1"
        self.resource_calls += 1
        raise YeastarRecordingDownloadLimitError(
            "The phone system is busy preparing another recording download.", 70651
        )

    async def stream_download(self, *_args: object, **_kwargs: object):
        self.stream_calls += 1
        raise AssertionError("stream must not start after 70651")


@pytest.mark.asyncio
async def test_70651_has_no_inner_retry_and_is_deferred_to_task_queue(
    tmp_path: Path,
) -> None:
    client = DownloadLimitClient()
    recordings = YeastarRecordings(
        client,
        _Redis(),
        tmp_path,
        download_limit_retries=0,
    )

    with pytest.raises(YeastarRecordingDownloadLimitError):
        await recordings.download("recording-1", tmp_path / "recording.mp3")

    assert client.resource_calls == 1
    assert client.stream_calls == 0
    assert not (tmp_path / "recording.mp3.part").exists()


def test_recording_timestamps_are_unix_seconds() -> None:
    value = datetime(2026, 7, 15, 12, 30, tzinfo=UTC)
    assert int(value.timestamp()) == 1_784_118_600


def _client_settings(tmp_path: Path) -> Settings:
    return Settings(
        APP_ENV="test",
        APP_SECRET_KEY="test-secret-key-that-is-at-least-32-characters",
        STORAGE_ROOT=tmp_path,
        YEASTAR_BASE_URL="https://pbx.example.test",
        YEASTAR_CLIENT_ID="client-id",
        YEASTAR_CLIENT_SECRET="client-secret",
    )


class _CapabilityRedis:
    def __init__(self) -> None:
        self.values: dict[str, str] = {}
        self.writes: list[tuple[str, int, str]] = []

    async def get(self, key: str) -> str | None:
        return self.values.get(key)

    async def setex(self, key: str, ttl: int, value: str) -> None:
        self.values[key] = value
        self.writes.append((key, ttl, value))


@pytest.mark.asyncio
async def test_stereo_capability_cache_is_scoped_to_pbx_configuration(
    tmp_path: Path,
) -> None:
    redis = _CapabilityRedis()
    first_settings = _client_settings(tmp_path)
    second_settings = first_settings.model_copy(
        update={
            "YEASTAR_BASE_URL": "https://second-pbx.example.test",
            "YEASTAR_CLIENT_ID": "second-client",
        }
    )
    first = YeastarClient(first_settings, redis)  # type: ignore[arg-type]
    second = YeastarClient(second_settings, redis)  # type: ignore[arg-type]
    first._authorized_get = AsyncMock(  # type: ignore[method-assign]
        return_value={"auto_record": {"enb_channel_separate": "1"}}
    )
    second._authorized_get = AsyncMock(  # type: ignore[method-assign]
        return_value={"auto_record": {"enb_channel_separate": "0"}}
    )

    assert await first.stereo_separated_recording_enabled() is True
    assert await second.stereo_separated_recording_enabled() is False
    assert len(redis.values) == 2
    assert all(key.startswith("yca:yeastar:stereo-capability:v2:") for key in redis.values)
    assert first_settings.YEASTAR_BASE_URL not in " ".join(redis.values)
    assert second_settings.YEASTAR_BASE_URL not in " ".join(redis.values)

    first._authorized_get.reset_mock()
    assert await first.stereo_separated_recording_enabled() is True
    first._authorized_get.assert_not_awaited()


@pytest.mark.asyncio
async def test_streamed_json_error_is_never_moved_as_audio(tmp_path: Path) -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.host == "pbx.example.test"
        assert request.url.path == "/api/temporary-resource/recording"
        return httpx.Response(
            200,
            headers={"content-type": "application/json"},
            json={"errcode": 70651, "errmsg": "busy"},
        )

    async with httpx.AsyncClient(
        transport=httpx.MockTransport(handler), base_url="https://pbx.example.test"
    ) as http:
        client = YeastarClient(_client_settings(tmp_path), object(), http)
        client.token_manager = SimpleNamespace(open_circuit=AsyncMock())
        destination = tmp_path / "recording.mp3.part"
        with pytest.raises(YeastarRecordingDownloadLimitError):
            await client.stream_download(
                "/api/temporary-resource/recording",
                recording_id="recording-1",
                access_token="safe-token",
                destination=destination,
            )

    assert not destination.exists()


@pytest.mark.asyncio
async def test_step_two_10004_refreshes_once_and_restarts_both_steps(
    tmp_path: Path,
) -> None:
    requests: list[tuple[str, str | None]] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        requests.append((request.url.path, request.url.params.get("access_token")))
        if request.url.path == "/api/download/old.mp3":
            return httpx.Response(
                200,
                headers={"content-type": "application/json"},
                json={"errcode": 10004, "errmsg": "expired"},
            )
        if request.url.path == "/openapi/v1.0/recording/download":
            assert request.url.params["id"] == "recording-1"
            return httpx.Response(
                200,
                json={
                    "errcode": 0,
                    "errmsg": "SUCCESS",
                    "download_resource_url": "/api/download/new.mp3",
                },
            )
        if request.url.path == "/api/download/new.mp3":
            return httpx.Response(
                200,
                headers={"content-type": "audio/mpeg"},
                content=b"ID3-safe-audio",
            )
        raise AssertionError(request.url.path)

    invalidate = AsyncMock(return_value=True)
    refresh = AsyncMock(return_value="refreshed-token")
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(handler), base_url="https://pbx.example.test"
    ) as http:
        client = YeastarClient(_client_settings(tmp_path), object(), http)
        client.token_manager = SimpleNamespace(
            invalidate_access_token=invalidate,
            refresh_access_token=refresh,
            open_circuit=AsyncMock(),
        )
        destination = tmp_path / "recording.mp3.part"
        result = await client.stream_download(
            "/api/download/old.mp3",
            recording_id="recording-1",
            access_token="expired-token",
            destination=destination,
        )

    invalidate.assert_awaited_once_with("expired-token")
    refresh.assert_awaited_once_with()
    assert requests == [
        ("/api/download/old.mp3", "expired-token"),
        ("/openapi/v1.0/recording/download", "refreshed-token"),
        ("/api/download/new.mp3", "refreshed-token"),
    ]
    assert result.size_bytes == len(b"ID3-safe-audio")
    assert destination.read_bytes() == b"ID3-safe-audio"


class _SerializedLock:
    def __init__(self, guard: asyncio.Lock) -> None:
        self.guard = guard

    async def __aenter__(self):
        await self.guard.acquire()
        return self

    async def __aexit__(self, *_args: object) -> None:
        self.guard.release()


class _SerializedRedis:
    def __init__(self) -> None:
        self.guard = asyncio.Lock()

    def lock(self, *_args: object, **_kwargs: object) -> _SerializedLock:
        return _SerializedLock(self.guard)


class _ConcurrentDownloadClient:
    def __init__(self) -> None:
        self.active = 0
        self.maximum_active = 0

    async def get(self, *_args: object, **_kwargs: object):
        raise AssertionError("search was not expected")

    async def request_recording_download_resource(self, recording_id: str):
        return {"download_resource_url": f"/api/download/{recording_id}.mp3"}, "token"

    async def stream_download(
        self, _resource_path: str, *, destination: Path, **_kwargs: object
    ) -> RecordingDownload:
        self.active += 1
        self.maximum_active = max(self.maximum_active, self.active)
        await asyncio.sleep(0)
        destination.write_bytes(b"audio")
        self.active -= 1
        return RecordingDownload(destination, 5, "audio/mpeg")


@pytest.mark.asyncio
async def test_recording_download_concurrency_is_globally_one(tmp_path: Path) -> None:
    client = _ConcurrentDownloadClient()
    service = YeastarRecordings(client, _SerializedRedis(), tmp_path)

    await asyncio.gather(
        service.download("one", tmp_path / "one.mp3"),
        service.download("two", tmp_path / "two.mp3"),
    )

    assert client.maximum_active == 1
