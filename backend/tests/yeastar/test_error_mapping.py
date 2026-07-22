from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable

import httpx
import pytest

from app.core.config import Settings
from app.services.yeastar.error_mapping import map_yeastar_error
from app.services.yeastar.errors import (
    YeastarApiDisabledError,
    YeastarAuthenticationError,
    YeastarIpBlockedError,
    YeastarIpForbiddenError,
    YeastarPermissionError,
    YeastarTemporarilyUnavailableError,
    YeastarUnsupportedVersionError,
)
from app.services.yeastar.http_client import YeastarHttpClient
from app.services.yeastar.schemas import AuthTrigger, ConnectionState


def configured_settings(**overrides: object) -> Settings:
    values: dict[str, object] = {
        "APP_ENV": "test",
        "APP_SECRET_KEY": "s" * 32,
        "YEASTAR_BASE_URL": "https://pbx.internal",
        "YEASTAR_CLIENT_ID": "client-id",
        "YEASTAR_CLIENT_SECRET": "client-secret",
        "YEASTAR_TRANSIENT_RETRY_COUNT": 1,
    }
    values.update(overrides)
    return Settings(_env_file=None, **values)


def client_for_handler(
    handler: Callable[[httpx.Request], httpx.Response | Awaitable[httpx.Response]],
    **settings: object,
) -> YeastarHttpClient:
    transport = httpx.MockTransport(handler)
    raw_client = httpx.AsyncClient(base_url="https://pbx.internal", transport=transport)
    return YeastarHttpClient(configured_settings(**settings), raw_client)


@pytest.mark.parametrize(
    ("code", "error_type", "state"),
    [
        (10002, YeastarUnsupportedVersionError, ConnectionState.UNSUPPORTED_API_VERSION),
        (10003, YeastarUnsupportedVersionError, ConnectionState.UNSUPPORTED_API_VERSION),
        (10005, YeastarAuthenticationError, ConnectionState.AUTH_REJECTED),
        (70004, YeastarIpBlockedError, ConnectionState.IP_BLOCKED),
        (70087, YeastarIpForbiddenError, ConnectionState.IP_NOT_ALLOWED),
        (70123, YeastarApiDisabledError, ConnectionState.API_DISABLED),
        (70656, YeastarApiDisabledError, ConnectionState.API_DISABLED),
        (80010, YeastarPermissionError, ConnectionState.PERMISSION_DENIED),
    ],
)
def test_critical_codes_preserve_exact_circuit_reason(
    code: int,
    error_type: type[Exception],
    state: ConnectionState,
) -> None:
    error = map_yeastar_error(code, "provider text must not be reflected")

    assert isinstance(error, error_type)
    assert error.opens_circuit is True
    assert error.connection_state == state
    assert error.errcode == code
    assert "provider text" not in str(error)


@pytest.mark.asyncio
async def test_http_200_with_nonzero_errcode_is_failure_and_user_agent_is_forced() -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(200, json={"errcode": 80010, "errmsg": "secret provider text"})

    client = client_for_handler(handler)
    with pytest.raises(YeastarPermissionError):
        await client.get("/openapi/v2.0/cdr/search")

    assert requests[0].headers["User-Agent"] == "YeastarCallAnalyzer/1.0"


@pytest.mark.asyncio
async def test_429_is_never_retried_but_503_has_one_retry() -> None:
    rate_limited_calls = 0

    def rate_limited(_: httpx.Request) -> httpx.Response:
        nonlocal rate_limited_calls
        rate_limited_calls += 1
        return httpx.Response(429, json={"errcode": 0})

    rate_limited_client = client_for_handler(rate_limited)
    with pytest.raises(YeastarTemporarilyUnavailableError):
        await rate_limited_client.get("/openapi/v1.0/system/information")
    assert rate_limited_calls == 1

    unavailable_calls = 0

    def unavailable(_: httpx.Request) -> httpx.Response:
        nonlocal unavailable_calls
        unavailable_calls += 1
        return httpx.Response(503, json={"errcode": 0})

    unavailable_client = client_for_handler(unavailable)
    with pytest.raises(YeastarTemporarilyUnavailableError):
        await unavailable_client.get("/openapi/v1.0/system/information")
    assert unavailable_calls == 2


@pytest.mark.asyncio
async def test_json_and_binary_requests_reject_redirects() -> None:
    def redirect(_: httpx.Request) -> httpx.Response:
        return httpx.Response(
            302,
            headers={"Location": "https://attacker.example/stolen"},
            json={"errcode": 0},
        )

    client = client_for_handler(redirect)
    with pytest.raises(Exception) as json_error:
        await client.get("/openapi/v1.0/system/information")
    assert getattr(json_error.value, "errcode", None) == 302

    with pytest.raises(Exception) as stream_error:
        async with client.stream("GET", "/api/download/resource"):
            pass
    assert getattr(stream_error.value, "errcode", None) == 302


class RecordingTokenManager:
    def __init__(self) -> None:
        self.token = "old-access-token"
        self.invalidations: list[str | None] = []
        self.opened: list[tuple[ConnectionState, int | None]] = []
        self.refreshes = 0

    async def get_access_token(self, trigger: AuthTrigger | None = None) -> str:
        return self.token

    async def invalidate_access_token(
        self, expected_access_token: str | None = None
    ) -> bool:
        self.invalidations.append(expected_access_token)
        return True

    async def refresh_access_token(self) -> str:
        self.refreshes += 1
        self.token = "new-access-token"
        return self.token

    async def open_circuit(
        self, reason: ConnectionState, *, last_errcode: int | None = None
    ) -> None:
        self.opened.append((reason, last_errcode))


@pytest.mark.asyncio
async def test_protected_request_opens_exact_circuit_on_first_or_retry_error() -> None:
    first_manager = RecordingTokenManager()

    def permission(_: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"errcode": 80010})

    first_client = client_for_handler(permission)
    with pytest.raises(YeastarPermissionError):
        await first_client.protected_request(
            "GET",
            "/openapi/v2.0/cdr/search",
            token_manager=first_manager,
            trigger=AuthTrigger.CALL_ANALYSIS,
        )
    assert first_manager.opened == [(ConnectionState.PERMISSION_DENIED, 80010)]

    retry_manager = RecordingTokenManager()
    calls = 0

    def expiry_then_permission(_: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        code = 10004 if calls == 1 else 80010
        return httpx.Response(200, json={"errcode": code})

    retry_client = client_for_handler(expiry_then_permission)
    with pytest.raises(YeastarPermissionError):
        await retry_client.protected_request(
            "GET",
            "/openapi/v2.0/cdr/search",
            token_manager=retry_manager,
            trigger=AuthTrigger.CALL_ANALYSIS,
        )
    assert retry_manager.invalidations == ["old-access-token"]
    assert retry_manager.refreshes == 1
    assert retry_manager.opened == [(ConnectionState.PERMISSION_DENIED, 80010)]


class ConditionalConcurrentTokenManager(RecordingTokenManager):
    def __init__(self) -> None:
        super().__init__()
        self.lock = asyncio.Lock()
        self.refreshed = asyncio.Event()
        self.get_barrier = asyncio.Event()
        self.get_calls = 0

    async def get_access_token(self, trigger: AuthTrigger | None = None) -> str:
        del trigger
        token = self.token
        self.get_calls += 1
        if self.get_calls == 2:
            self.get_barrier.set()
        await asyncio.wait_for(self.get_barrier.wait(), timeout=1)
        return token

    async def invalidate_access_token(
        self, expected_access_token: str | None = None
    ) -> bool:
        async with self.lock:
            self.invalidations.append(expected_access_token)
            return expected_access_token == self.token

    async def refresh_access_token(self) -> str:
        async with self.lock:
            if self.token == "old-access-token":
                self.refreshes += 1
                self.token = "new-access-token"
                self.refreshed.set()
            return self.token


@pytest.mark.asyncio
async def test_late_concurrent_10004_does_not_invalidate_refreshed_token() -> None:
    manager = ConditionalConcurrentTokenManager()
    old_calls = 0

    async def handler(request: httpx.Request) -> httpx.Response:
        nonlocal old_calls
        token = request.url.params["access_token"]
        if token == "old-access-token":
            old_calls += 1
            if old_calls == 2:
                await asyncio.wait_for(manager.refreshed.wait(), timeout=1)
            return httpx.Response(200, json={"errcode": 10004})
        return httpx.Response(200, json={"errcode": 0, "value": "ok"})

    client = client_for_handler(handler)
    first, second = await asyncio.gather(
        client.protected_request(
            "GET",
            "/openapi/v2.0/cdr/search",
            token_manager=manager,
            trigger=AuthTrigger.CALL_ANALYSIS,
        ),
        client.protected_request(
            "GET",
            "/openapi/v2.0/cdr/search",
            token_manager=manager,
            trigger=AuthTrigger.CALL_ANALYSIS,
        ),
    )

    assert first["errcode"] == second["errcode"] == 0
    assert manager.refreshes == 1
    assert manager.invalidations == ["old-access-token", "old-access-token"]
