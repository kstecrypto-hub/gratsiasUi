from __future__ import annotations

import asyncio
import time
from datetime import UTC, datetime, timedelta

import pytest

from app.core.config import Settings
from app.services.yeastar.circuit_breaker import YeastarCircuitBreaker
from app.services.yeastar.errors import (
    YeastarAuthenticationError,
    YeastarAuthenticationRequiredError,
    YeastarCircuitOpenError,
    YeastarTokenStateError,
)
from app.services.yeastar.schemas import (
    AuthTrigger,
    ConnectionState,
    TokenResponse,
    TokenState,
)
from app.services.yeastar.token_manager import YeastarTokenManager
from app.services.yeastar.token_store import (
    YEASTAR_TOKEN_LOCK_KEY,
    YEASTAR_TOKEN_STATE_KEY,
    RedisOwnerLock,
    YeastarTokenStore,
)


class MemoryRedis:
    def __init__(self) -> None:
        self.values: dict[str, object] = {}
        self.expiries: dict[str, float] = {}

    def _prune(self, key: str) -> None:
        expiry = self.expiries.get(key)
        if expiry is not None and expiry <= time.monotonic():
            self.values.pop(key, None)
            self.expiries.pop(key, None)

    async def get(self, key: str) -> object | None:
        self._prune(key)
        return self.values.get(key)

    async def set(
        self,
        key: str,
        value: object,
        *,
        nx: bool = False,
        px: int | None = None,
        ex: int | None = None,
    ) -> bool:
        self._prune(key)
        if nx and key in self.values:
            return False
        self.values[key] = value
        if px is not None:
            self.expiries[key] = time.monotonic() + px / 1000
        elif ex is not None:
            self.expiries[key] = time.monotonic() + ex
        else:
            self.expiries.pop(key, None)
        return True

    async def delete(self, *keys: str) -> int:
        removed = 0
        for key in keys:
            self._prune(key)
            removed += int(key in self.values)
            self.values.pop(key, None)
            self.expiries.pop(key, None)
        return removed

    async def eval(
        self,
        script: str,
        number_of_keys: int,
        key: str,
        owner: str,
    ) -> int:
        del script, number_of_keys
        self._prune(key)
        if self.values.get(key) == owner:
            await self.delete(key)
            return 1
        return 0


def configured_settings(**overrides: object) -> Settings:
    values: dict[str, object] = {
        "APP_ENV": "test",
        "APP_SECRET_KEY": "s" * 32,
        "YEASTAR_BASE_URL": "https://pbx.internal",
        "YEASTAR_CLIENT_ID": "client-id",
        "YEASTAR_CLIENT_SECRET": "client-secret",
        "YEASTAR_TOKEN_LOCK_WAIT_SECONDS": 2,
    }
    values.update(overrides)
    return Settings(_env_file=None, **values)


def token_state(
    settings: Settings,
    *,
    access_token: str = "access-token-value",
    refresh_token: str = "refresh-token-value",
    generation: int = 1,
) -> TokenState:
    now = datetime.now(UTC)
    return TokenState(
        access_token=access_token,
        access_token_expires_at=now + timedelta(minutes=10),
        refresh_token=refresh_token,
        refresh_token_expires_at=now + timedelta(hours=1),
        issued_at=now,
        generation=generation,
        configuration_fingerprint=settings.yeastar_configuration_fingerprint,
    )


@pytest.mark.asyncio
async def test_token_state_is_encrypted_and_ttl_tracks_refresh_expiry() -> None:
    redis = MemoryRedis()
    settings = configured_settings()
    store = YeastarTokenStore(redis, settings.APP_SECRET_KEY or "")  # type: ignore[arg-type]
    state = token_state(settings)

    await store.write(state)
    raw = await redis.get(YEASTAR_TOKEN_STATE_KEY)

    assert isinstance(raw, bytes)
    assert b"access-token-value" not in raw
    assert b"refresh-token-value" not in raw
    assert 3500 <= redis.expiries[YEASTAR_TOKEN_STATE_KEY] - time.monotonic() <= 3601
    restored = await store.read()
    assert restored is not None
    assert restored.access_token_value == "access-token-value"
    assert restored.refresh_token_value == "refresh-token-value"


@pytest.mark.asyncio
async def test_conditional_invalidation_cannot_expire_a_new_generation() -> None:
    redis = MemoryRedis()
    settings = configured_settings()
    store = YeastarTokenStore(redis, settings.APP_SECRET_KEY or "")  # type: ignore[arg-type]
    await store.write(token_state(settings, access_token="old-token"))

    assert await store.invalidate_access_token("old-token") is True
    await store.write(
        token_state(
            settings,
            access_token="new-token",
            refresh_token="new-refresh-token",
            generation=2,
        )
    )

    late_result = await store.invalidate_access_token("old-token")
    current = await store.read()

    assert late_result is False
    assert current is not None
    assert current.access_token_value == "new-token"
    assert current.access_token_expires_at > datetime.now(UTC)


@pytest.mark.asyncio
async def test_owner_mismatch_cannot_release_another_owners_lock() -> None:
    redis = MemoryRedis()
    stale_owner = RedisOwnerLock(redis, timeout_seconds=30, wait_seconds=0)  # type: ignore[arg-type]
    assert await stale_owner.acquire() is True
    assert stale_owner.owner is not None

    await redis.set(YEASTAR_TOKEN_LOCK_KEY, "replacement-owner", px=30_000)

    assert await stale_owner.release() is False
    assert await redis.get(YEASTAR_TOKEN_LOCK_KEY) == "replacement-owner"


class CountingAuthHttpClient:
    def __init__(self) -> None:
        self.calls = 0

    async def request_model(self, model: object, *args: object, **kwargs: object) -> TokenResponse:
        del model, args, kwargs
        self.calls += 1
        await asyncio.sleep(0)
        return TokenResponse(
            access_token_expire_time=1800,
            access_token="issued-access-token",
            refresh_token_expire_time=86400,
            refresh_token="issued-refresh-token",
        )

    async def request_json(self, *args: object, **kwargs: object) -> dict[str, int]:
        del args, kwargs
        return {"errcode": 0}

    async def aclose(self) -> None:
        return None


class RefreshHttpClient(CountingAuthHttpClient):
    def __init__(self) -> None:
        super().__init__()
        self.refresh_bodies: list[object] = []

    async def request_model(self, model: object, *args: object, **kwargs: object) -> TokenResponse:
        del model, args
        self.calls += 1
        self.refresh_bodies.append(kwargs.get("json_body"))
        await asyncio.sleep(0)
        return TokenResponse(
            access_token_expire_time=1800,
            access_token="latest-access-token",
            refresh_token_expire_time=86400,
            refresh_token="latest-refresh-token",
        )


class RejectingAuthHttpClient(CountingAuthHttpClient):
    async def request_model(self, model: object, *args: object, **kwargs: object) -> TokenResponse:
        del model, args, kwargs
        self.calls += 1
        await asyncio.sleep(0.05)
        raise YeastarAuthenticationError("Credentials were rejected.", 10005)


@pytest.mark.asyncio
async def test_concurrent_workers_share_one_initial_authentication() -> None:
    redis = MemoryRedis()
    settings = configured_settings()
    http = CountingAuthHttpClient()
    manager = YeastarTokenManager(
        settings,
        redis=redis,  # type: ignore[arg-type]
        http_client=http,  # type: ignore[arg-type]
        circuit_breaker=YeastarCircuitBreaker(redis),  # type: ignore[arg-type]
    )

    tokens = await asyncio.gather(
        *(manager.get_access_token(AuthTrigger.OPERATOR_SYNC) for _ in range(100))
    )

    assert tokens == ["issued-access-token"] * 100
    assert http.calls == 1


@pytest.mark.asyncio
async def test_near_expiry_refreshes_once_and_persists_latest_token_pair() -> None:
    redis = MemoryRedis()
    settings = configured_settings()
    store = YeastarTokenStore(
        redis,  # type: ignore[arg-type]
        settings.APP_SECRET_KEY or "",
        configuration_fingerprint=settings.yeastar_configuration_fingerprint,
    )
    now = datetime.now(UTC)
    await store.write(
        TokenState(
            access_token="nearly-expired-token",
            access_token_expires_at=now + timedelta(seconds=30),
            refresh_token="previous-refresh-token",
            refresh_token_expires_at=now + timedelta(hours=1),
            issued_at=now,
            generation=1,
            configuration_fingerprint=settings.yeastar_configuration_fingerprint,
        )
    )
    http = RefreshHttpClient()
    manager = YeastarTokenManager(
        settings,
        redis=redis,  # type: ignore[arg-type]
        http_client=http,  # type: ignore[arg-type]
        store=store,
        circuit_breaker=YeastarCircuitBreaker(redis),  # type: ignore[arg-type]
    )

    tokens = await asyncio.gather(
        *(manager.get_access_token(AuthTrigger.CALL_ANALYSIS) for _ in range(50))
    )
    persisted = await store.read()

    assert tokens == ["latest-access-token"] * 50
    assert http.calls == 1
    assert http.refresh_bodies == [{"refresh_token": "previous-refresh-token"}]
    assert persisted is not None
    assert persisted.access_token_value == "latest-access-token"
    assert persisted.refresh_token_value == "latest-refresh-token"
    assert persisted.generation == 2


@pytest.mark.asyncio
async def test_parallel_invalid_authentication_makes_exactly_one_provider_attempt() -> None:
    redis = MemoryRedis()
    settings = configured_settings()
    http = RejectingAuthHttpClient()
    breaker = YeastarCircuitBreaker(redis)  # type: ignore[arg-type]
    manager = YeastarTokenManager(
        settings,
        redis=redis,  # type: ignore[arg-type]
        http_client=http,  # type: ignore[arg-type]
        circuit_breaker=breaker,
    )

    results = await asyncio.gather(
        *(
            manager.get_access_token(AuthTrigger.MANUAL_CONNECTION_TEST)
            for _ in range(100)
        ),
        return_exceptions=True,
    )

    assert http.calls == 1
    assert sum(isinstance(item, YeastarAuthenticationError) for item in results) == 1
    assert sum(isinstance(item, YeastarCircuitOpenError) for item in results) == 99
    circuit = await breaker.get_state()
    assert circuit is not None
    assert circuit.reason == ConnectionState.AUTH_REJECTED
    assert circuit.last_errcode == 10005


@pytest.mark.asyncio
async def test_triggerless_call_never_creates_a_new_token() -> None:
    redis = MemoryRedis()
    settings = configured_settings()
    http = CountingAuthHttpClient()
    manager = YeastarTokenManager(
        settings,
        redis=redis,  # type: ignore[arg-type]
        http_client=http,  # type: ignore[arg-type]
        circuit_breaker=YeastarCircuitBreaker(redis),  # type: ignore[arg-type]
    )

    with pytest.raises(YeastarAuthenticationRequiredError):
        await manager.get_access_token()

    assert http.calls == 0


@pytest.mark.asyncio
async def test_invalid_configuration_can_clear_local_state_without_network_or_circuit_reset() -> None:
    redis = MemoryRedis()
    valid_settings = configured_settings()
    store = YeastarTokenStore(
        redis,
        valid_settings.APP_SECRET_KEY or "",  # type: ignore[arg-type]
        configuration_fingerprint=valid_settings.yeastar_configuration_fingerprint,
    )
    await store.write(token_state(valid_settings))
    breaker = YeastarCircuitBreaker(redis)  # type: ignore[arg-type]
    await breaker.open(ConnectionState.AUTH_REJECTED, last_errcode=10005)
    http = CountingAuthHttpClient()
    invalid_settings = configured_settings(YEASTAR_BASE_URL="https://pbx.internal?token=secret")
    manager = YeastarTokenManager(
        invalid_settings,
        redis=redis,  # type: ignore[arg-type]
        http_client=http,  # type: ignore[arg-type]
        store=store,
        circuit_breaker=breaker,
    )

    await manager.clear_local_token_state()

    assert await store.read() is None
    assert (await breaker.get_state()).reason == ConnectionState.AUTH_REJECTED  # type: ignore[union-attr]
    assert http.calls == 0


class FailingWriteTokenStore(YeastarTokenStore):
    async def write(self, state: TokenState) -> None:
        del state
        raise RuntimeError("simulated Redis persistence failure")


@pytest.mark.asyncio
async def test_token_persistence_failure_opens_manual_gate() -> None:
    redis = MemoryRedis()
    settings = configured_settings()
    store = FailingWriteTokenStore(
        redis,  # type: ignore[arg-type]
        settings.APP_SECRET_KEY or "",  # type: ignore[arg-type]
        configuration_fingerprint=settings.yeastar_configuration_fingerprint,
    )
    breaker = YeastarCircuitBreaker(redis)  # type: ignore[arg-type]
    manager = YeastarTokenManager(
        settings,
        redis=redis,  # type: ignore[arg-type]
        http_client=CountingAuthHttpClient(),  # type: ignore[arg-type]
        store=store,
        circuit_breaker=breaker,
    )

    with pytest.raises(YeastarTokenStateError):
        await manager.get_access_token(AuthTrigger.MANUAL_CONNECTION_TEST)

    circuit = await breaker.get_state()
    assert circuit is not None
    assert circuit.reason == ConnectionState.TOKEN_REFRESH_FAILED
