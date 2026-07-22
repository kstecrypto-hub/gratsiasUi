from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta

from redis.asyncio import Redis

from app.core.config import Settings, get_settings
from app.core.logging import register_secret
from app.core.redis import get_redis
from app.services.yeastar.circuit_breaker import YeastarCircuitBreaker
from app.services.yeastar.errors import (
    YeastarAuthenticationRequiredError,
    YeastarCircuitOpenError,
    YeastarConfigurationError,
    YeastarError,
    YeastarLockTimeoutError,
    YeastarTokenRefreshError,
    YeastarTokenStateError,
)
from app.services.yeastar.http_client import YeastarHttpClient
from app.services.yeastar.schemas import (
    AuthTrigger,
    CircuitBreakerState,
    ConnectionState,
    TokenResponse,
    TokenState,
)
from app.services.yeastar.token_store import (
    YEASTAR_TOKEN_STATE_KEY,
    RedisOwnerLock,
    YeastarTokenStore,
)


class YeastarTokenManager:
    """Shared, lazy Yeastar token lifecycle backed by encrypted Redis state."""

    def __init__(
        self,
        settings: Settings | None = None,
        redis: Redis | None = None,
        http_client: YeastarHttpClient | None = None,
        store: YeastarTokenStore | None = None,
        circuit_breaker: YeastarCircuitBreaker | None = None,
    ) -> None:
        self.settings = settings if settings is not None else get_settings()
        self.redis = redis if redis is not None else get_redis()
        self.http_client = (
            http_client if http_client is not None else YeastarHttpClient(self.settings)
        )
        self._owns_http_client = http_client is None
        self._store = store
        self.circuit_breaker = (
            circuit_breaker
            if circuit_breaker is not None
            else YeastarCircuitBreaker(self.redis)
        )

    def _get_store(self) -> YeastarTokenStore:
        """Return encrypted local storage without requiring a valid PBX config."""
        if not self.settings.APP_SECRET_KEY:
            raise YeastarConfigurationError(
                "Application security must be configured before connecting."
            )
        if self._store is None:
            self._store = YeastarTokenStore(
                self.redis,
                self.settings.APP_SECRET_KEY,
                configuration_fingerprint=self.settings.yeastar_configuration_fingerprint,
                lock_timeout_seconds=self.settings.yeastar_token_lock_timeout_seconds,
                lock_wait_seconds=self.settings.yeastar_token_lock_wait_seconds,
            )
        return self._store

    def _require_connection_configuration(self) -> None:
        errors = self.settings.yeastar_configuration_errors
        if errors:
            raise YeastarConfigurationError(errors[0]["message"])

    async def _clear_if_configuration_changed(
        self, store: YeastarTokenStore
    ) -> bool:
        current = await store.read()
        if current is None or self._fingerprint_matches(current):
            return False
        async with store.owner_lock():
            current = await store.read()
            if current is None or self._fingerprint_matches(current):
                return False
            await store.clear()
            await self.circuit_breaker.open(ConnectionState.NOT_TESTED)
            return True

    async def __aenter__(self) -> "YeastarTokenManager":
        return self

    async def __aexit__(self, *_: object) -> None:
        await self.aclose()

    async def aclose(self) -> None:
        if self._owns_http_client:
            await self.http_client.aclose()

    @staticmethod
    def _auth_path(base_path: str, endpoint: str) -> str:
        return f"{base_path.rstrip('/')}/{endpoint.lstrip('/')}"

    def _access_is_usable(self, state: TokenState, now: datetime) -> bool:
        return state.access_token_expires_at > now + timedelta(
            seconds=self.settings.yeastar_token_refresh_skew_seconds
        )

    @staticmethod
    def _refresh_is_usable(state: TokenState, now: datetime) -> bool:
        return bool(state.refresh_token_value) and state.refresh_token_expires_at > now

    def _fingerprint_matches(self, state: TokenState) -> bool:
        expected = self.settings.yeastar_configuration_fingerprint
        return bool(expected) and state.configuration_fingerprint == expected

    async def get_access_token(self, trigger: AuthTrigger | None = None) -> str:
        store = self._get_store()
        manual = trigger == AuthTrigger.MANUAL_CONNECTION_TEST
        changed = await self._clear_if_configuration_changed(store)
        self._require_connection_configuration()
        if changed and not manual:
            raise YeastarCircuitOpenError(ConnectionState.NOT_TESTED)
        now = datetime.now(UTC)
        state = await store.read()
        await self.circuit_breaker.assert_closed(manual_override=manual)
        if state is not None and self._access_is_usable(state, now):
            return state.access_token_value

        if state is None and trigger is None:
            raise YeastarAuthenticationRequiredError(
                "Test the phone-system connection before using it."
            )

        breaker_snapshot = await self.circuit_breaker.get_state()
        async with store.owner_lock():
            await self._assert_attempt_allowed_after_wait(trigger, breaker_snapshot)
            state = await store.read()
            now = datetime.now(UTC)
            if state is not None and not self._fingerprint_matches(state):
                await store.clear()
                await self.circuit_breaker.open(ConnectionState.NOT_TESTED)
                state = None
                if not manual:
                    raise YeastarCircuitOpenError(ConnectionState.NOT_TESTED)
            if state is not None and self._access_is_usable(state, now):
                return state.access_token_value
            if state is not None and self._refresh_is_usable(state, now):
                return await self._refresh_locked(store, state)
            effective_trigger = (
                trigger
                if state is None
                else AuthTrigger.TOKEN_RENEWAL
            )
            if effective_trigger is None:
                raise YeastarAuthenticationRequiredError(
                    "Test the phone-system connection before using it."
                )
            return await self._obtain_initial_locked(store, state, effective_trigger)

    async def _assert_attempt_allowed_after_wait(
        self,
        trigger: AuthTrigger | None,
        snapshot: CircuitBreakerState | None,
    ) -> None:
        current = await self.circuit_breaker.get_state()
        if current is None:
            return
        if trigger != AuthTrigger.MANUAL_CONNECTION_TEST:
            raise YeastarCircuitOpenError(current.reason, current.last_errcode)
        if snapshot is None or current.opened_at != snapshot.opened_at:
            raise YeastarCircuitOpenError(current.reason, current.last_errcode)

    async def obtain_initial_token(self, trigger: AuthTrigger) -> str:
        if trigger not in {
            AuthTrigger.MANUAL_CONNECTION_TEST,
            AuthTrigger.OPERATOR_SYNC,
            AuthTrigger.CALL_ANALYSIS,
            AuthTrigger.TOKEN_RENEWAL,
        }:
            raise YeastarConfigurationError("Authentication trigger is invalid.")
        store = self._get_store()
        changed = await self._clear_if_configuration_changed(store)
        self._require_connection_configuration()
        if changed and trigger != AuthTrigger.MANUAL_CONNECTION_TEST:
            raise YeastarCircuitOpenError(ConnectionState.NOT_TESTED)
        snapshot = await self.circuit_breaker.get_state()
        if snapshot is not None and trigger != AuthTrigger.MANUAL_CONNECTION_TEST:
            raise YeastarCircuitOpenError(snapshot.reason, snapshot.last_errcode)
        async with store.owner_lock():
            await self._assert_attempt_allowed_after_wait(trigger, snapshot)
            state = await store.read()
            now = datetime.now(UTC)
            if state is not None and self._fingerprint_matches(state):
                if self._access_is_usable(state, now):
                    return state.access_token_value
                if self._refresh_is_usable(state, now):
                    return await self._refresh_locked(store, state)
            elif state is not None:
                await store.clear()
                await self.circuit_breaker.open(ConnectionState.NOT_TESTED)
                state = None
                if trigger != AuthTrigger.MANUAL_CONNECTION_TEST:
                    raise YeastarCircuitOpenError(ConnectionState.NOT_TESTED)
            return await self._obtain_initial_locked(store, state, trigger)

    async def _obtain_initial_locked(
        self,
        store: YeastarTokenStore,
        previous: TokenState | None,
        trigger: AuthTrigger,
    ) -> str:
        try:
            response = await self.http_client.request_model(
                TokenResponse,
                "POST",
                self._auth_path(self.settings.YEASTAR_AUTH_API_PATH, "get_token"),
                json_body={
                    "username": str(self.settings.YEASTAR_CLIENT_ID),
                    "password": str(self.settings.YEASTAR_CLIENT_SECRET),
                },
                transient_retry=False,
            )
        except YeastarError as exc:
            await self.circuit_breaker.open_for_error(exc)
            raise
        state = self._state_from_response(response, previous)
        await self._persist_state(store, state)
        return state.access_token_value

    async def refresh_access_token(self) -> str:
        store = self._get_store()
        changed = await self._clear_if_configuration_changed(store)
        self._require_connection_configuration()
        if changed:
            raise YeastarCircuitOpenError(ConnectionState.NOT_TESTED)
        await self.circuit_breaker.assert_closed()
        async with store.owner_lock():
            await self.circuit_breaker.assert_closed()
            state = await store.read()
            now = datetime.now(UTC)
            if state is not None and not self._fingerprint_matches(state):
                await store.clear()
                await self.circuit_breaker.open(ConnectionState.NOT_TESTED)
                raise YeastarCircuitOpenError(ConnectionState.NOT_TESTED)
            if state is not None and self._access_is_usable(state, now):
                return state.access_token_value
            if state is not None and self._refresh_is_usable(state, now):
                return await self._refresh_locked(store, state)
            if state is None:
                raise YeastarAuthenticationRequiredError(
                    "Test the phone-system connection before using it."
                )
            return await self._obtain_initial_locked(
                store, state, AuthTrigger.TOKEN_RENEWAL
            )

    async def _refresh_locked(
        self,
        store: YeastarTokenStore,
        previous: TokenState,
    ) -> str:
        try:
            response = await self.http_client.request_model(
                TokenResponse,
                "POST",
                self._auth_path(self.settings.YEASTAR_AUTH_API_PATH, "refresh_token"),
                json_body={"refresh_token": previous.refresh_token_value},
                transient_retry=False,
            )
        except YeastarError as exc:
            reason = (
                exc.connection_state
                if exc.opens_circuit
                else ConnectionState.TOKEN_REFRESH_FAILED
            )
            await self.circuit_breaker.open(reason, last_errcode=exc.errcode)
            if exc.opens_circuit:
                raise
            raise YeastarTokenRefreshError(
                "The phone-system session could not be renewed.", exc.errcode
            ) from exc
        state = self._state_from_response(response, previous)
        await self._persist_state(store, state)
        return state.access_token_value

    async def _persist_state(
        self,
        store: YeastarTokenStore,
        state: TokenState,
    ) -> None:
        write_task = asyncio.create_task(store.write(state))
        try:
            await asyncio.shield(write_task)
        except asyncio.CancelledError:
            # Once a provider response contains a new refresh token, complete
            # its durable write before honoring caller cancellation. Otherwise
            # a superseded refresh token may remain in Redis.
            try:
                await asyncio.shield(write_task)
            except Exception:
                await self._open_persistence_failure_circuit()
            raise
        except Exception as exc:
            # Once Yeastar has issued a token pair, losing the latest refresh
            # token can consume the provider token allowance or cause reuse of
            # a superseded token. Stop all automatic auth until an operator
            # explicitly tests the connection again.
            await self._open_persistence_failure_circuit()
            raise YeastarTokenStateError(
                "The phone-system session could not be stored safely."
            ) from exc

    async def _open_persistence_failure_circuit(self) -> None:
        try:
            await self.circuit_breaker.open(ConnectionState.TOKEN_REFRESH_FAILED)
        except Exception:
            pass

    def _state_from_response(
        self,
        response: TokenResponse,
        previous: TokenState | None,
    ) -> TokenState:
        issued_at = datetime.now(UTC)
        register_secret(response.access_token.get_secret_value())
        register_secret(response.refresh_token.get_secret_value())
        return TokenState(
            access_token=response.access_token,
            access_token_expires_at=issued_at
            + timedelta(seconds=response.access_token_expire_time),
            refresh_token=response.refresh_token,
            refresh_token_expires_at=issued_at
            + timedelta(seconds=response.refresh_token_expire_time),
            issued_at=issued_at,
            generation=(previous.generation + 1 if previous is not None else 1),
            configuration_fingerprint=self.settings.yeastar_configuration_fingerprint,
        )

    async def invalidate_access_token(
        self, expected_access_token: str | None = None
    ) -> bool:
        return await self._get_store().invalidate_access_token(expected_access_token)

    async def _revoke_current_token_locked(self, store: YeastarTokenStore) -> None:
        state = await store.read()
        try:
            self._require_connection_configuration()
            if state is not None and state.access_token_value:
                await self.http_client.request_json(
                    "GET",
                    self._auth_path(self.settings.YEASTAR_AUTH_API_PATH, "del_token"),
                    params={"access_token": state.access_token_value},
                    transient_retry=False,
                )
        finally:
            await store.clear()

    async def revoke_current_token(
        self,
        *,
        owner_lock: RedisOwnerLock | None = None,
    ) -> None:
        """Revoke and clear the shared token under the authentication lock.

        Reset acquires the owner lock before changing its circuit state, then
        passes that same lock here so the revoke does not deadlock by trying to
        acquire the non-reentrant Redis lock a second time.
        """

        store = self._get_store()
        if owner_lock is None:
            async with store.owner_lock():
                await self._revoke_current_token_locked(store)
            return
        if owner_lock.key != store.lock_key or not await owner_lock.owned():
            raise YeastarLockTimeoutError(
                "Phone-system authentication lock ownership was lost."
            )
        await self._revoke_current_token_locked(store)

    async def clear_local_token_state(self) -> None:
        if self.settings.APP_SECRET_KEY:
            store = self._get_store()
            async with store.owner_lock():
                await store.clear()
        else:
            # A reset must remain possible when the KDF secret itself was
            # removed or mistyped. The encrypted value is opaque and can be
            # safely discarded without decrypting it.
            await self.redis.delete(YEASTAR_TOKEN_STATE_KEY)

    async def open_circuit(
        self,
        reason: ConnectionState,
        *,
        last_errcode: int | None = None,
    ) -> None:
        await self.circuit_breaker.open(reason, last_errcode=last_errcode)

    async def mark_connection_successful(self) -> None:
        await self.circuit_breaker.close()

    async def get_connection_state(self) -> ConnectionState:
        try:
            store = self._get_store()
            state = await store.read()
        except YeastarConfigurationError:
            return ConnectionState.NOT_CONFIGURED
        if state is not None and not self._fingerprint_matches(state):
            async with store.owner_lock():
                current = await store.read()
                if current is not None and not self._fingerprint_matches(current):
                    await store.clear()
                    await self.circuit_breaker.open(ConnectionState.NOT_TESTED)
            state = None
        if not self.settings.yeastar_configured:
            return ConnectionState.NOT_CONFIGURED
        circuit = await self.circuit_breaker.get_state()
        if circuit is not None:
            return circuit.reason
        if state is not None and self._fingerprint_matches(state):
            return ConnectionState.CONNECTED
        return ConnectionState.NOT_TESTED


__all__ = ["YeastarTokenManager"]
