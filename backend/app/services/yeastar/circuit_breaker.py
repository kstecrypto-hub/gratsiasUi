from __future__ import annotations

from datetime import UTC, datetime

from redis.asyncio import Redis

from app.services.yeastar.errors import YeastarCircuitOpenError, YeastarError
from app.services.yeastar.schemas import CircuitBreakerState, ConnectionState


YEASTAR_CIRCUIT_KEY = "yeastar:auth:circuit:v1"


class YeastarCircuitBreaker:
    def __init__(self, redis: Redis, *, key: str = YEASTAR_CIRCUIT_KEY) -> None:
        self.redis = redis
        self.key = key

    async def get_state(self) -> CircuitBreakerState | None:
        raw = await self.redis.get(self.key)
        if raw is None:
            return None
        try:
            return CircuitBreakerState.model_validate_json(raw)
        except (ValueError, TypeError):
            # Corrupt safety state must never be interpreted as a closed
            # circuit: doing so could resume credential attempts after an IP
            # block or authentication rejection. Replace the unreadable value
            # with a valid, persistent manual gate. A deliberate connection
            # test can still override and close it after a successful check.
            return await self.open(ConnectionState.TOKEN_REFRESH_FAILED)

    async def is_open(self) -> bool:
        return await self.get_state() is not None

    async def assert_closed(self, *, manual_override: bool = False) -> None:
        state = await self.get_state()
        if state is not None and not manual_override:
            raise YeastarCircuitOpenError(state.reason, state.last_errcode)

    async def open(
        self,
        reason: ConnectionState,
        *,
        last_errcode: int | None = None,
    ) -> CircuitBreakerState:
        state = CircuitBreakerState(
            opened_at=datetime.now(UTC),
            reason=reason,
            last_errcode=last_errcode,
            manual_reset_required=True,
        )
        await self.redis.set(
            self.key,
            state.model_dump_json(),
        )
        return state

    async def open_for_error(self, error: YeastarError) -> CircuitBreakerState:
        return await self.open(
            error.connection_state,
            last_errcode=error.errcode,
        )

    async def close(self) -> None:
        await self.redis.delete(self.key)

    async def get_connection_state(self, *, configured: bool) -> ConnectionState:
        if not configured:
            return ConnectionState.NOT_CONFIGURED
        state = await self.get_state()
        return state.reason if state is not None else ConnectionState.NOT_TESTED


__all__ = ["YEASTAR_CIRCUIT_KEY", "YeastarCircuitBreaker"]
