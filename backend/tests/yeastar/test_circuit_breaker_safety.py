from __future__ import annotations

import pytest

from app.services.yeastar.circuit_breaker import (
    YEASTAR_CIRCUIT_KEY,
    YeastarCircuitBreaker,
)
from app.services.yeastar.errors import YeastarCircuitOpenError
from app.services.yeastar.schemas import CircuitBreakerState, ConnectionState


class MemoryRedis:
    def __init__(self, initial: str) -> None:
        self.values: dict[str, str] = {YEASTAR_CIRCUIT_KEY: initial}

    async def get(self, key: str) -> str | None:
        return self.values.get(key)

    async def set(self, key: str, value: str) -> bool:
        self.values[key] = value
        return True

    async def delete(self, key: str) -> int:
        return int(self.values.pop(key, None) is not None)


@pytest.mark.asyncio
@pytest.mark.parametrize("corrupt_value", ["not-json", "{}", '{"state":"closed"}'])
async def test_corrupt_circuit_state_is_replaced_by_durable_manual_gate(
    corrupt_value: str,
) -> None:
    redis = MemoryRedis(corrupt_value)
    breaker = YeastarCircuitBreaker(redis)  # type: ignore[arg-type]

    recovered = await breaker.get_state()

    assert recovered is not None
    assert recovered.state == "open"
    assert recovered.reason == ConnectionState.TOKEN_REFRESH_FAILED
    assert recovered.last_errcode is None
    assert recovered.manual_reset_required is True

    persisted_raw = redis.values[YEASTAR_CIRCUIT_KEY]
    assert persisted_raw != corrupt_value
    persisted = CircuitBreakerState.model_validate_json(persisted_raw)
    assert persisted == recovered

    # A second reader sees the same durable gate rather than reopening it with
    # a new timestamp or treating corruption as a closed circuit.
    assert await breaker.get_state() == recovered
    assert await breaker.is_open() is True
    with pytest.raises(YeastarCircuitOpenError) as error:
        await breaker.assert_closed()
    assert error.value.connection_state == ConnectionState.TOKEN_REFRESH_FAILED

    # Only the explicit manual-test path may bypass this fail-closed marker.
    await breaker.assert_closed(manual_override=True)
