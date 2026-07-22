from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from app.models.enums import RunStatus
from app.services.yeastar.synchronization import synchronize_operators


class _ScalarRows:
    def __init__(self, rows: list[object]) -> None:
        self._rows = rows

    def all(self) -> list[object]:
        return self._rows


class _SerializedRedis:
    """Redis-lock double that proves both calls overlap before the first may finish."""

    def __init__(self) -> None:
        self.guard = asyncio.Lock()
        self.attempts = 0
        self.two_attempts = asyncio.Event()

    def lock(self, *_: object, **__: object) -> "_LockContext":
        return _LockContext(self)


class _LockContext:
    def __init__(self, owner: _SerializedRedis) -> None:
        self.owner = owner

    async def __aenter__(self) -> "_LockContext":
        self.owner.attempts += 1
        if self.owner.attempts >= 2:
            self.owner.two_attempts.set()
        await self.owner.guard.acquire()
        return self

    async def __aexit__(self, *_: object) -> None:
        self.owner.guard.release()


class _StatefulSession:
    """Shared persistence double with a unique SyncRun slot."""

    def __init__(self, run: object | None = None) -> None:
        self.run = run
        self.operators: list[object] = []
        self.sync_run_adds = 0
        self.flush_calls = 0
        self.commit_calls = 0
        self.rollback_calls = 0

    async def scalar(self, _statement: object) -> object | None:
        return self.run

    async def scalars(self, _statement: object) -> _ScalarRows:
        return _ScalarRows(list(self.operators))

    def add(self, item: object) -> None:
        if hasattr(item, "idempotency_key"):
            if self.run is not None and self.run is not item:
                raise AssertionError("duplicate same-minute SyncRun insert")
            self.run = item
            self.sync_run_adds += 1
        elif hasattr(item, "yeastar_extension_id") and item not in self.operators:
            self.operators.append(item)

    async def flush(self) -> None:
        self.flush_calls += 1

    async def commit(self) -> None:
        self.commit_calls += 1

    async def rollback(self) -> None:
        self.rollback_calls += 1


@pytest.mark.asyncio
async def test_completed_same_bucket_sync_returns_saved_result_without_reprocessing() -> None:
    completed_at = datetime(2026, 7, 15, 10, 0, tzinfo=UTC)
    existing = SimpleNamespace(
        status=RunStatus.COMPLETED,
        records_seen=2,
        records_created=2,
        records_updated=0,
        completed_at=completed_at,
    )
    redis = _SerializedRedis()
    session = _StatefulSession(existing)
    client = SimpleNamespace(redis=redis, list_extensions=AsyncMock())

    result = await synchronize_operators(session, client)

    assert (result.created, result.updated, result.total) == (2, 0, 2)
    assert result.synchronized_at == completed_at
    client.list_extensions.assert_not_awaited()
    assert session.sync_run_adds == 0
    assert session.flush_calls == 0
    assert session.commit_calls == 0


@pytest.mark.asyncio
async def test_concurrent_same_bucket_sync_is_serialized_and_calls_provider_once() -> None:
    redis = _SerializedRedis()
    session = _StatefulSession()

    async def list_extensions() -> list[dict[str, str]]:
        await asyncio.wait_for(redis.two_attempts.wait(), timeout=1)
        return [
            {
                "id": "101",
                "number": "2001",
                "caller_id_name": "Operator One",
                "email_addr": "operator@example.test",
            }
        ]

    provider_call = AsyncMock(side_effect=list_extensions)
    client = SimpleNamespace(redis=redis, list_extensions=provider_call)

    first, second = await asyncio.gather(
        synchronize_operators(session, client),
        synchronize_operators(session, client),
    )

    assert redis.attempts == 2
    assert provider_call.await_count == 1
    assert session.sync_run_adds == 1
    assert session.run.status == RunStatus.COMPLETED
    assert session.rollback_calls == 0
    assert {(first.created, first.updated, first.total), (second.created, second.updated, second.total)} == {
        (1, 0, 1)
    }
