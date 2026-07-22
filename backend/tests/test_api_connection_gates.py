from __future__ import annotations

from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from uuid import uuid4

import pytest
from fastapi import HTTPException, Request

import app.api.jobs as jobs_api
import app.api.operators as operators_api
from app.core.config import Settings
from app.models import IntegrationStatus
from app.models.enums import YeastarConnectionStatus
from app.schemas.jobs import JobCreate
from app.services.yeastar.circuit_breaker import YeastarCircuitBreaker
from app.services.yeastar.schemas import ConnectionState


def _settings() -> Settings:
    return Settings(
        APP_ENV="test",
        APP_SECRET_KEY="connection-gate-test-key-with-at-least-32-characters",
        YEASTAR_BASE_URL="https://pbx.example.test",
        YEASTAR_CLIENT_ID="client-id",
        YEASTAR_CLIENT_SECRET="client-secret",
        OPENAI_API_KEY="openai-test-key",
    )


def _request(path: str) -> Request:
    return Request(
        {
            "type": "http",
            "method": "POST",
            "path": path,
            "headers": [],
            "client": ("test", 1234),
        }
    )


class _GateSession:
    def __init__(self, integration: IntegrationStatus) -> None:
        self.integration = integration
        self.scalar_calls = 0
        self.commit_calls = 0

    async def scalar(self, _statement: object) -> IntegrationStatus:
        self.scalar_calls += 1
        return self.integration

    async def commit(self) -> None:
        self.commit_calls += 1


class _Redis:
    def __init__(self) -> None:
        self.values: dict[str, str] = {}

    async def get(self, key: str) -> str | None:
        return self.values.get(key)

    async def set(self, key: str, value: str) -> None:
        self.values[key] = value


class _ForbiddenYeastarClient:
    def __init__(self, *_args: object, **_kwargs: object) -> None:
        raise AssertionError("The provider client must not be built before the connection gate passes.")


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("connection_status", "fingerprint_matches", "circuit_open"),
    [
        (YeastarConnectionStatus.NOT_TESTED, True, False),
        (YeastarConnectionStatus.CONNECTED, False, False),
        (YeastarConnectionStatus.CONNECTED, True, True),
    ],
)
async def test_operator_sync_requires_connected_matching_configuration(
    monkeypatch: pytest.MonkeyPatch,
    connection_status: YeastarConnectionStatus,
    fingerprint_matches: bool,
    circuit_open: bool,
) -> None:
    settings = _settings()
    integration = IntegrationStatus(
        provider="yeastar",
        status=connection_status,
        configuration_fingerprint=(
            settings.yeastar_configuration_fingerprint
            if fingerprint_matches
            else "stale-fingerprint"
        ),
        capabilities_json={},
    )
    session = _GateSession(integration)
    redis = _Redis()
    if circuit_open:
        await YeastarCircuitBreaker(redis).open(ConnectionState.AUTH_REJECTED)  # type: ignore[arg-type]

    async def reconcile(*_args: object, **_kwargs: object) -> tuple[IntegrationStatus, bool, bool]:
        return integration, True, False

    monkeypatch.setattr(operators_api, "reconcile_configuration_fingerprint", reconcile)
    monkeypatch.setattr(operators_api, "get_redis", lambda: redis)
    monkeypatch.setattr(operators_api, "YeastarClient", _ForbiddenYeastarClient)

    with pytest.raises(HTTPException) as raised:
        await operators_api.sync_operators(
            request=_request("/api/operators/sync"),
            user=SimpleNamespace(id=uuid4()),
            db=session,  # type: ignore[arg-type]
            settings=settings,
        )

    assert raised.value.status_code == 409
    assert raised.value.detail == (
        "Phone-system connection is not ready. Test the connection in Settings."
    )
    assert session.commit_calls == 1


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("connection_status", "fingerprint_matches", "circuit_open", "expected_detail"),
    [
        (
            YeastarConnectionStatus.NOT_TESTED,
            True,
            False,
            "Phone-system connection is not ready. Test the connection in Settings.",
        ),
        (
            YeastarConnectionStatus.CONNECTED,
            False,
            False,
            "Phone-system settings changed. Test the connection in Settings.",
        ),
        (
            YeastarConnectionStatus.CONNECTED,
            True,
            True,
            "Phone-system connection is not ready. Test the connection in Settings.",
        ),
    ],
)
async def test_job_creation_requires_connected_matching_configuration(
    monkeypatch: pytest.MonkeyPatch,
    connection_status: YeastarConnectionStatus,
    fingerprint_matches: bool,
    circuit_open: bool,
    expected_detail: str,
) -> None:
    settings = _settings()
    integration = IntegrationStatus(
        provider="yeastar",
        status=connection_status,
        configuration_fingerprint=(
            settings.yeastar_configuration_fingerprint
            if fingerprint_matches
            else "stale-fingerprint"
        ),
        capabilities_json={},
    )
    session = _GateSession(integration)
    redis = _Redis()
    if circuit_open:
        await YeastarCircuitBreaker(redis).open(ConnectionState.AUTH_REJECTED)  # type: ignore[arg-type]
    monkeypatch.setattr(jobs_api, "get_redis", lambda: redis)
    now = datetime.now(UTC)

    with pytest.raises(HTTPException) as raised:
        await jobs_api.create_job(
            payload=JobCreate(
                date_from=now - timedelta(hours=1),
                date_to=now,
                operator_ids=[uuid4()],
            ),
            request=_request("/api/jobs"),
            user=SimpleNamespace(id=uuid4()),
            db=session,  # type: ignore[arg-type]
            settings=settings,
        )

    assert raised.value.status_code == 409
    assert raised.value.detail == expected_detail
    assert session.scalar_calls == 1
    assert session.commit_calls == 0
