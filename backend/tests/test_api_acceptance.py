from __future__ import annotations

import csv
import io
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path
from types import SimpleNamespace
from uuid import UUID, uuid4

import fakeredis.aioredis
import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

import app.api.auth as auth_api
import app.api.dependencies as api_dependencies
import app.api.health as health_api
import app.api.jobs as jobs_api
import app.api.results as results_api
import app.api.settings as settings_api
import app.auth.sessions as sessions_module
import app.workers.pipeline as pipeline_module
from app.auth.passwords import hash_password
from app.auth.sessions import SESSION_PREFIX
from app.core.config import Settings, get_settings
from app.core.middleware import CSRFMiddleware, SecurityHeadersMiddleware
from app.database.base import Base
from app.database.session import get_db
from app.models import (
    Call,
    CallParticipant,
    Keyword,
    KeywordCategory,
    KeywordMatch,
    IntegrationStatus,
    Operator,
    ProcessingJob,
    ProcessingJobItem,
    Recording,
    Transcript,
    TranscriptSegment,
    User,
)
from app.models.enums import (
    Direction,
    ItemStatus,
    JobStatus,
    MatchMethod,
    ParticipantRole,
    RecordingStatus,
    SpeakerSource,
    TranscriptStatus,
    YeastarConnectionStatus,
)
from app.services.audio import AudioProcessor
from app.services.yeastar.circuit_breaker import YEASTAR_CIRCUIT_KEY
from app.services.yeastar.configuration_store import YEASTAR_CONFIGURATION_STATE_KEY
from app.services.transcription.configuration_store import OPENAI_CONFIGURATION_STATE_KEY
from app.services.yeastar.errors import YeastarLockTimeoutError
from app.services.yeastar.schemas import CircuitBreakerState, ConnectionState
from app.services.yeastar.token_store import (
    YEASTAR_TOKEN_LOCK_KEY,
    YEASTAR_TOKEN_STATE_KEY,
)
from app.workers.pipeline import cleanup_retention_records


ADMIN_EMAIL = "git-safety-test@example.com"
ADMIN_PASSWORD = "correct horse battery staple"


@dataclass
class APIHarness:
    client: AsyncClient
    sessions: async_sessionmaker[AsyncSession]
    settings: Settings
    redis: fakeredis.aioredis.FakeRedis
    admin_id: UUID
    csrf_token: str | None = None

    async def login(self) -> str:
        csrf_response = await self.client.get("/api/auth/csrf")
        assert csrf_response.status_code == 200
        preauth_token = csrf_response.json()["csrf_token"]
        response = await self.client.post(
            "/api/auth/login",
            json={"email": ADMIN_EMAIL, "password": ADMIN_PASSWORD},
            headers={"X-CSRF-Token": preauth_token},
        )
        assert response.status_code == 200, response.text
        self.csrf_token = response.json()["csrf_token"]
        return self.csrf_token

    def csrf_headers(self) -> dict[str, str]:
        assert self.csrf_token is not None
        return {"X-CSRF-Token": self.csrf_token}


@pytest.fixture
async def api_harness(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    storage_root = tmp_path / "storage"
    storage_root.mkdir()
    database_path = (tmp_path / "acceptance.sqlite3").as_posix()
    settings = Settings(
        APP_ENV="test",
        APP_SECRET_KEY="acceptance-test-secret-key-with-at-least-32-characters",
        APP_TIMEZONE="Europe/Athens",
        SECURE_COOKIES=True,
        STORAGE_ROOT=storage_root,
        DATABASE_URL=f"sqlite+aiosqlite:///{database_path}",
        REDIS_URL="redis://unused.invalid/0",
        ADMIN_EMAIL=ADMIN_EMAIL,
        ADMIN_PASSWORD=ADMIN_PASSWORD,
        LOGIN_RATE_LIMIT="2/15minutes",
        YEASTAR_BASE_URL="https://pbx.example.test",
        YEASTAR_CLIENT_ID="test-client-id",
        YEASTAR_CLIENT_SECRET="test-client-secret",
        OPENAI_API_KEY="test-openai-key",
        TRANSCRIPT_RETENTION_DAYS=1,
    )
    engine = create_async_engine(settings.DATABASE_URL)
    async with engine.begin() as connection:
        await connection.run_sync(Base.metadata.create_all)
    session_factory = async_sessionmaker(engine, expire_on_commit=False)
    fake_redis = fakeredis.aioredis.FakeRedis(decode_responses=True)

    class FakeredisOwnerLock:
        def __init__(
            self,
            redis: fakeredis.aioredis.FakeRedis,
            key: str = YEASTAR_TOKEN_LOCK_KEY,
            **_kwargs: object,
        ) -> None:
            self.redis = redis
            self.key = key
            self.owner = str(uuid4())
            self.acquired = False

        async def acquire(self) -> bool:
            self.acquired = bool(
                await self.redis.set(self.key, self.owner, nx=True, px=30_000)
            )
            if not self.acquired:
                raise YeastarLockTimeoutError("Test lock is busy.")
            return True

        async def release(self) -> bool:
            if not self.acquired or await self.redis.get(self.key) != self.owner:
                return False
            await self.redis.delete(self.key)
            self.acquired = False
            return True

        async def __aenter__(self) -> "FakeredisOwnerLock":
            await self.acquire()
            return self

        async def __aexit__(self, *_args: object) -> None:
            await self.release()

    monkeypatch.setattr(sessions_module, "get_settings", lambda: settings)
    monkeypatch.setattr(sessions_module, "get_redis", lambda: fake_redis)
    monkeypatch.setattr(auth_api, "get_redis", lambda: fake_redis)
    monkeypatch.setattr(api_dependencies, "get_redis", lambda: fake_redis)
    monkeypatch.setattr(health_api, "get_redis", lambda: fake_redis)
    monkeypatch.setattr(jobs_api, "get_redis", lambda: fake_redis)
    monkeypatch.setattr(settings_api, "get_redis", lambda: fake_redis)
    monkeypatch.setattr(settings_api, "RedisOwnerLock", FakeredisOwnerLock)
    monkeypatch.setattr(
        health_api.celery_app.control,
        "ping",
        lambda *, timeout: [{"private-worker-name": {"ok": "pong"}}],
    )

    test_app = FastAPI()
    test_app.add_middleware(CSRFMiddleware)
    test_app.add_middleware(SecurityHeadersMiddleware, settings=settings)
    for router in (
        auth_api.router,
        health_api.router,
        settings_api.router,
        jobs_api.router,
        results_api.router,
    ):
        test_app.include_router(router, prefix="/api")

    async def override_db():
        async with session_factory() as session:
            try:
                yield session
            except Exception:
                await session.rollback()
                raise

    test_app.dependency_overrides[get_db] = override_db
    test_app.dependency_overrides[get_settings] = lambda: settings

    async with session_factory() as session:
        administrator = User(
            email=ADMIN_EMAIL,
            password_hash=hash_password(ADMIN_PASSWORD),
            is_active=True,
        )
        session.add(administrator)
        await session.commit()
        admin_id = administrator.id

    client = AsyncClient(
        transport=ASGITransport(app=test_app),
        base_url="https://testserver",
        headers={"User-Agent": "acceptance-test"},
    )
    try:
        yield APIHarness(client, session_factory, settings, fake_redis, admin_id)
    finally:
        await client.aclose()
        await fake_redis.aclose()
        await engine.dispose()


@pytest.mark.asyncio
async def test_secure_login_session_csrf_and_logout(api_harness: APIHarness) -> None:
    assert (await api_harness.client.get("/api/auth/me")).status_code == 401

    csrf_response = await api_harness.client.get("/api/auth/csrf")
    assert csrf_response.status_code == 200
    assert csrf_response.headers["cache-control"] == "no-store"
    assert "default-src 'none'" in csrf_response.headers["content-security-policy"]
    preauth_token = csrf_response.json()["csrf_token"]

    login_payload = {"email": ADMIN_EMAIL, "password": ADMIN_PASSWORD}
    assert (
        await api_harness.client.post("/api/auth/login", json=login_payload)
    ).status_code == 403
    assert (
        await api_harness.client.post(
            "/api/auth/login",
            json=login_payload,
            headers={"X-CSRF-Token": "incorrect-token"},
        )
    ).status_code == 403

    login = await api_harness.client.post(
        "/api/auth/login",
        json=login_payload,
        headers={"X-CSRF-Token": preauth_token},
    )
    assert login.status_code == 200
    api_harness.csrf_token = login.json()["csrf_token"]
    assert login.json()["user"] == {"id": str(api_harness.admin_id), "email": ADMIN_EMAIL}
    assert "password" not in login.text.lower()

    cookies = login.headers.get_list("set-cookie")
    session_cookie = next(item for item in cookies if item.startswith("yca_session="))
    csrf_cookie = next(item for item in cookies if item.startswith("yca_csrf="))
    assert "httponly" in session_cookie.lower()
    assert "secure" in session_cookie.lower()
    assert "samesite=strict" in session_cookie.lower()
    assert "httponly" not in csrf_cookie.lower()
    assert "secure" in csrf_cookie.lower()
    assert "samesite=strict" in csrf_cookie.lower()

    session_keys = await api_harness.redis.keys(f"{SESSION_PREFIX}*")
    assert len(session_keys) == 1
    assert 0 < await api_harness.redis.ttl(session_keys[0]) <= api_harness.settings.SESSION_TTL_SECONDS
    assert (await api_harness.client.get("/api/auth/me")).status_code == 200

    # A browser reload loses the in-memory token; /csrf must recover the token
    # bound to the still-valid Redis session so the next mutation can succeed.
    api_harness.csrf_token = None
    recovered = await api_harness.client.get("/api/auth/csrf")
    assert recovered.status_code == 200
    assert recovered.json()["csrf_token"] == login.json()["csrf_token"]
    api_harness.csrf_token = recovered.json()["csrf_token"]

    assert (await api_harness.client.post("/api/auth/logout")).status_code == 403
    logout = await api_harness.client.post(
        "/api/auth/logout", headers=api_harness.csrf_headers()
    )
    assert logout.status_code == 200
    assert await api_harness.redis.keys(f"{SESSION_PREFIX}*") == []
    assert (await api_harness.client.get("/api/auth/me")).status_code == 401


@pytest.mark.asyncio
async def test_yeastar_configuration_put_is_authenticated_csrf_safe_and_redacted(
    api_harness: APIHarness,
) -> None:
    sentinel = "ui-secret-must-never-be-reflected-7f25"
    payload = {
        "Name": "UI PBX",
        "Settings": {
            "BaseUrl": "https://ui-pbx.example.test:8088",
            "ClientId": "ui-client-id-7f25",
            "ClientSecret": sentinel,
            "DateFormat": "MM/dd/yyyy HH:mm:ss",
            "PageSize": 500,
            "IgnoreSslErrors": False,
        },
    }

    preauth = await api_harness.client.get("/api/auth/csrf")
    unauthenticated = await api_harness.client.put(
        "/api/settings/yeastar/configuration",
        headers={"X-CSRF-Token": preauth.json()["csrf_token"]},
        json=payload,
    )
    assert unauthenticated.status_code == 401

    await api_harness.login()
    assert (
        await api_harness.client.put(
            "/api/settings/yeastar/configuration",
            json=payload,
        )
    ).status_code == 403

    malformed = {
        **payload,
        "Settings": {
            **payload["Settings"],
            "ClientSecret": {"sentinel": sentinel},
        },
    }
    rejected = await api_harness.client.put(
        "/api/settings/yeastar/configuration",
        headers=api_harness.csrf_headers(),
        json=malformed,
    )
    assert rejected.status_code == 422
    assert sentinel not in rejected.text
    assert rejected.json()["errors"][0]["field"] == "Settings.ClientSecret"

    wrong_types = {
        **payload,
        "Settings": {
            **payload["Settings"],
            "PageSize": "500",
            "IgnoreSslErrors": "false",
        },
    }
    wrong_type_response = await api_harness.client.put(
        "/api/settings/yeastar/configuration",
        headers=api_harness.csrf_headers(),
        json=wrong_types,
    )
    assert wrong_type_response.status_code == 422
    assert {item["field"] for item in wrong_type_response.json()["errors"]} == {
        "Settings.PageSize",
        "Settings.IgnoreSslErrors",
    }
    assert sentinel not in wrong_type_response.text

    await api_harness.redis.set(YEASTAR_TOKEN_STATE_KEY, "old-token-state")
    saved = await api_harness.client.put(
        "/api/settings/yeastar/configuration",
        headers=api_harness.csrf_headers(),
        json=payload,
    )
    assert saved.status_code == 200, saved.text
    assert saved.json()["valid"] is True
    assert saved.json()["configuration"]["Settings"]["ClientSecret"] == "[REDACTED]"
    assert sentinel not in saved.text

    encrypted = await api_harness.redis.get(YEASTAR_CONFIGURATION_STATE_KEY)
    assert encrypted is not None
    assert sentinel not in str(encrypted)
    assert "ui-client-id-7f25" not in str(encrypted)
    assert await api_harness.redis.get(YEASTAR_TOKEN_STATE_KEY) is None
    assert await api_harness.redis.get(YEASTAR_CIRCUIT_KEY) is not None

    safe_configuration = await api_harness.client.get(
        "/api/settings/yeastar/configuration"
    )
    assert safe_configuration.status_code == 200
    assert safe_configuration.json()["Settings"]["ClientId"] == "[CONFIGURED]"
    assert safe_configuration.json()["Settings"]["ClientSecret"] == "[REDACTED]"
    assert sentinel not in safe_configuration.text


@pytest.mark.asyncio
async def test_openai_configuration_put_is_authenticated_csrf_safe_and_write_only(
    api_harness: APIHarness,
) -> None:
    sentinel = "ui-openai-key-must-never-be-reflected-7f25"
    payload = {"api_key": sentinel}

    unauthenticated = await api_harness.client.get("/api/settings/openai/configuration")
    assert unauthenticated.status_code == 401

    await api_harness.login()
    assert (
        await api_harness.client.put(
            "/api/settings/openai/configuration",
            json=payload,
        )
    ).status_code == 403

    malformed = await api_harness.client.put(
        "/api/settings/openai/configuration",
        headers=api_harness.csrf_headers(),
        json={"api_key": {"sentinel": sentinel}},
    )
    assert malformed.status_code == 422
    assert sentinel not in malformed.text
    assert malformed.json()["errors"] == [
        {"field": "api_key", "message": "Enter a valid OpenAI API key."}
    ]

    saved = await api_harness.client.put(
        "/api/settings/openai/configuration",
        headers=api_harness.csrf_headers(),
        json=payload,
    )
    assert saved.status_code == 200, saved.text
    assert saved.json() == {
        "valid": True,
        "errors": [],
        "configuration": {"api_key": "[CONFIGURED]"},
    }
    assert sentinel not in saved.text

    encrypted = await api_harness.redis.get(OPENAI_CONFIGURATION_STATE_KEY)
    assert encrypted is not None
    assert sentinel not in str(encrypted)

    preserved = await api_harness.client.put(
        "/api/settings/openai/configuration",
        headers=api_harness.csrf_headers(),
        json={"api_key": ""},
    )
    assert preserved.status_code == 200
    assert preserved.json()["configuration"] == {"api_key": "[CONFIGURED]"}

    safe_configuration = await api_harness.client.get(
        "/api/settings/openai/configuration"
    )
    assert safe_configuration.status_code == 200
    assert safe_configuration.json() == {"api_key": "[CONFIGURED]"}
    assert sentinel not in safe_configuration.text


@pytest.mark.asyncio
async def test_expired_session_cookie_does_not_lock_out_a_new_login(
    api_harness: APIHarness,
) -> None:
    await api_harness.login()
    session_keys = await api_harness.redis.keys(f"{SESSION_PREFIX}*")
    assert len(session_keys) == 1
    await api_harness.redis.delete(*session_keys)

    preauth = await api_harness.client.get("/api/auth/csrf")
    assert preauth.status_code == 200
    api_harness.csrf_token = preauth.json()["csrf_token"]
    assert "yca_session" not in api_harness.client.cookies

    login = await api_harness.client.post(
        "/api/auth/login",
        json={"email": ADMIN_EMAIL, "password": ADMIN_PASSWORD},
        headers=api_harness.csrf_headers(),
    )
    assert login.status_code == 200, login.text


@pytest.mark.asyncio
async def test_login_failures_are_rate_limited_by_identity(api_harness: APIHarness) -> None:
    csrf_response = await api_harness.client.get("/api/auth/csrf")
    token = csrf_response.json()["csrf_token"]
    headers = {"X-CSRF-Token": token}
    payload = {"email": ADMIN_EMAIL.upper(), "password": "wrong-password"}

    first = await api_harness.client.post("/api/auth/login", json=payload, headers=headers)
    second = await api_harness.client.post("/api/auth/login", json=payload, headers=headers)
    blocked = await api_harness.client.post("/api/auth/login", json=payload, headers=headers)

    assert first.status_code == second.status_code == 401
    assert first.json() == second.json() == {"detail": "Email or password is incorrect."}
    assert blocked.status_code == 429
    assert blocked.json() == {"detail": "Too many sign-in attempts. Try again later."}
    assert int(blocked.headers["retry-after"]) > 0


@pytest.mark.asyncio
async def test_missing_external_credentials_are_explicit_and_do_not_break_dashboard(
    api_harness: APIHarness,
) -> None:
    api_harness.settings.YEASTAR_BASE_URL = None
    api_harness.settings.YEASTAR_CLIENT_ID = None
    api_harness.settings.YEASTAR_CLIENT_SECRET = None
    api_harness.settings.OPENAI_API_KEY = None

    yeastar = await api_harness.client.get("/api/health/yeastar")
    openai = await api_harness.client.get("/api/health/openai")
    assert yeastar.status_code == 200
    assert yeastar.json() == {
        "status": "not_configured",
        "last_successful_connection_at": None,
    }
    assert openai.status_code == 503
    assert openai.json()["status"] == "not_configured"
    assert openai.json()["configured"] is False

    await api_harness.login()
    configuration = await api_harness.client.get("/api/configuration")
    assert configuration.status_code == 200
    assert configuration.json()["yeastar"]["status"] == "not_configured"
    assert configuration.json()["openai"]["status"] == "Not configured"

    dashboard = await api_harness.client.get("/api/dashboard")
    assert dashboard.status_code == 200
    assert dashboard.json()["has_data"] is False

    create = await api_harness.client.post(
        "/api/jobs",
        headers=api_harness.csrf_headers(),
        json={
            "date_from": "2026-07-15T12:00:00",
            "date_to": "2026-07-15T13:00:00",
            "operator_ids": [str(uuid4())],
        },
    )
    assert create.status_code == 503
    assert "not configured" in create.json()["detail"].lower()
    async with api_harness.sessions() as session:
        assert (await session.scalar(select(func.count()).select_from(ProcessingJob)) or 0) == 0


@pytest.mark.asyncio
async def test_configuration_requires_a_live_processing_worker(
    api_harness: APIHarness, monkeypatch: pytest.MonkeyPatch
) -> None:
    await api_harness.login()

    ready = await api_harness.client.get("/api/configuration")
    assert ready.status_code == 200
    assert ready.json()["processing"] == {
        "configured": True,
        "status": "Ready",
        "message": None,
    }
    assert "private-worker-name" not in ready.text
    assert ready.json()["yeastar"] == {
        "configured": True,
        "status": "not_tested",
        "message": "Test the phone-system connection in Settings.",
    }

    async with api_harness.sessions() as session:
        session.add(
            IntegrationStatus(
                provider="yeastar",
                status=YeastarConnectionStatus.CONNECTED,
                configuration_fingerprint=(
                    api_harness.settings.yeastar_configuration_fingerprint
                ),
                capabilities_json={
                    "extensions": True,
                    "cdr_v2": True,
                    "recordings": True,
                },
            )
        )
        await session.commit()
    connected = await api_harness.client.get("/api/configuration")
    assert connected.json()["yeastar"] == {
        "configured": True,
        "status": "connected",
        "message": None,
    }

    await api_harness.redis.set(
        YEASTAR_CIRCUIT_KEY,
        CircuitBreakerState(
            opened_at=datetime.now(UTC),
            reason=ConnectionState.AUTH_REJECTED,
            last_errcode=10005,
        ).model_dump_json(),
    )
    blocked_configuration = await api_harness.client.get("/api/configuration")
    blocked_health = await api_harness.client.get("/api/health/yeastar")
    blocked_status = await api_harness.client.get("/api/settings/yeastar/status")
    assert blocked_configuration.json()["yeastar"]["status"] == "auth_rejected"
    assert blocked_health.json()["status"] == "auth_rejected"
    assert blocked_status.json()["status"] == "auth_rejected"
    assert blocked_status.json()["last_error_reference"] == "YS-10005"
    await api_harness.redis.delete(YEASTAR_CIRCUIT_KEY)

    monkeypatch.setattr(
        health_api.celery_app.control,
        "ping",
        lambda *, timeout: [],
    )
    unavailable = await api_harness.client.get("/api/configuration")
    assert unavailable.status_code == 200
    assert unavailable.json()["processing"] == {
        "configured": False,
        "status": "Unavailable",
        "message": "The background processing service is unavailable.",
    }


@pytest.mark.asyncio
async def test_dashboard_has_a_true_empty_business_state(api_harness: APIHarness) -> None:
    await api_harness.login()

    response = await api_harness.client.get("/api/dashboard")

    assert response.status_code == 200
    assert response.json() == {
        "has_data": False,
        "calls_analyzed": 0,
        "calls_with_recordings": 0,
        "calls_transcribed": 0,
        "calls_with_matches": 0,
        "failed_calls": 0,
        "processing_jobs": 0,
        "results_by_operator": [],
        "results_by_keyword_category": [],
        "recent_jobs": [],
    }
    current = await api_harness.client.get("/api/jobs/current")
    assert current.status_code == 200
    assert current.json() is None


class CapturingTask:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def delay(self, identifier: str) -> SimpleNamespace:
        self.calls.append(identifier)
        return SimpleNamespace(id=f"test-task-{len(self.calls)}")


async def _operator(session: AsyncSession, name: str = "Operator One") -> Operator:
    operator = Operator(
        yeastar_extension_id=f"ext-{uuid4()}",
        extension_number=str(uuid4().int)[-6:],
        display_name=name,
        enabled=True,
        last_synced_at=datetime.now(UTC),
    )
    session.add(operator)
    await session.flush()
    return operator


async def _call(
    session: AsyncSession,
    *,
    started_at: datetime,
    direction: Direction = Direction.INBOUND,
    caller_number: str = "+302101234567",
    has_recording: bool = False,
) -> Call:
    call = Call(
        yeastar_uid=f"call-{uuid4()}",
        started_at=started_at,
        caller_number=caller_number,
        callee_number="101",
        direction=direction,
        duration_seconds=90,
        has_recording=has_recording,
        processing_status="completed",
    )
    session.add(call)
    await session.flush()
    return call


async def _recording(
    session: AsyncSession,
    call: Call,
    *,
    storage_key: str | None = None,
) -> Recording:
    recording = Recording(
        call_id=call.id,
        yeastar_recording_id=f"recording-{uuid4()}",
        storage_key=storage_key,
        mime_type="audio/wav",
        status=RecordingStatus.COMPLETED,
    )
    session.add(recording)
    await session.flush()
    return recording


@pytest.mark.asyncio
async def test_job_progress_retry_only_failed_items_and_cancel(
    api_harness: APIHarness, monkeypatch: pytest.MonkeyPatch
) -> None:
    now = datetime.now(UTC)
    async with api_harness.sessions() as session:
        operator = await _operator(session)
        completed_call = await _call(session, started_at=now - timedelta(minutes=2))
        failed_call = await _call(session, started_at=now - timedelta(minutes=1), has_recording=True)
        failed_recording = await _recording(session, failed_call)
        retry_job = ProcessingJob(
            idempotency_key=f"retry-job-{uuid4()}",
            requested_by_id=api_harness.admin_id,
            status=JobStatus.COMPLETED_WITH_ERRORS,
            date_from=now - timedelta(hours=1),
            date_to=now,
            selected_operator_ids=[str(operator.id)],
            selected_category_ids=[],
            request_filters={},
            progress_percent=100,
            current_stage="Complete with some errors",
            calls_found=2,
            recordings_found=1,
            calls_completed=1,
            calls_failed=1,
            completed_at=now,
        )
        session.add(retry_job)
        await session.flush()
        completed_item = ProcessingJobItem(
            job_id=retry_job.id,
            call_id=completed_call.id,
            operator_id=operator.id,
            idempotency_key=f"completed-item-{uuid4()}",
            status=ItemStatus.COMPLETED,
            stage="completed",
            completed_at=now,
        )
        failed_item = ProcessingJobItem(
            job_id=retry_job.id,
            call_id=failed_call.id,
            operator_id=operator.id,
            recording_id=failed_recording.id,
            idempotency_key=f"failed-item-{uuid4()}",
            status=ItemStatus.FAILED,
            stage="failed",
            error_category="temporary",
            error_message="Safe failure message.",
            completed_at=now,
        )

        cancel_call = await _call(session, started_at=now)
        cancel_job = ProcessingJob(
            idempotency_key=f"cancel-job-{uuid4()}",
            requested_by_id=api_harness.admin_id,
            status=JobStatus.QUEUED,
            date_from=now - timedelta(hours=1),
            date_to=now,
            selected_operator_ids=[str(operator.id)],
            selected_category_ids=[],
            request_filters={},
            current_stage="Queued",
        )
        session.add(cancel_job)
        await session.flush()
        cancel_item = ProcessingJobItem(
            job_id=cancel_job.id,
            call_id=cancel_call.id,
            operator_id=operator.id,
            idempotency_key=f"cancel-item-{uuid4()}",
            status=ItemStatus.QUEUED,
            stage="queued",
        )
        session.add_all([completed_item, failed_item, cancel_item])
        await session.commit()
        retry_job_id = retry_job.id
        failed_item_id = failed_item.id
        completed_item_id = completed_item.id
        cancel_job_id = cancel_job.id
        cancel_item_id = cancel_item.id

    task = CapturingTask()
    monkeypatch.setattr(jobs_api, "process_job_item", task)
    await api_harness.login()

    before = await api_harness.client.get(f"/api/jobs/{retry_job_id}")
    assert before.status_code == 200
    assert before.json()["progress_percent"] == 100
    assert before.json()["status"] == "completed_with_errors"

    cancelled = await api_harness.client.post(
        f"/api/jobs/{cancel_job_id}/cancel", headers=api_harness.csrf_headers()
    )
    assert cancelled.status_code == 200
    cancel_body = cancelled.json()
    assert cancel_body["status"] == "cancelled"
    assert cancel_body["cancellation_requested"] is True
    assert {item["id"]: item["status"] for item in cancel_body["items"]}[
        str(cancel_item_id)
    ] == "cancelled"
    assert (
        await api_harness.client.post(
            f"/api/jobs/{cancel_job_id}/cancel", headers=api_harness.csrf_headers()
        )
    ).status_code == 409

    retried = await api_harness.client.post(
        f"/api/jobs/{retry_job_id}/retry", headers=api_harness.csrf_headers()
    )
    assert retried.status_code == 202, retried.text
    retry_body = retried.json()
    assert retry_body["status"] == "queued"
    assert retry_body["current_stage"] == "Queued for retry"
    assert retry_body["attempt_count"] == 2
    assert retry_body["progress_percent"] == 50
    assert retry_body["calls_completed"] == 1
    assert retry_body["calls_failed"] == 0
    statuses = {item["id"]: item["status"] for item in retry_body["items"]}
    assert statuses[str(completed_item_id)] == "completed"
    assert statuses[str(failed_item_id)] == "queued"
    assert task.calls == [str(failed_item_id)]


@pytest.mark.asyncio
async def test_call_retry_uses_only_the_latest_parent_job(
    api_harness: APIHarness, monkeypatch: pytest.MonkeyPatch
) -> None:
    now = datetime.now(UTC)
    async with api_harness.sessions() as session:
        operator = await _operator(session)
        call = await _call(session, started_at=now, has_recording=True)
        call.processing_status = "failed"
        recording = await _recording(session, call)
        historical_job = ProcessingJob(
            idempotency_key=f"old-call-retry-{uuid4()}",
            requested_by_id=api_harness.admin_id,
            status=JobStatus.COMPLETED_WITH_ERRORS,
            date_from=now - timedelta(days=2),
            date_to=now - timedelta(days=1),
            selected_operator_ids=[str(operator.id)],
            selected_category_ids=[],
            request_filters={},
            progress_percent=100,
            current_stage="Complete with some errors",
            calls_found=1,
            calls_failed=1,
            completed_at=now - timedelta(days=1),
            created_at=now - timedelta(days=2),
        )
        latest_job = ProcessingJob(
            idempotency_key=f"latest-call-retry-{uuid4()}",
            requested_by_id=api_harness.admin_id,
            status=JobStatus.COMPLETED_WITH_ERRORS,
            date_from=now - timedelta(days=1),
            date_to=now,
            selected_operator_ids=[str(operator.id)],
            selected_category_ids=[],
            request_filters={},
            progress_percent=100,
            current_stage="Complete with some errors",
            calls_found=1,
            calls_failed=1,
            completed_at=now,
            created_at=now - timedelta(days=1),
        )
        current_context_job = ProcessingJob(
            idempotency_key=f"current-context-{uuid4()}",
            requested_by_id=api_harness.admin_id,
            status=JobStatus.COMPLETED,
            date_from=now - timedelta(hours=1),
            date_to=now,
            selected_operator_ids=[str(operator.id)],
            selected_category_ids=[],
            request_filters={},
            progress_percent=100,
            current_stage="Complete",
            completed_at=now,
            created_at=now,
        )
        session.add_all([historical_job, latest_job, current_context_job])
        await session.flush()
        historical_item = ProcessingJobItem(
            job_id=historical_job.id,
            call_id=call.id,
            operator_id=operator.id,
            recording_id=recording.id,
            idempotency_key=f"old-call-item-{uuid4()}",
            status=ItemStatus.FAILED,
            stage="failed",
            completed_at=now - timedelta(days=1),
        )
        latest_item = ProcessingJobItem(
            job_id=latest_job.id,
            call_id=call.id,
            operator_id=operator.id,
            recording_id=recording.id,
            idempotency_key=f"latest-call-item-{uuid4()}",
            status=ItemStatus.FAILED,
            stage="failed",
            completed_at=now,
        )
        session.add_all([historical_item, latest_item])
        await session.commit()
        historical_job_id = historical_job.id
        latest_job_id = latest_job.id
        current_context_job_id = current_context_job.id
        historical_item_id = historical_item.id
        latest_item_id = latest_item.id
        call_id = call.id

    task = CapturingTask()
    monkeypatch.setattr(results_api, "process_job_item", task)
    await api_harness.login()

    response = await api_harness.client.post(
        f"/api/calls/{call_id}/retry", headers=api_harness.csrf_headers()
    )

    assert response.status_code == 202, response.text
    assert task.calls == [str(latest_item_id)]
    current = await api_harness.client.get("/api/jobs/current")
    active = await api_harness.client.get("/api/jobs/active")
    assert current.status_code == 200
    assert active.status_code == 200
    assert current.json()["id"] == str(current_context_job_id)
    assert current.json()["is_current"] is True
    assert active.json()["id"] == str(latest_job_id)
    assert active.json()["is_current"] is False
    async with api_harness.sessions() as session:
        historical_job = await session.get(ProcessingJob, historical_job_id)
        latest_job = await session.get(ProcessingJob, latest_job_id)
        historical_item = await session.get(ProcessingJobItem, historical_item_id)
        latest_item = await session.get(ProcessingJobItem, latest_item_id)
        call = await session.get(Call, call_id)
        assert historical_job is not None
        assert latest_job is not None
        assert historical_item is not None
        assert latest_item is not None
        assert call is not None
        assert historical_job.status == JobStatus.COMPLETED_WITH_ERRORS
        assert historical_item.status == ItemStatus.FAILED
        assert latest_job.status == JobStatus.QUEUED
        assert latest_job.current_stage == "Queued for call retry"
        assert latest_item.status == ItemStatus.QUEUED
        assert call.processing_status == "queued"

    conflict = await api_harness.client.post(
        f"/api/calls/{call_id}/retry", headers=api_harness.csrf_headers()
    )
    assert conflict.status_code == 409
    assert "already running" in conflict.json()["detail"].lower()


@pytest.mark.asyncio
async def test_job_creation_interprets_naive_dates_as_athens_wall_time(
    api_harness: APIHarness, monkeypatch: pytest.MonkeyPatch
) -> None:
    async with api_harness.sessions() as session:
        operator = await _operator(session)
        session.add(
            IntegrationStatus(
                provider="yeastar",
                status=YeastarConnectionStatus.CONNECTED,
                configuration_fingerprint=(
                    api_harness.settings.yeastar_configuration_fingerprint
                ),
                capabilities_json={},
            )
        )
        await session.commit()
        operator_id = operator.id

    task = CapturingTask()
    monkeypatch.setattr(jobs_api, "process_analysis_job", task)
    await api_harness.login()
    response = await api_harness.client.post(
        "/api/jobs",
        headers=api_harness.csrf_headers(),
        json={
            "date_from": "2026-07-15T12:00:00",
            "date_to": "2026-07-15T13:00:00",
            "operator_ids": [str(operator_id)],
            "idempotency_key": "athens-date-test-0001",
        },
    )

    assert response.status_code == 202, response.text
    body = response.json()
    assert datetime.fromisoformat(body["date_from"].replace("Z", "+00:00")) == datetime(
        2026, 7, 15, 9, 0, tzinfo=UTC
    )
    assert datetime.fromisoformat(body["date_to"].replace("Z", "+00:00")) == datetime(
        2026, 7, 15, 10, 0, tzinfo=UTC
    )
    assert task.calls == [body["id"]]
    assert body["is_current"] is True

    current = await api_harness.client.get("/api/jobs/current")
    assert current.status_code == 200
    assert current.json()["id"] == body["id"]
    assert current.json()["is_current"] is True

    duplicate = await api_harness.client.post(
        "/api/jobs",
        headers=api_harness.csrf_headers(),
        json={
            "date_from": "2026-07-15T12:00:00",
            "date_to": "2026-07-15T13:00:00",
            "operator_ids": [str(operator_id)],
            "idempotency_key": "athens-date-test-0001",
        },
    )
    assert duplicate.status_code == 202
    assert duplicate.json()["id"] == body["id"]

    conflicting = await api_harness.client.post(
        "/api/jobs",
        headers=api_harness.csrf_headers(),
        json={
            "date_from": "2026-07-15T13:00:00",
            "date_to": "2026-07-15T14:00:00",
            "operator_ids": [str(operator_id)],
            "idempotency_key": "athens-date-test-0002",
        },
    )
    assert conflicting.status_code == 409
    assert "already running" in conflicting.json()["detail"].lower()


@pytest.mark.asyncio
async def test_results_filters_newest_first_and_csv_is_safe_for_spreadsheets(
    api_harness: APIHarness,
) -> None:
    now = datetime(2026, 7, 15, 12, 0, tzinfo=UTC)
    async with api_harness.sessions() as session:
        formula_operator = await _operator(session, "=HYPERLINK(\"https://invalid\")")
        ordinary_operator = await _operator(session, "Operator Two")
        matched_call = await _call(
            session,
            started_at=now - timedelta(hours=2),
            caller_number="+302101111111",
            has_recording=True,
        )
        newer_call = await _call(
            session,
            started_at=now - timedelta(hours=1),
            direction=Direction.OUTBOUND,
            caller_number="+302102222222",
        )
        newer_call.duration_seconds = 120
        matched_participant = CallParticipant(
            call_id=matched_call.id,
            operator_id=formula_operator.id,
            provider_extension_id=formula_operator.yeastar_extension_id,
            provider_extension_number=formula_operator.extension_number,
            role=ParticipantRole.ANSWERING_OPERATOR,
            answered=True,
            attribution_source=SpeakerSource.YEASTAR_EXTENSION,
        )
        newer_participant = CallParticipant(
            call_id=newer_call.id,
            operator_id=ordinary_operator.id,
            provider_extension_id=ordinary_operator.yeastar_extension_id,
            provider_extension_number=ordinary_operator.extension_number,
            role=ParticipantRole.ANSWERING_OPERATOR,
            answered=True,
            attribution_source=SpeakerSource.YEASTAR_EXTENSION,
        )
        recording = await _recording(session, matched_call)
        category = KeywordCategory(name="+Sensitive category", active=True)
        session.add(category)
        await session.flush()
        keyword = Keyword(
            category_id=category.id,
            canonical_phrase="@dangerous keyword",
            normalized_phrase="dangerous keyword",
            active=True,
        )
        session.add(keyword)
        await session.flush()
        transcript = Transcript(
            call_id=matched_call.id,
            recording_id=recording.id,
            operator_id=formula_operator.id,
            idempotency_key=f"transcript-{uuid4()}",
            status=TranscriptStatus.COMPLETED,
            model="gpt-4o-transcribe",
            language="el",
            completed_at=now,
            source_audio_sha256="a" * 64,
            is_diarized=False,
        )
        direct_search_recording = await _recording(session, newer_call)
        direct_search_transcript = Transcript(
            call_id=newer_call.id,
            recording_id=direct_search_recording.id,
            operator_id=ordinary_operator.id,
            idempotency_key=f"direct-search-transcript-{uuid4()}",
            status=TranscriptStatus.COMPLETED,
            model="gpt-4o-transcribe",
            language="el",
            completed_at=now,
            source_audio_sha256="b" * 64,
            is_diarized=False,
        )
        session.add_all([transcript, direct_search_transcript])
        await session.flush()
        segment = TranscriptSegment(
            transcript_id=transcript.id,
            call_id=matched_call.id,
            operator_id=formula_operator.id,
            speaker_label=formula_operator.display_name,
            speaker_source=SpeakerSource.YEASTAR_EXTENSION,
            start_seconds=Decimal("5.000"),
            end_seconds=Decimal("8.000"),
            original_text="Spreadsheet-shaped content",
            normalized_text="spreadsheet shaped content",
            transcription_model="gpt-4o-transcribe",
            sequence_number=1,
        )
        direct_search_segment = TranscriptSegment(
            transcript_id=direct_search_transcript.id,
            call_id=newer_call.id,
            operator_id=ordinary_operator.id,
            speaker_label=ordinary_operator.display_name,
            speaker_source=SpeakerSource.YEASTAR_EXTENSION,
            start_seconds=Decimal("3.000"),
            end_seconds=Decimal("5.000"),
            original_text="Direct-only phrase",
            normalized_text="direct only phrase",
            transcription_model="gpt-4o-transcribe",
            sequence_number=1,
        )
        session.add_all([segment, direct_search_segment])
        await session.flush()
        match = KeywordMatch(
            keyword_id=keyword.id,
            operator_id=formula_operator.id,
            call_id=matched_call.id,
            transcript_segment_id=segment.id,
            original_matched_text="=1+1",
            normalized_match="1 1",
            context_before="-before",
            context_after="@after",
            start_seconds=Decimal("5.000"),
            end_seconds=Decimal("6.000"),
            match_method=MatchMethod.EXACT_PHRASE,
            match_score=Decimal("1.000"),
        )
        result_job = ProcessingJob(
            idempotency_key=f"results-job-{uuid4()}",
            requested_by_id=api_harness.admin_id,
            status=JobStatus.COMPLETED,
            date_from=now - timedelta(days=1),
            date_to=now + timedelta(days=1),
            selected_operator_ids=[str(formula_operator.id), str(ordinary_operator.id)],
            selected_category_ids=[],
            request_filters={},
            progress_percent=100,
            current_stage="Complete",
            calls_found=2,
            recordings_found=2,
            calls_completed=2,
            completed_at=now,
        )
        session.add(result_job)
        await session.flush()
        result_items = [
            ProcessingJobItem(
                job_id=result_job.id,
                call_id=matched_call.id,
                operator_id=formula_operator.id,
                recording_id=recording.id,
                idempotency_key=f"result-item-{uuid4()}",
                status=ItemStatus.COMPLETED,
                stage="completed",
                completed_at=now,
            ),
            ProcessingJobItem(
                job_id=result_job.id,
                call_id=newer_call.id,
                operator_id=ordinary_operator.id,
                recording_id=direct_search_recording.id,
                idempotency_key=f"result-item-{uuid4()}",
                status=ItemStatus.COMPLETED,
                stage="completed",
                completed_at=now,
            ),
        ]
        session.add_all(
            [matched_participant, newer_participant, match, *result_items]
        )
        await session.commit()
        matched_call_id = matched_call.id
        newer_call_id = newer_call.id
        formula_operator_id = formula_operator.id
        category_id = category.id
        keyword_id = keyword.id
        segment_id = segment.id
        result_job_id = result_job.id

    await api_harness.login()
    unfiltered = await api_harness.client.get("/api/results")
    assert unfiltered.status_code == 200, unfiltered.text
    assert [item["call_id"] for item in unfiltered.json()["items"]] == [
        str(newer_call_id),
        str(matched_call_id),
    ]
    assert {item["job_id"] for item in unfiltered.json()["items"]} == {
        str(result_job_id)
    }
    assert "+302101111111" not in unfiltered.text

    matched = await api_harness.client.get("/api/results", params={"has_matches": "true"})
    unmatched = await api_harness.client.get("/api/results", params={"has_matches": "false"})
    assert [item["call_id"] for item in matched.json()["items"]] == [str(matched_call_id)]
    assert [item["call_id"] for item in unmatched.json()["items"]] == [str(newer_call_id)]

    by_operator = await api_harness.client.get(
        "/api/results", params={"operator_id": str(formula_operator_id)}
    )
    by_category = await api_harness.client.get(
        "/api/results", params={"category_id": str(category_id)}
    )
    by_keyword = await api_harness.client.get(
        "/api/results", params={"keyword_id": str(keyword_id)}
    )
    by_keyword_text = await api_harness.client.get(
        "/api/results", params={"keyword": "dangerous"}
    )
    by_transcript_text = await api_harness.client.get(
        "/api/results", params={"transcript_query": "spreadsheet-shaped"}
    )
    by_direct_transcript_text = await api_harness.client.get(
        "/api/results", params={"transcript_query": "direct-only"}
    )
    outbound = await api_harness.client.get("/api/results", params={"direction": "outbound"})
    after_cutoff = await api_harness.client.get(
        "/api/results",
        params={"date_from": (now - timedelta(minutes=90)).isoformat()},
    )
    athens_day = await api_harness.client.get(
        "/api/results", params={"date_from": "2026-07-15", "date_to": "2026-07-15"}
    )
    assert by_operator.json()["total"] == 1
    assert by_category.json()["total"] == 1
    assert by_keyword.json()["total"] == 1
    assert by_keyword_text.json()["total"] == 1
    assert by_keyword_text.json()["items"][0]["match_count"] == 1
    assert by_keyword_text.json()["items"][0]["keywords_found"] == [
        "@dangerous keyword"
    ]
    assert by_transcript_text.json()["total"] == 1
    assert [item["call_id"] for item in by_transcript_text.json()["items"]] == [
        str(matched_call_id)
    ]
    assert [item["call_id"] for item in by_direct_transcript_text.json()["items"]] == [
        str(newer_call_id)
    ]
    assert by_direct_transcript_text.json()["items"][0]["match_count"] == 0
    assert [item["call_id"] for item in outbound.json()["items"]] == [str(newer_call_id)]
    assert after_cutoff.status_code == 200, after_cutoff.text
    assert [item["call_id"] for item in after_cutoff.json()["items"]] == [str(newer_call_id)]
    assert athens_day.status_code == 200, athens_day.text
    assert athens_day.json()["total"] == 2
    contradictory = await api_harness.client.get(
        "/api/results",
        params={"has_matches": "false", "keyword": "dangerous"},
    )
    assert contradictory.status_code == 422

    detail = await api_harness.client.get(f"/api/calls/{matched_call_id}")
    assert detail.status_code == 200, detail.text
    assert detail.json()["matches"][0]["transcript_segment_id"] == str(segment_id)
    assert detail.json()["transcript_segments"][0]["id"] == str(segment_id)

    operator_ascending = await api_harness.client.get(
        "/api/results", params={"sort": "operator", "order": "asc"}
    )
    duration_descending = await api_harness.client.get(
        "/api/results", params={"sort": "duration_seconds", "order": "desc"}
    )
    matches_descending = await api_harness.client.get(
        "/api/results", params={"sort": "match_count", "order": "desc"}
    )
    assert [item["call_id"] for item in operator_ascending.json()["items"]] == [
        str(matched_call_id),
        str(newer_call_id),
    ]
    assert [item["call_id"] for item in duration_descending.json()["items"]] == [
        str(newer_call_id),
        str(matched_call_id),
    ]
    assert matches_descending.json()["items"][0]["call_id"] == str(matched_call_id)
    assert (
        await api_harness.client.get("/api/results", params={"sort": "not-a-field"})
    ).status_code == 422

    exported = await api_harness.client.get(
        "/api/results/export.csv", params={"has_matches": "true"}
    )
    assert exported.status_code == 200
    assert exported.content.startswith(b"\xef\xbb\xbf")
    rows = list(csv.reader(io.StringIO(exported.content.decode("utf-8-sig"))))
    assert rows[0] == [
        "Date",
        "Time",
        "Operator",
        "Phone number",
        "Duration",
        "Direction",
        "Keyword category",
        "Keyword",
        "Transcript excerpt",
        "Timestamp",
        "Match count",
    ]
    assert len(rows) == 2
    data = rows[1]
    assert data[0:2] == ["2026-07-15", "13:00:00"]
    assert data[2].startswith("'=")
    assert data[3].startswith("'+")
    assert data[6].startswith("'+")
    assert data[7].startswith("'@")
    assert data[8].startswith("'-")


@pytest.mark.asyncio
async def test_current_results_replace_the_view_but_history_remains_searchable(
    api_harness: APIHarness,
) -> None:
    now = datetime.now(UTC)
    async with api_harness.sessions() as session:
        operator = await _operator(session, "History Operator")
        historical_call = await _call(
            session,
            started_at=now - timedelta(days=2),
            caller_number="+302101111111",
        )
        current_call = await _call(
            session,
            started_at=now - timedelta(days=1),
            caller_number="+302102222222",
        )
        historical_job = ProcessingJob(
            idempotency_key=f"historical-results-{uuid4()}",
            requested_by_id=api_harness.admin_id,
            status=JobStatus.COMPLETED_WITH_ERRORS,
            date_from=now - timedelta(days=3),
            date_to=now - timedelta(days=2),
            selected_operator_ids=[str(operator.id)],
            selected_category_ids=[],
            request_filters={},
            progress_percent=100,
            current_stage="Complete with some errors",
            calls_found=1,
            calls_failed=1,
            completed_at=now - timedelta(days=2),
            created_at=now - timedelta(days=2),
        )
        current_job = ProcessingJob(
            idempotency_key=f"current-results-{uuid4()}",
            requested_by_id=api_harness.admin_id,
            status=JobStatus.COMPLETED,
            date_from=now - timedelta(days=2),
            date_to=now,
            selected_operator_ids=[str(operator.id)],
            selected_category_ids=[],
            request_filters={},
            progress_percent=100,
            current_stage="Complete",
            calls_found=1,
            calls_completed=1,
            completed_at=now - timedelta(days=1),
            created_at=now - timedelta(days=1),
        )
        session.add_all([historical_job, current_job])
        await session.flush()
        session.add_all(
            [
                ProcessingJobItem(
                    job_id=historical_job.id,
                    call_id=historical_call.id,
                    operator_id=operator.id,
                    idempotency_key=f"historical-item-{uuid4()}",
                    status=ItemStatus.FAILED,
                    stage="failed",
                    error_category="historical_failure",
                    error_message="A safe historical failure.",
                    completed_at=now - timedelta(days=2),
                ),
                ProcessingJobItem(
                    job_id=current_job.id,
                    call_id=current_call.id,
                    operator_id=operator.id,
                    idempotency_key=f"current-item-{uuid4()}",
                    status=ItemStatus.COMPLETED,
                    stage="completed",
                    completed_at=now - timedelta(days=1),
                ),
            ]
        )
        await session.commit()
        historical_job_id = historical_job.id
        current_job_id = current_job.id
        historical_call_id = historical_call.id
        current_call_id = current_call.id

    await api_harness.login()

    current = await api_harness.client.get("/api/results")
    historical = await api_harness.client.get(
        "/api/results", params={"job_id": str(historical_job_id)}
    )
    jobs = await api_harness.client.get("/api/jobs")

    assert current.status_code == 200
    assert [item["call_id"] for item in current.json()["items"]] == [
        str(current_call_id)
    ]
    assert current.json()["items"][0]["job_id"] == str(current_job_id)
    assert current.json()["items"][0]["processing_status"] == "completed"
    assert historical.status_code == 200
    assert [item["call_id"] for item in historical.json()["items"]] == [
        str(historical_call_id)
    ]
    assert historical.json()["items"][0]["processing_status"] == "failed"
    assert jobs.status_code == 200
    assert [item["id"] for item in jobs.json()["items"]] == [
        str(current_job_id),
        str(historical_job_id),
    ]
    assert [item["is_current"] for item in jobs.json()["items"]] == [True, False]

    exported = await api_harness.client.get(
        "/api/results/export.csv", params={"job_id": str(historical_job_id)}
    )
    assert exported.status_code == 200
    rows = list(csv.reader(io.StringIO(exported.content.decode("utf-8-sig"))))
    assert len(rows) == 2
    assert rows[1][2] == "History Operator"


@pytest.mark.asyncio
async def test_audio_streaming_requires_authentication_and_honors_byte_ranges(
    api_harness: APIHarness,
) -> None:
    payload = b"RIFF-test-audio-bytes"
    relative_key = "recordings/range-test.wav"
    audio_path = api_harness.settings.STORAGE_ROOT / relative_key
    audio_path.parent.mkdir(parents=True)
    audio_path.write_bytes(payload)
    async with api_harness.sessions() as session:
        call = await _call(
            session,
            started_at=datetime.now(UTC),
            has_recording=True,
        )
        recording = await _recording(session, call, storage_key=relative_key)
        recording.mime_type = "application/octet-stream"
        recording.file_extension = ".wav"
        await session.commit()
        call_id = call.id

    assert (await api_harness.client.get(f"/api/calls/{call_id}/audio")).status_code == 401
    await api_harness.login()

    partial = await api_harness.client.get(
        f"/api/calls/{call_id}/audio", headers={"Range": "bytes=5-9"}
    )
    assert partial.status_code == 206
    assert partial.content == payload[5:10]
    assert partial.headers["accept-ranges"] == "bytes"
    assert partial.headers["content-range"] == f"bytes 5-9/{len(payload)}"
    assert partial.headers["content-length"] == "5"
    assert partial.headers["content-type"].startswith("audio/wav")

    suffix = await api_harness.client.get(
        f"/api/calls/{call_id}/audio", headers={"Range": "bytes=-4"}
    )
    assert suffix.status_code == 206
    assert suffix.content == payload[-4:]

    unsatisfiable = await api_harness.client.get(
        f"/api/calls/{call_id}/audio", headers={"Range": "bytes=999-"}
    )
    assert unsatisfiable.status_code == 416
    assert unsatisfiable.headers["content-range"] == f"bytes */{len(payload)}"
    assert unsatisfiable.content == b""


@pytest.mark.asyncio
async def test_retention_cleanup_retries_failed_audio_deletion_on_next_run(
    api_harness: APIHarness, monkeypatch: pytest.MonkeyPatch
) -> None:
    relative_key = "recordings/retention-retry.wav"
    audio_path = api_harness.settings.STORAGE_ROOT / relative_key
    audio_path.parent.mkdir(parents=True)
    audio_path.write_bytes(b"old audio")
    now = datetime.now(UTC)
    async with api_harness.sessions() as session:
        call = await _call(session, started_at=now - timedelta(days=5), has_recording=True)
        recording = await _recording(session, call, storage_key=relative_key)
        transcript = Transcript(
            call_id=call.id,
            recording_id=recording.id,
            operator_id=None,
            idempotency_key=f"retention-transcript-{uuid4()}",
            status=TranscriptStatus.COMPLETED,
            model="gpt-4o-transcribe-diarize",
            language="el",
            completed_at=now - timedelta(days=2),
            source_audio_sha256="b" * 64,
            is_diarized=True,
        )
        session.add(transcript)
        await session.commit()
        recording_id = recording.id

    monkeypatch.setattr(pipeline_module, "AsyncSessionFactory", api_harness.sessions)
    monkeypatch.setattr(pipeline_module, "get_settings", lambda: api_harness.settings)
    attempts: list[Path] = []

    def fail_once_then_remove(
        self: AudioProcessor, paths: list[Path]
    ) -> tuple[list[Path], list[Path]]:
        path_list = list(paths)
        attempts.extend(path_list)
        if len(attempts) == 1:
            return [], path_list
        removed: list[Path] = []
        failed: list[Path] = []
        for path in path_list:
            try:
                path.unlink(missing_ok=True)
                removed.append(path)
            except OSError:
                failed.append(path)
        return removed, failed

    monkeypatch.setattr(AudioProcessor, "remove_files", fail_once_then_remove)

    first = await cleanup_retention_records()
    assert first == {
        "transcripts_deleted": 1,
        "audio_deleted": 0,
        "cleanup_failed": 1,
    }
    assert audio_path.exists()
    async with api_harness.sessions() as session:
        assert (await session.scalar(select(func.count()).select_from(Transcript)) or 0) == 0
        pending = await session.get(Recording, recording_id)
        assert pending is not None
        assert pending.storage_key == relative_key
        assert pending.last_error_category == "retention_cleanup_pending"

    second = await cleanup_retention_records()
    assert second == {
        "transcripts_deleted": 0,
        "audio_deleted": 1,
        "cleanup_failed": 0,
    }
    assert not audio_path.exists()
    assert attempts == [audio_path, audio_path]
    async with api_harness.sessions() as session:
        cleaned = await session.get(Recording, recording_id)
        assert cleaned is not None
        assert cleaned.storage_key is None
        assert cleaned.status == RecordingStatus.DELETED
        assert cleaned.deleted_at is not None
        assert cleaned.last_error_category is None
        assert cleaned.last_error_message is None
