from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from fastapi import HTTPException
from starlette.requests import Request

from app.api import health as health_api
from app.api import settings as settings_api
from app.core.config import Settings
from app.models import IntegrationStatus
from app.models.enums import YeastarConnectionStatus
from app.schemas.configuration import YeastarConnectionConfigurationUpdate
from app.services.yeastar.capabilities import build_capability_profile
from app.services.yeastar.circuit_breaker import YEASTAR_CIRCUIT_KEY, YeastarCircuitBreaker
from app.services.yeastar.configuration_store import (
    YEASTAR_CONFIGURATION_STATE_KEY,
    load_effective_yeastar_settings,
)
from app.services.yeastar.errors import YeastarLockTimeoutError
from app.services.yeastar.integration import runtime_cdr_api_version
from app.services.yeastar.schemas import ConnectionState, SystemInformation
from app.services.yeastar.token_store import YEASTAR_TOKEN_LOCK_KEY, YEASTAR_TOKEN_STATE_KEY


def configured_settings(tmp_path: Path) -> Settings:
    return Settings(
        APP_ENV="test",
        APP_SECRET_KEY="test-secret-key-that-is-at-least-32-characters",
        STORAGE_ROOT=tmp_path,
        YEASTAR_NAME="Main Yeastar PBX",
        YEASTAR_BASE_URL="https://pbx.example.test:8088",
        YEASTAR_CLIENT_ID="client-id",
        YEASTAR_CLIENT_SECRET="never-return-this-secret",
        YEASTAR_DATE_FORMAT="MM/dd/yyyy HH:mm:ss",
        YEASTAR_PAGE_SIZE=500,
        YEASTAR_IGNORE_SSL_ERRORS=True,
    )


class _Rows:
    def all(self) -> list[object]:
        return []


class _Session:
    def __init__(self) -> None:
        self.commits = 0

    async def scalars(self, _statement: object) -> _Rows:
        return _Rows()

    async def commit(self) -> None:
        self.commits += 1


class _Redis:
    def __init__(self) -> None:
        self.values: dict[str, object] = {}

    async def get(self, key: str) -> object | None:
        return self.values.get(key)

    async def set(self, key: str, value: object, **kwargs: object) -> bool:
        if kwargs.get("nx") and key in self.values:
            return False
        self.values[key] = value
        return True

    async def delete(self, *keys: str) -> int:
        removed = 0
        for key in keys:
            removed += int(key in self.values)
            self.values.pop(key, None)
        return removed

    def pipeline(self, **_kwargs: object) -> "_Pipeline":
        return _Pipeline(self)

    async def eval(self, _script: str, key_count: int, *args: object) -> int:
        if key_count == 1:
            key, owner = str(args[0]), args[1]
            if self.values.get(key) != owner:
                return 0
            self.values.pop(key, None)
            return 1
        if key_count == 3:
            configuration_key, circuit_key, token_key = map(str, args[:3])
            encrypted, circuit = args[3:]
            self.values[configuration_key] = encrypted
            self.values[circuit_key] = circuit
            self.values.pop(token_key, None)
            return 1
        raise AssertionError("unexpected Redis script")


class _Pipeline:
    def __init__(self, redis: _Redis) -> None:
        self.redis = redis
        self.operations: list[tuple[str, tuple[object, ...]]] = []

    async def __aenter__(self) -> "_Pipeline":
        return self

    async def __aexit__(self, *_args: object) -> None:
        return None

    def set(self, key: str, value: object) -> "_Pipeline":
        self.operations.append(("set", (key, value)))
        return self

    def mset(self, values: dict[str, object]) -> "_Pipeline":
        self.operations.append(("mset", (values,)))
        return self

    def delete(self, *keys: str) -> "_Pipeline":
        self.operations.append(("delete", tuple(keys)))
        return self

    async def execute(self) -> list[object]:
        results: list[object] = []
        for operation, values in self.operations:
            if operation == "set":
                results.append(await self.redis.set(str(values[0]), values[1]))
            elif operation == "mset":
                for key, value in dict(values[0]).items():
                    await self.redis.set(str(key), value)
                results.append(True)
            else:
                results.append(await self.redis.delete(*(str(value) for value in values)))
        return results


def _request(path: str) -> Request:
    return Request(
        {
            "type": "http",
            "method": "POST",
            "path": path,
            "headers": [],
            "client": ("127.0.0.1", 1234),
        }
    )


@pytest.mark.asyncio
async def test_configuration_and_validation_are_exact_and_construct_no_client(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    settings = configured_settings(tmp_path)

    class ForbiddenClient:
        def __init__(self, *_args: object, **_kwargs: object) -> None:
            raise AssertionError("local endpoint constructed a provider client")

    monkeypatch.setattr(settings_api, "YeastarClient", ForbiddenClient)
    configuration = await settings_api.get_yeastar_configuration(object(), settings)
    validation = await settings_api.validate_yeastar_configuration(object(), settings)

    expected = {
        "Name": "Main Yeastar PBX",
        "Settings": {
            "BaseUrl": "https://pbx.example.test:8088",
            "ClientId": "[CONFIGURED]",
            "ClientSecret": "[REDACTED]",
            "DateFormat": "MM/dd/yyyy HH:mm:ss",
            "PageSize": 500,
            "IgnoreSslErrors": True,
        },
    }
    assert configuration.model_dump() == expected
    assert validation.model_dump() == {
        "valid": True,
        "errors": [],
        "configuration": expected,
    }
    assert "never-return-this-secret" not in json.dumps(validation.model_dump())


@pytest.mark.asyncio
async def test_ui_configuration_save_is_encrypted_local_and_requires_manual_test(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    settings = configured_settings(tmp_path).model_copy(
        update={
            "YEASTAR_NAME": "",
            "YEASTAR_BASE_URL": None,
            "YEASTAR_CLIENT_ID": None,
            "YEASTAR_CLIENT_SECRET": None,
        }
    )
    payload = YeastarConnectionConfigurationUpdate.model_validate(
        {
            "Name": "Office PBX",
            "Settings": {
                "BaseUrl": "https://pbx.ui.example:8088/",
                "ClientId": "ui-client-id",
                "ClientSecret": "ui-client-secret",
                "DateFormat": "yyyy-MM-dd HH:mm:ss",
                "PageSize": 250,
                "IgnoreSslErrors": False,
            },
        }
    )
    redis = _Redis()
    redis.values[YEASTAR_TOKEN_STATE_KEY] = "old-encrypted-token"
    row = IntegrationStatus(
        provider="yeastar",
        status=YeastarConnectionStatus.CONNECTED,
        capabilities_json={"extensions": True},
    )

    class ForbiddenClient:
        def __init__(self, *_args: object, **_kwargs: object) -> None:
            raise AssertionError("saving configuration contacted the provider")

    async def get_row(*_args: object, **_kwargs: object):
        return row

    async def no_audit(*_args: object, **_kwargs: object) -> None:
        return None

    monkeypatch.setattr(settings_api, "get_redis", lambda: redis)
    monkeypatch.setattr(settings_api, "get_integration_status", get_row)
    monkeypatch.setattr(settings_api, "audit", no_audit)
    monkeypatch.setattr(settings_api, "YeastarClient", ForbiddenClient)
    session = _Session()

    response = await settings_api.update_yeastar_configuration(
        payload,
        _request("/api/settings/yeastar/configuration"),
        object(),
        session,
        settings,
    )

    assert response.model_dump()["valid"] is True
    encrypted = redis.values[YEASTAR_CONFIGURATION_STATE_KEY]
    assert isinstance(encrypted, bytes)
    assert b"ui-client-id" not in encrypted
    assert b"ui-client-secret" not in encrypted
    assert YEASTAR_TOKEN_STATE_KEY not in redis.values
    assert '"reason":"not_tested"' in str(redis.values[YEASTAR_CIRCUIT_KEY])
    assert row.status == YeastarConnectionStatus.NOT_TESTED
    assert row.last_tested_at is None
    assert row.configuration_fingerprint
    effective = await load_effective_yeastar_settings(redis, settings)  # type: ignore[arg-type]
    assert effective.YEASTAR_BASE_URL == "https://pbx.ui.example:8088"
    assert effective.YEASTAR_CLIENT_ID == "ui-client-id"
    assert effective.YEASTAR_CLIENT_SECRET == "ui-client-secret"


@pytest.mark.asyncio
async def test_idempotent_ui_save_preserves_connected_state_and_token(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    settings = configured_settings(tmp_path)
    payload = YeastarConnectionConfigurationUpdate.model_validate(
        {
            "Name": settings.YEASTAR_NAME,
            "Settings": {
                "BaseUrl": settings.YEASTAR_BASE_URL,
                "ClientId": "",
                "ClientSecret": "",
                "DateFormat": settings.YEASTAR_DATE_FORMAT,
                "PageSize": 500,
                "IgnoreSslErrors": True,
            },
        }
    )
    redis = _Redis()
    redis.values[YEASTAR_TOKEN_STATE_KEY] = "still-valid-encrypted-token"
    row = IntegrationStatus(
        provider="yeastar",
        status=YeastarConnectionStatus.CONNECTED,
        configuration_fingerprint=settings.yeastar_configuration_fingerprint,
        model_name="P-Series",
        capabilities_json={"extensions": True},
    )

    async def get_row(*_args: object, **_kwargs: object):
        return row

    async def no_audit(*_args: object, **_kwargs: object) -> None:
        return None

    monkeypatch.setattr(settings_api, "get_redis", lambda: redis)
    monkeypatch.setattr(settings_api, "get_integration_status", get_row)
    monkeypatch.setattr(settings_api, "audit", no_audit)

    response = await settings_api.update_yeastar_configuration(
        payload,
        _request("/api/settings/yeastar/configuration"),
        object(),
        _Session(),
        settings,
    )

    assert response.model_dump()["valid"] is True
    assert redis.values[YEASTAR_TOKEN_STATE_KEY] == "still-valid-encrypted-token"
    assert YEASTAR_CIRCUIT_KEY not in redis.values
    assert row.status == YeastarConnectionStatus.CONNECTED
    assert row.model_name == "P-Series"
    assert row.capabilities_json == {"extensions": True}
    assert YEASTAR_CONFIGURATION_STATE_KEY in redis.values


@pytest.mark.asyncio
async def test_busy_auth_lock_cannot_leave_a_partial_configuration_transition(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    settings = configured_settings(tmp_path)
    payload = YeastarConnectionConfigurationUpdate.model_validate(
        {
            "Name": "Changed PBX",
            "Settings": {
                "BaseUrl": "https://changed.example.test",
                "ClientId": "changed-client",
                "ClientSecret": "changed-secret",
                "DateFormat": "MM/dd/yyyy HH:mm:ss",
                "PageSize": 500,
                "IgnoreSslErrors": True,
            },
        }
    )
    redis = _Redis()
    redis.values[YEASTAR_TOKEN_STATE_KEY] = "existing-token"
    row = IntegrationStatus(
        provider="yeastar",
        status=YeastarConnectionStatus.CONNECTED,
        configuration_fingerprint=settings.yeastar_configuration_fingerprint,
        model_name="Existing PBX",
        capabilities_json={"extensions": True},
    )

    class BusyTokenLock:
        def __init__(self, _redis: object, key: str = YEASTAR_TOKEN_LOCK_KEY, **_kwargs: object):
            self.key = key

        async def acquire(self) -> bool:
            if self.key == settings_api.YEASTAR_MANUAL_TEST_LOCK:
                return True
            raise YeastarLockTimeoutError("busy")

        async def release(self) -> bool:
            return True

        async def __aenter__(self) -> "BusyTokenLock":
            await self.acquire()
            return self

        async def __aexit__(self, *_args: object) -> None:
            await self.release()

    async def get_row(*_args: object, **_kwargs: object):
        return row

    monkeypatch.setattr(settings_api, "get_redis", lambda: redis)
    monkeypatch.setattr(settings_api, "get_integration_status", get_row)
    monkeypatch.setattr(settings_api, "RedisOwnerLock", BusyTokenLock)

    with pytest.raises(HTTPException) as raised:
        await settings_api.update_yeastar_configuration(
            payload,
            _request("/api/settings/yeastar/configuration"),
            object(),
            _Session(),
            settings,
        )

    assert raised.value.status_code == 409
    assert redis.values[YEASTAR_TOKEN_STATE_KEY] == "existing-token"
    assert YEASTAR_CONFIGURATION_STATE_KEY not in redis.values
    assert YEASTAR_CIRCUIT_KEY not in redis.values
    assert row.status == YeastarConnectionStatus.CONNECTED
    assert row.model_name == "Existing PBX"


def test_blank_ui_credentials_preserve_existing_values(tmp_path: Path) -> None:
    existing = configured_settings(tmp_path)
    payload = YeastarConnectionConfigurationUpdate.model_validate(
        {
            "Name": "Renamed PBX",
            "Settings": {
                "BaseUrl": "https://new-pbx.example.test",
                "ClientId": "",
                "ClientSecret": "",
                "DateFormat": "MM/dd/yyyy HH:mm:ss",
                "PageSize": 500,
                "IgnoreSslErrors": True,
            },
        }
    )

    candidate = settings_api._configuration_update_settings(existing, existing, payload)

    assert candidate.YEASTAR_CLIENT_ID == "client-id"
    assert candidate.YEASTAR_CLIENT_SECRET == "never-return-this-secret"

    exact_secret_payload = payload.model_copy(deep=True)
    exact_secret_payload.Settings.ClientSecret = type(
        exact_secret_payload.Settings.ClientSecret
    )("  significant whitespace  ")
    exact_secret = settings_api._configuration_update_settings(
        existing,
        existing,
        exact_secret_payload,
    )
    assert exact_secret.YEASTAR_CLIENT_SECRET == "  significant whitespace  "


@pytest.mark.asyncio
async def test_missing_configuration_test_returns_safe_validation_with_zero_network(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    settings = configured_settings(tmp_path).model_copy(
        update={
            "YEASTAR_BASE_URL": None,
            "YEASTAR_CLIENT_ID": None,
            "YEASTAR_CLIENT_SECRET": None,
        }
    )

    class ForbiddenClient:
        def __init__(self, *_args: object, **_kwargs: object) -> None:
            raise AssertionError("invalid configuration constructed a provider client")

    monkeypatch.setattr(settings_api, "YeastarClient", ForbiddenClient)
    monkeypatch.setattr(settings_api, "get_redis", lambda: _Redis())
    response = await settings_api.test_yeastar_connection(
        _request("/api/settings/yeastar/test"),
        object(),
        object(),
        settings,
    )

    assert response.status_code == 422
    payload = json.loads(response.body)
    assert payload["valid"] is False
    assert payload["configuration"]["Settings"]["ClientId"] == "[NOT CONFIGURED]"
    assert payload["configuration"]["Settings"]["ClientSecret"] == "[NOT CONFIGURED]"


@pytest.mark.asyncio
async def test_status_and_health_are_local_only(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    settings = configured_settings(tmp_path)
    row = IntegrationStatus(
        provider="yeastar",
        status=YeastarConnectionStatus.CONNECTED,
        configuration_fingerprint=settings.yeastar_configuration_fingerprint,
        model_name="P-Series Cloud Edition",
        firmware_version="84.23.0.123",
        capabilities_json={"extensions": True, "cdr_v2": True, "recordings": True},
    )

    class ForbiddenClient:
        def __init__(self, *_args: object, **_kwargs: object) -> None:
            raise AssertionError("local status constructed a provider client")

    async def reconcile(*_args: object, **_kwargs: object):
        return row, True, False

    async def get_row(*_args: object, **_kwargs: object):
        return row

    monkeypatch.setattr(settings_api, "YeastarClient", ForbiddenClient)
    monkeypatch.setattr(settings_api, "reconcile_configuration_fingerprint", reconcile)
    monkeypatch.setattr(health_api, "get_integration_status", get_row)
    redis = _Redis()
    monkeypatch.setattr(settings_api, "get_redis", lambda: redis)
    monkeypatch.setattr(health_api, "get_redis", lambda: redis)
    session = _Session()

    status = await settings_api.get_yeastar_status(object(), session, settings)
    health = await health_api.health_yeastar(session, settings)

    assert status.status == YeastarConnectionStatus.CONNECTED
    assert status.capabilities.model_dump() == {
        "extensions": True,
        "cdr_v2": True,
        "cdr_api_version": None,
        "recordings": True,
    }
    assert health.status == YeastarConnectionStatus.CONNECTED


def test_runtime_cdr_api_version_uses_legacy_only_when_it_was_persisted() -> None:
    assert runtime_cdr_api_version(None) == "v2"
    assert runtime_cdr_api_version({}) == "v2"
    assert runtime_cdr_api_version({"cdr_api_version": "v2"}) == "v2"
    assert runtime_cdr_api_version({"cdr_api_version": "unexpected"}) == "v2"
    assert runtime_cdr_api_version({"cdr_api_version": "v1"}) == "v1"


@pytest.mark.asyncio
async def test_runtime_circuit_overrides_stale_connected_status(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    settings = configured_settings(tmp_path)
    row = IntegrationStatus(
        provider="yeastar",
        status=YeastarConnectionStatus.CONNECTED,
        configuration_fingerprint=settings.yeastar_configuration_fingerprint,
        capabilities_json={"extensions": True, "cdr_v2": True, "recordings": True},
    )
    redis = _Redis()
    await YeastarCircuitBreaker(redis).open(
        ConnectionState.AUTH_REJECTED,
        last_errcode=10005,
    )

    async def reconcile(*_args: object, **_kwargs: object):
        return row, True, False

    async def get_row(*_args: object, **_kwargs: object):
        return row

    monkeypatch.setattr(settings_api, "reconcile_configuration_fingerprint", reconcile)
    monkeypatch.setattr(settings_api, "get_redis", lambda: redis)
    monkeypatch.setattr(health_api, "get_integration_status", get_row)
    monkeypatch.setattr(health_api, "get_redis", lambda: redis)
    session = _Session()

    local_status = await settings_api.get_yeastar_status(object(), session, settings)
    health = await health_api.health_yeastar(session, settings)

    assert local_status.status == YeastarConnectionStatus.AUTH_REJECTED
    assert local_status.last_error_reference == "YS-10005"
    assert health.status == YeastarConnectionStatus.AUTH_REJECTED


@pytest.mark.asyncio
async def test_manual_test_response_is_sanitized_with_exact_casing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    settings = configured_settings(tmp_path)
    row = IntegrationStatus(
        provider="yeastar",
        status=YeastarConnectionStatus.NOT_TESTED,
        configuration_fingerprint=settings.yeastar_configuration_fingerprint,
        capabilities_json={},
    )
    information = SystemInformation(
        device_name="Office PBX",
        model_name="P-Series Cloud Edition",
        firmware_version="84.23.0.123",
        system_date_format="MM/DD/YYYY",
        system_time_format="HH:mm:ss",
        timestamp=1_783_500_000,
    )
    capabilities = build_capability_profile(
        information.model_name or "", information.firmware_version or ""
    )
    manager = SimpleNamespace(
        mark_connection_successful=AsyncMock(),
        open_circuit=AsyncMock(),
    )

    class FakeClient:
        def __init__(self, *_args: object, **_kwargs: object) -> None:
            self.token_manager = manager

        async def inspect_connection(self):
            return information, capabilities

        async def aclose(self) -> None:
            return None

    async def reconcile(*_args: object, **_kwargs: object):
        return row, True, False

    async def no_audit(*_args: object, **_kwargs: object) -> None:
        return None

    monkeypatch.setattr(settings_api, "YeastarClient", FakeClient)
    monkeypatch.setattr(settings_api, "reconcile_configuration_fingerprint", reconcile)
    monkeypatch.setattr(settings_api, "get_redis", lambda: _Redis())
    monkeypatch.setattr(settings_api, "audit", no_audit)
    session = _Session()

    response = await settings_api.test_yeastar_connection(
        _request("/api/settings/yeastar/test"),
        object(),
        session,
        settings,
    )

    payload = response.model_dump(mode="json")
    assert payload["configurationAccepted"] is True
    assert set(payload["configuration"]) == {"Name", "Settings"}
    assert set(payload["configuration"]["Settings"]) == {
        "BaseUrl",
        "ClientId",
        "ClientSecret",
        "DateFormat",
        "PageSize",
        "IgnoreSslErrors",
    }
    assert payload["configuration"]["Settings"]["ClientSecret"] == "[REDACTED]"
    assert payload["connection"] == {
        "status": "connected",
        "model": "P-Series Cloud Edition",
        "firmwareVersion": "84.23.0.123",
    }
    assert "never-return-this-secret" not in json.dumps(payload)


@pytest.mark.asyncio
async def test_manual_test_reloads_configuration_after_action_lock(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    old_settings = configured_settings(tmp_path)
    new_settings = old_settings.model_copy(
        update={
            "YEASTAR_BASE_URL": "https://newly-saved.example.test",
            "YEASTAR_CLIENT_ID": "new-client",
            "YEASTAR_CLIENT_SECRET": "new-secret",
        }
    )
    loads = iter((old_settings, new_settings))
    captured_urls: list[str | None] = []
    row = IntegrationStatus(
        provider="yeastar",
        status=YeastarConnectionStatus.NOT_TESTED,
        configuration_fingerprint=new_settings.yeastar_configuration_fingerprint,
        capabilities_json={},
    )
    information = SystemInformation(
        model_name="P-Series Cloud Edition",
        firmware_version="84.23.0.123",
    )
    capabilities = build_capability_profile(
        information.model_name or "",
        information.firmware_version or "",
    )
    manager = SimpleNamespace(
        mark_connection_successful=AsyncMock(),
        open_circuit=AsyncMock(),
    )

    async def load_after_race(*_args: object, **_kwargs: object) -> Settings:
        return next(loads)

    async def reconcile(*_args: object, **_kwargs: object):
        return row, True, False

    async def no_audit(*_args: object, **_kwargs: object) -> None:
        return None

    class CapturingClient:
        def __init__(self, *, settings: Settings, **_kwargs: object) -> None:
            captured_urls.append(settings.YEASTAR_BASE_URL)
            self.token_manager = manager

        async def inspect_connection(self):
            return information, capabilities

        async def aclose(self) -> None:
            return None

    monkeypatch.setattr(settings_api, "get_redis", lambda: _Redis())
    monkeypatch.setattr(settings_api, "_load_effective_configuration", load_after_race)
    monkeypatch.setattr(settings_api, "reconcile_configuration_fingerprint", reconcile)
    monkeypatch.setattr(settings_api, "YeastarClient", CapturingClient)
    monkeypatch.setattr(settings_api, "audit", no_audit)

    response = await settings_api.test_yeastar_connection(
        _request("/api/settings/yeastar/test"),
        object(),
        _Session(),
        old_settings,
    )

    assert response.connection.status == YeastarConnectionStatus.CONNECTED
    assert captured_urls == ["https://newly-saved.example.test"]


@pytest.mark.asyncio
async def test_reset_leaves_not_tested_breaker_open(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    settings = configured_settings(tmp_path)
    row = IntegrationStatus(
        provider="yeastar",
        status=YeastarConnectionStatus.CONNECTED,
        configuration_fingerprint=settings.yeastar_configuration_fingerprint,
        capabilities_json={"extensions": True},
    )
    reasons: list[str] = []
    revoke = AsyncMock()

    class FakeBreaker:
        def __init__(self, _redis: object) -> None:
            pass

        async def open(self, reason) -> None:
            reasons.append(reason.value)

    class FakeManager:
        def __init__(self, *_args: object, **_kwargs: object) -> None:
            pass

        async def revoke_current_token(self, **_kwargs: object) -> None:
            await revoke()

        async def aclose(self) -> None:
            return None

    async def get_row(*_args: object, **_kwargs: object):
        return row

    async def no_audit(*_args: object, **_kwargs: object) -> None:
        return None

    monkeypatch.setattr(settings_api, "get_redis", lambda: _Redis())
    monkeypatch.setattr(settings_api, "YeastarCircuitBreaker", FakeBreaker)
    monkeypatch.setattr(settings_api, "YeastarTokenManager", FakeManager)
    monkeypatch.setattr(settings_api, "get_integration_status", get_row)
    monkeypatch.setattr(settings_api, "audit", no_audit)
    session = _Session()

    response = await settings_api.reset_yeastar_connection(
        _request("/api/settings/yeastar/reset"), object(), session, settings
    )

    assert reasons == ["not_tested"]
    revoke.assert_awaited_once_with()
    assert response.status == YeastarConnectionStatus.NOT_TESTED
    assert row.capabilities_json == {}


@pytest.mark.asyncio
async def test_reset_busy_auth_lock_does_not_mutate_connection_state(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    settings = configured_settings(tmp_path)
    redis = _Redis()
    redis.values[YEASTAR_TOKEN_STATE_KEY] = "existing-token"
    row = IntegrationStatus(
        provider="yeastar",
        status=YeastarConnectionStatus.CONNECTED,
        configuration_fingerprint=settings.yeastar_configuration_fingerprint,
        model_name="Existing PBX",
        capabilities_json={"extensions": True},
    )
    breaker_open = AsyncMock()

    class BusyTokenLock:
        def __init__(
            self,
            _redis: object,
            key: str = YEASTAR_TOKEN_LOCK_KEY,
            **_kwargs: object,
        ) -> None:
            self.key = key

        async def acquire(self) -> bool:
            if self.key == settings_api.YEASTAR_MANUAL_TEST_LOCK:
                return True
            raise YeastarLockTimeoutError("busy")

        async def release(self) -> bool:
            return True

    class ForbiddenManager:
        def __init__(self, *_args: object, **_kwargs: object) -> None:
            raise AssertionError("Reset constructed a manager without the auth lock")

    class FakeBreaker:
        def __init__(self, _redis: object) -> None:
            pass

        async def open(self, reason: object) -> None:
            await breaker_open(reason)

    async def get_row(*_args: object, **_kwargs: object):
        return row

    monkeypatch.setattr(settings_api, "get_redis", lambda: redis)
    monkeypatch.setattr(settings_api, "RedisOwnerLock", BusyTokenLock)
    monkeypatch.setattr(settings_api, "YeastarCircuitBreaker", FakeBreaker)
    monkeypatch.setattr(settings_api, "YeastarTokenManager", ForbiddenManager)
    monkeypatch.setattr(settings_api, "get_integration_status", get_row)
    session = _Session()

    with pytest.raises(HTTPException) as raised:
        await settings_api.reset_yeastar_connection(
            _request("/api/settings/yeastar/reset"), object(), session, settings
        )

    assert raised.value.status_code == 409
    assert raised.value.detail == (
        "Phone-system authentication is busy. Try resetting again."
    )
    breaker_open.assert_not_awaited()
    assert redis.values[YEASTAR_TOKEN_STATE_KEY] == "existing-token"
    assert YEASTAR_CIRCUIT_KEY not in redis.values
    assert session.commits == 0
    assert row.status == YeastarConnectionStatus.CONNECTED
    assert row.model_name == "Existing PBX"
    assert row.capabilities_json == {"extensions": True}
