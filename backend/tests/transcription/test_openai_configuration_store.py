from __future__ import annotations

import json

import pytest
from starlette.requests import Request

from app.api import settings as settings_api
from app.core.config import Settings
from app.core.logging import redact_text
from app.schemas.configuration import OpenAIConfigurationUpdate
from app.services.transcription.configuration_store import (
    OPENAI_CONFIGURATION_STATE_KEY,
    OpenAIConfiguration,
    OpenAIConfigurationStateError,
    OpenAIConfigurationStore,
    load_effective_openai_settings,
    overlay_openai_configuration,
)


class MemoryRedis:
    def __init__(self) -> None:
        self.values: dict[str, object] = {}
        self.set_options: dict[str, object] = {}

    async def get(self, key: str) -> object | None:
        return self.values.get(key)

    async def set(self, key: str, value: object, **options: object) -> bool:
        self.values[key] = value
        self.set_options = options
        return True

    async def delete(self, *keys: str) -> int:
        removed = 0
        for key in keys:
            removed += int(key in self.values)
            self.values.pop(key, None)
        return removed


class Session:
    def __init__(self) -> None:
        self.commits = 0

    async def commit(self) -> None:
        self.commits += 1


class OwnerLock:
    def __init__(self, *_args: object, **_kwargs: object) -> None:
        self.acquired = False

    async def acquire(self) -> bool:
        self.acquired = True
        return True

    async def release(self) -> bool:
        self.acquired = False
        return True


def base_settings(**overrides: object) -> Settings:
    values: dict[str, object] = {
        "APP_ENV": "test",
        "APP_SECRET_KEY": "application-secret-key-with-32-characters",
        "OPENAI_API_KEY": "environment-openai-key",
    }
    values.update(overrides)
    return Settings(_env_file=None, **values)


def request(path: str) -> Request:
    return Request(
        {
            "type": "http",
            "method": "PUT",
            "path": path,
            "headers": [],
            "client": ("127.0.0.1", 1234),
        }
    )


@pytest.mark.asyncio
async def test_openai_configuration_is_encrypted_and_never_serialized_plaintext() -> None:
    redis = MemoryRedis()
    settings = base_settings()
    api_key = "ui-openai-key-never-returned"
    store = OpenAIConfigurationStore(redis, settings.APP_SECRET_KEY or "")  # type: ignore[arg-type]

    await store.write(OpenAIConfiguration(api_key))
    raw = await redis.get(OPENAI_CONFIGURATION_STATE_KEY)

    assert isinstance(raw, bytes)
    assert api_key.encode() not in raw
    assert redis.set_options == {}
    assert json.loads(store._fernet.decrypt(raw)) == {"api_key": api_key}  # noqa: SLF001

    restored = await store.read()
    assert restored is not None
    assert restored.api_key_value == api_key
    assert api_key not in redact_text(api_key)


def test_openai_overlay_replaces_only_the_key() -> None:
    base = base_settings()

    effective = overlay_openai_configuration(base, OpenAIConfiguration("ui-openai-key"))

    assert effective.OPENAI_API_KEY == "ui-openai-key"
    assert effective.OPENAI_TRANSCRIPTION_MODEL == base.OPENAI_TRANSCRIPTION_MODEL
    assert effective.DATABASE_URL == base.DATABASE_URL
    assert base.OPENAI_API_KEY == "environment-openai-key"


@pytest.mark.asyncio
async def test_openai_configuration_fails_closed_when_ciphertext_is_unreadable() -> None:
    redis = MemoryRedis()
    redis.values[OPENAI_CONFIGURATION_STATE_KEY] = b"corrupted-encrypted-state"
    original = redis.values[OPENAI_CONFIGURATION_STATE_KEY]
    settings = base_settings()
    store = OpenAIConfigurationStore(redis, settings.APP_SECRET_KEY or "")  # type: ignore[arg-type]

    with pytest.raises(OpenAIConfigurationStateError) as raised:
        await store.read()

    assert str(raised.value) == "The saved OpenAI configuration cannot be read safely."
    assert redis.values[OPENAI_CONFIGURATION_STATE_KEY] == original


@pytest.mark.asyncio
async def test_effective_openai_settings_uses_ui_override_and_keeps_environment_fallback() -> None:
    redis = MemoryRedis()
    base = base_settings()
    store = OpenAIConfigurationStore(redis, base.APP_SECRET_KEY or "")  # type: ignore[arg-type]

    assert await load_effective_openai_settings(redis, base) is base  # type: ignore[arg-type]
    await store.write(OpenAIConfiguration("ui-openai-key"))
    effective = await load_effective_openai_settings(redis, base)  # type: ignore[arg-type]
    assert effective.OPENAI_API_KEY == "ui-openai-key"

    await store.clear()
    assert await load_effective_openai_settings(redis, base) is base  # type: ignore[arg-type]


@pytest.mark.asyncio
async def test_openai_settings_routes_are_write_only_and_blank_preserves_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    redis = MemoryRedis()
    settings = base_settings(OPENAI_API_KEY=None)
    session = Session()
    recorded_audits: list[dict[str, object]] = []
    api_key = "ui-openai-key-never-returned"

    async def record_audit(*_args: object, **kwargs: object) -> None:
        details = kwargs.get("details", {})
        assert isinstance(details, dict)
        recorded_audits.append(details)

    monkeypatch.setattr(settings_api, "get_redis", lambda: redis)
    monkeypatch.setattr(settings_api, "RedisOwnerLock", OwnerLock)
    monkeypatch.setattr(settings_api, "audit", record_audit)

    saved = await settings_api.update_openai_configuration(
        OpenAIConfigurationUpdate.model_validate({"api_key": api_key}),
        request("/api/settings/openai/configuration"),
        object(),
        session,  # type: ignore[arg-type]
        settings,
    )
    assert saved.model_dump() == {
        "valid": True,
        "errors": [],
        "configuration": {"api_key": "[CONFIGURED]"},
    }
    encrypted = await redis.get(OPENAI_CONFIGURATION_STATE_KEY)
    assert encrypted is not None
    assert api_key not in str(encrypted)
    assert api_key not in json.dumps(recorded_audits)
    assert session.commits == 1

    preserved = await settings_api.update_openai_configuration(
        OpenAIConfigurationUpdate.model_validate({"api_key": ""}),
        request("/api/settings/openai/configuration"),
        object(),
        session,  # type: ignore[arg-type]
        settings,
    )
    assert preserved.model_dump()["configuration"] == {"api_key": "[CONFIGURED]"}
    assert await redis.get(OPENAI_CONFIGURATION_STATE_KEY) == encrypted
    assert len(recorded_audits) == 1

    safe = await settings_api.get_openai_configuration(object(), settings)
    assert safe.model_dump() == {"api_key": "[CONFIGURED]"}
    assert api_key not in json.dumps(safe.model_dump())


@pytest.mark.asyncio
async def test_openai_connection_test_uses_effective_ui_key_without_reflecting_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    redis = MemoryRedis()
    settings = base_settings(OPENAI_API_KEY=None)
    session = Session()
    api_key = "ui-openai-key-never-returned"
    store = OpenAIConfigurationStore(redis, settings.APP_SECRET_KEY or "")  # type: ignore[arg-type]
    await store.write(OpenAIConfiguration(api_key))
    captured: list[str | None] = []

    class Client:
        def __init__(self, *, settings: Settings) -> None:
            captured.append(settings.OPENAI_API_KEY)

        async def __aenter__(self) -> "Client":
            return self

        async def __aexit__(self, *_args: object) -> None:
            return None

        async def test_connection(self) -> bool:
            return True

    async def no_audit(*_args: object, **_kwargs: object) -> None:
        return None

    monkeypatch.setattr(settings_api, "get_redis", lambda: redis)
    monkeypatch.setattr(settings_api, "RedisOwnerLock", OwnerLock)
    monkeypatch.setattr(settings_api, "OpenAITranscriptionClient", Client)
    monkeypatch.setattr(settings_api, "audit", no_audit)

    response = await settings_api.test_openai_configuration(
        request("/api/settings/openai/test"),
        object(),
        session,  # type: ignore[arg-type]
        settings,
    )
    assert response.model_dump() == {
        "configurationAccepted": True,
        "configuration": {"api_key": "[CONFIGURED]"},
        "connection": {"status": "connected"},
    }
    assert captured == [api_key]
    assert api_key not in json.dumps(response.model_dump())
