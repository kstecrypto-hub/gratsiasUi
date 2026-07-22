from __future__ import annotations

import base64
import json

import pytest
from cryptography.fernet import Fernet, InvalidToken
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.hkdf import HKDF

from app.core.config import Settings
from app.core.logging import redact_text
from app.services.yeastar.configuration_store import (
    YEASTAR_CONFIGURATION_STATE_KEY,
    YeastarConfigurationStore,
    load_effective_yeastar_settings,
    overlay_yeastar_configuration,
)
from app.services.yeastar.errors import YeastarConfigurationStateError
from app.services.yeastar.schemas import YeastarConnectionConfig


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


def base_settings(**overrides: object) -> Settings:
    values: dict[str, object] = {
        "APP_ENV": "test",
        "APP_SECRET_KEY": "application-secret-key-with-32-characters",
        "YEASTAR_NAME": "Environment PBX",
        "YEASTAR_BASE_URL": "https://environment-pbx.example",
        "YEASTAR_CLIENT_ID": "environment-client-id",
        "YEASTAR_CLIENT_SECRET": "environment-client-secret",
        "YEASTAR_IGNORE_SSL_ERRORS": True,
        "YEASTAR_ALLOW_HTTP": False,
    }
    values.update(overrides)
    return Settings(_env_file=None, **values)


def saved_configuration() -> YeastarConnectionConfig:
    return YeastarConnectionConfig.model_validate(
        {
            "Name": "UI-managed PBX",
            "Settings": {
                "BaseUrl": "https://ui-pbx.example:8088",
                "ClientId": "ui-client-identifier",
                "ClientSecret": "ui-client-secret-value",
                "DateFormat": "yyyy-MM-dd HH:mm:ss",
                "PageSize": 250,
                "IgnoreSslErrors": False,
            },
        }
    )


@pytest.mark.asyncio
async def test_configuration_is_encrypted_without_plaintext_or_ttl() -> None:
    redis = MemoryRedis()
    settings = base_settings()
    store = YeastarConfigurationStore(redis, settings.APP_SECRET_KEY or "")  # type: ignore[arg-type]

    await store.write(saved_configuration())
    raw = await redis.get(YEASTAR_CONFIGURATION_STATE_KEY)

    assert isinstance(raw, bytes)
    assert b"ui-pbx.example" not in raw
    assert b"ui-client-identifier" not in raw
    assert b"ui-client-secret-value" not in raw
    assert redis.set_options == {}
    assert json.loads(store._fernet.decrypt(raw)) == {  # noqa: SLF001
        "Name": "UI-managed PBX",
        "Settings": {
            "BaseUrl": "https://ui-pbx.example:8088",
            "ClientId": "ui-client-identifier",
            "ClientSecret": "ui-client-secret-value",
            "DateFormat": "yyyy-MM-dd HH:mm:ss",
            "PageSize": 250,
            "IgnoreSslErrors": False,
        },
    }


@pytest.mark.asyncio
async def test_configuration_round_trip_preserves_exact_contract_and_registers_secrets() -> None:
    redis = MemoryRedis()
    settings = base_settings()
    store = YeastarConfigurationStore(redis, settings.APP_SECRET_KEY or "")  # type: ignore[arg-type]
    expected = saved_configuration()

    await store.write(expected)
    restored = await store.read()

    assert restored == expected
    assert restored is not None
    assert restored.model_dump(mode="json") == {
        "Name": "UI-managed PBX",
        "Settings": {
            "BaseUrl": "https://ui-pbx.example:8088",
            "ClientId": "ui-client-identifier",
            "ClientSecret": "**********",
            "DateFormat": "yyyy-MM-dd HH:mm:ss",
            "PageSize": 250,
            "IgnoreSslErrors": False,
        },
    }
    assert "ui-client-identifier" not in redact_text("ui-client-identifier")
    assert "ui-client-secret-value" not in redact_text("ui-client-secret-value")


@pytest.mark.asyncio
async def test_configuration_uses_an_independent_kdf_domain_from_token_state() -> None:
    redis = MemoryRedis()
    settings = base_settings()
    secret = settings.APP_SECRET_KEY or ""
    store = YeastarConfigurationStore(redis, secret)  # type: ignore[arg-type]
    await store.write(saved_configuration())
    raw = await redis.get(YEASTAR_CONFIGURATION_STATE_KEY)
    assert isinstance(raw, bytes)

    token_state_key = HKDF(
        algorithm=hashes.SHA256(),
        length=32,
        salt=b"yeastar-call-analyzer:token-state:v1",
        info=b"encrypted-shared-yeastar-token-state",
    ).derive(secret.encode("utf-8"))
    token_state_fernet = Fernet(base64.urlsafe_b64encode(token_state_key))

    with pytest.raises(InvalidToken):
        token_state_fernet.decrypt(raw)


def test_overlay_replaces_only_ui_managed_fields_and_applies_tls_policy() -> None:
    base = base_settings()

    effective = overlay_yeastar_configuration(base, saved_configuration())

    assert effective.YEASTAR_NAME == "UI-managed PBX"
    assert effective.YEASTAR_BASE_URL == "https://ui-pbx.example:8088"
    assert effective.YEASTAR_CLIENT_ID == "ui-client-identifier"
    assert effective.YEASTAR_CLIENT_SECRET == "ui-client-secret-value"
    assert effective.YEASTAR_DATE_FORMAT == "yyyy-MM-dd HH:mm:ss"
    assert effective.YEASTAR_PAGE_SIZE == 250
    assert effective.YEASTAR_IGNORE_SSL_ERRORS is False
    assert effective.yeastar_verify_ssl is True
    assert effective.YEASTAR_ALLOW_HTTP is base.YEASTAR_ALLOW_HTTP
    assert effective.DATABASE_URL == base.DATABASE_URL
    assert base.YEASTAR_NAME == "Environment PBX"


@pytest.mark.asyncio
async def test_corrupted_configuration_fails_closed_and_is_not_deleted() -> None:
    redis = MemoryRedis()
    redis.values[YEASTAR_CONFIGURATION_STATE_KEY] = b"corrupted-encrypted-state"
    original = redis.values[YEASTAR_CONFIGURATION_STATE_KEY]
    settings = base_settings()
    store = YeastarConfigurationStore(redis, settings.APP_SECRET_KEY or "")  # type: ignore[arg-type]

    with pytest.raises(YeastarConfigurationStateError) as raised:
        await store.read()

    assert str(raised.value) == "The saved phone-system configuration cannot be read safely."
    assert redis.values[YEASTAR_CONFIGURATION_STATE_KEY] == original


@pytest.mark.asyncio
async def test_empty_store_falls_back_to_the_unchanged_base_settings() -> None:
    redis = MemoryRedis()
    base = base_settings()

    effective = await load_effective_yeastar_settings(redis, base)  # type: ignore[arg-type]

    assert effective is base


@pytest.mark.asyncio
async def test_effective_settings_loads_saved_configuration_and_clear_restores_fallback() -> None:
    redis = MemoryRedis()
    base = base_settings()
    store = YeastarConfigurationStore(redis, base.APP_SECRET_KEY or "")  # type: ignore[arg-type]
    await store.write(saved_configuration())

    effective = await load_effective_yeastar_settings(redis, base)  # type: ignore[arg-type]
    assert effective.YEASTAR_BASE_URL == "https://ui-pbx.example:8088"
    assert effective.YEASTAR_CLIENT_SECRET == "ui-client-secret-value"

    await store.clear()
    assert await load_effective_yeastar_settings(redis, base) is base  # type: ignore[arg-type]
