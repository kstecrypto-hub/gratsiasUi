from __future__ import annotations

import base64
import json

from cryptography.fernet import Fernet, InvalidToken
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.hkdf import HKDF
from redis.asyncio import Redis

from app.core.config import Settings
from app.core.logging import register_secret
from app.services.yeastar.errors import YeastarConfigurationStateError
from app.services.yeastar.schemas import YeastarConnectionConfig


YEASTAR_CONFIGURATION_STATE_KEY = "yeastar:configuration:state:v1"

_CONFIGURATION_KDF_SALT = b"yeastar-call-analyzer:configuration-state:v1"
_CONFIGURATION_KDF_INFO = b"encrypted-ui-managed-yeastar-configuration"
_UNREADABLE_CONFIGURATION_MESSAGE = (
    "The saved phone-system configuration cannot be read safely."
)


def _register_credentials(configuration: YeastarConnectionConfig) -> None:
    register_secret(configuration.Settings.ClientId)
    register_secret(configuration.Settings.ClientSecret.get_secret_value())


class YeastarConfigurationStore:
    """Persistent encrypted storage for UI-managed Yeastar configuration."""

    def __init__(
        self,
        redis: Redis,
        app_secret_key: str,
        *,
        state_key: str = YEASTAR_CONFIGURATION_STATE_KEY,
    ) -> None:
        if not app_secret_key:
            raise YeastarConfigurationStateError(
                "Application security is required to save phone-system configuration."
            )
        self.redis = redis
        self.state_key = state_key
        derived = HKDF(
            algorithm=hashes.SHA256(),
            length=32,
            salt=_CONFIGURATION_KDF_SALT,
            info=_CONFIGURATION_KDF_INFO,
        ).derive(app_secret_key.encode("utf-8"))
        self._fernet = Fernet(base64.urlsafe_b64encode(derived))

    @staticmethod
    def _serialized(configuration: YeastarConnectionConfig) -> bytes:
        payload = {
            "Name": configuration.Name,
            "Settings": {
                "BaseUrl": configuration.Settings.BaseUrl,
                "ClientId": configuration.Settings.ClientId,
                "ClientSecret": configuration.Settings.ClientSecret.get_secret_value(),
                "DateFormat": configuration.Settings.DateFormat,
                "PageSize": configuration.Settings.PageSize,
                "IgnoreSslErrors": configuration.Settings.IgnoreSslErrors,
            },
        }
        return json.dumps(
            payload,
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode("utf-8")

    async def read(self) -> YeastarConnectionConfig | None:
        encrypted = await self.redis.get(self.state_key)
        if encrypted is None:
            return None
        try:
            ciphertext = encrypted if isinstance(encrypted, bytes) else encrypted.encode("utf-8")
            plaintext = self._fernet.decrypt(ciphertext)
            configuration = YeastarConnectionConfig.model_validate_json(plaintext)
        except (AttributeError, InvalidToken, TypeError, ValueError):
            # Do not delete unreadable state: silently falling back to another
            # credential source could connect to the wrong phone system.
            raise YeastarConfigurationStateError(_UNREADABLE_CONFIGURATION_MESSAGE) from None
        _register_credentials(configuration)
        return configuration

    async def write(self, configuration: YeastarConnectionConfig) -> None:
        encrypted = self.encrypt(configuration)
        # This is durable application configuration, not an expiring token.
        await self.redis.set(self.state_key, encrypted)

    def encrypt(self, configuration: YeastarConnectionConfig) -> bytes:
        """Return an authenticated ciphertext suitable for an atomic Redis write."""
        _register_credentials(configuration)
        return self._fernet.encrypt(self._serialized(configuration))

    async def clear(self) -> None:
        await self.redis.delete(self.state_key)


def overlay_yeastar_configuration(
    base_settings: Settings,
    configuration: YeastarConnectionConfig,
) -> Settings:
    """Return Settings with only the UI-managed Yeastar fields replaced."""

    _register_credentials(configuration)
    return base_settings.model_copy(
        update={
            "YEASTAR_NAME": configuration.Name,
            "YEASTAR_BASE_URL": configuration.Settings.BaseUrl,
            "YEASTAR_CLIENT_ID": configuration.Settings.ClientId,
            "YEASTAR_CLIENT_SECRET": configuration.Settings.ClientSecret.get_secret_value(),
            "YEASTAR_DATE_FORMAT": configuration.Settings.DateFormat,
            "YEASTAR_PAGE_SIZE": configuration.Settings.PageSize,
            "YEASTAR_IGNORE_SSL_ERRORS": configuration.Settings.IgnoreSslErrors,
        }
    )


async def load_effective_yeastar_settings(
    redis: Redis,
    base_settings: Settings,
) -> Settings:
    """Overlay saved UI configuration, or retain environment defaults when absent."""

    if not base_settings.APP_SECRET_KEY:
        if await redis.get(YEASTAR_CONFIGURATION_STATE_KEY) is not None:
            raise YeastarConfigurationStateError(_UNREADABLE_CONFIGURATION_MESSAGE)
        return base_settings
    store = YeastarConfigurationStore(redis, base_settings.APP_SECRET_KEY)
    configuration = await store.read()
    if configuration is None:
        return base_settings
    return overlay_yeastar_configuration(base_settings, configuration)


__all__ = [
    "YEASTAR_CONFIGURATION_STATE_KEY",
    "YeastarConfigurationStore",
    "load_effective_yeastar_settings",
    "overlay_yeastar_configuration",
]
