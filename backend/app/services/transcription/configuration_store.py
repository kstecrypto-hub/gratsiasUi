from __future__ import annotations

import base64
import json

from cryptography.fernet import Fernet, InvalidToken
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.hkdf import HKDF
from pydantic import SecretStr
from redis.asyncio import Redis

from app.core.config import Settings
from app.core.logging import register_secret


OPENAI_CONFIGURATION_STATE_KEY = "openai:configuration:state:v1"

_CONFIGURATION_KDF_SALT = b"yeastar-call-analyzer:openai-configuration-state:v1"
_CONFIGURATION_KDF_INFO = b"encrypted-ui-managed-openai-configuration"
_UNREADABLE_CONFIGURATION_MESSAGE = "The saved OpenAI configuration cannot be read safely."


class OpenAIConfigurationStateError(RuntimeError):
    """The encrypted, UI-managed OpenAI configuration cannot be trusted."""


class OpenAIConfiguration:
    """Internal credential value. It is never serialized into API responses."""

    def __init__(self, api_key: str) -> None:
        self.api_key = SecretStr(api_key)

    @property
    def api_key_value(self) -> str:
        return self.api_key.get_secret_value()


def _register_api_key(configuration: OpenAIConfiguration) -> None:
    register_secret(configuration.api_key_value)


class OpenAIConfigurationStore:
    """Durable encrypted storage for the UI-managed OpenAI API key."""

    def __init__(
        self,
        redis: Redis,
        app_secret_key: str,
        *,
        state_key: str = OPENAI_CONFIGURATION_STATE_KEY,
    ) -> None:
        if not app_secret_key:
            raise OpenAIConfigurationStateError(
                "Application security is required to save OpenAI configuration."
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
    def _serialized(configuration: OpenAIConfiguration) -> bytes:
        return json.dumps(
            {"api_key": configuration.api_key_value},
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")

    async def read(self) -> OpenAIConfiguration | None:
        encrypted = await self.redis.get(self.state_key)
        if encrypted is None:
            return None
        try:
            ciphertext = encrypted if isinstance(encrypted, bytes) else encrypted.encode("utf-8")
            plaintext = self._fernet.decrypt(ciphertext)
            payload = json.loads(plaintext)
            api_key = payload["api_key"]
            if not isinstance(api_key, str) or not api_key.strip():
                raise ValueError("invalid api key")
            configuration = OpenAIConfiguration(api_key)
        except (AttributeError, InvalidToken, KeyError, TypeError, ValueError, json.JSONDecodeError):
            # Do not delete unreadable state: silently falling back to an
            # environment key could send customer audio to a different account.
            raise OpenAIConfigurationStateError(_UNREADABLE_CONFIGURATION_MESSAGE) from None
        _register_api_key(configuration)
        return configuration

    def encrypt(self, configuration: OpenAIConfiguration) -> bytes:
        """Return an authenticated ciphertext suitable for an atomic Redis write."""
        _register_api_key(configuration)
        return self._fernet.encrypt(self._serialized(configuration))

    async def write(self, configuration: OpenAIConfiguration) -> None:
        # This is durable application configuration, not an expiring session.
        await self.redis.set(self.state_key, self.encrypt(configuration))

    async def clear(self) -> None:
        await self.redis.delete(self.state_key)


def overlay_openai_configuration(
    base_settings: Settings,
    configuration: OpenAIConfiguration,
) -> Settings:
    """Return Settings with only the UI-managed OpenAI key replaced."""

    _register_api_key(configuration)
    return base_settings.model_copy(update={"OPENAI_API_KEY": configuration.api_key_value})


async def load_effective_openai_settings(
    redis: Redis,
    base_settings: Settings,
) -> Settings:
    """Overlay the saved UI key, or retain the environment fallback when absent."""

    if not base_settings.APP_SECRET_KEY:
        if await redis.get(OPENAI_CONFIGURATION_STATE_KEY) is not None:
            raise OpenAIConfigurationStateError(_UNREADABLE_CONFIGURATION_MESSAGE)
        return base_settings
    store = OpenAIConfigurationStore(redis, base_settings.APP_SECRET_KEY)
    configuration = await store.read()
    if configuration is None:
        return base_settings
    return overlay_openai_configuration(base_settings, configuration)


__all__ = [
    "OPENAI_CONFIGURATION_STATE_KEY",
    "OpenAIConfiguration",
    "OpenAIConfigurationStateError",
    "OpenAIConfigurationStore",
    "load_effective_openai_settings",
    "overlay_openai_configuration",
]
