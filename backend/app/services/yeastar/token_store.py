from __future__ import annotations

import asyncio
import base64
import json
import math
import secrets
import time
from datetime import UTC, datetime

from cryptography.fernet import Fernet, InvalidToken
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.hkdf import HKDF
from redis.asyncio import Redis

from app.core.logging import register_secret
from app.services.yeastar.errors import YeastarLockTimeoutError
from app.services.yeastar.schemas import TokenState


YEASTAR_TOKEN_STATE_KEY = "yeastar:auth:state:v1"
YEASTAR_TOKEN_LOCK_KEY = "yeastar:auth:lock:v1"

_RELEASE_SCRIPT = """
if redis.call('get', KEYS[1]) == ARGV[1] then
  return redis.call('del', KEYS[1])
end
return 0
"""


def _as_text(value: bytes | str) -> str:
    return value.decode("utf-8") if isinstance(value, bytes) else value


class RedisOwnerLock:
    """Bounded Redis lock with owner-checked atomic release."""

    def __init__(
        self,
        redis: Redis,
        key: str = YEASTAR_TOKEN_LOCK_KEY,
        *,
        timeout_seconds: float = 30.0,
        wait_seconds: float = 15.0,
    ) -> None:
        self.redis = redis
        self.key = key
        self.timeout_seconds = max(0.1, timeout_seconds)
        self.wait_seconds = max(0.0, wait_seconds)
        self.owner: str | None = None
        self.acquired = False

    async def acquire(self) -> bool:
        owner = secrets.token_urlsafe(32)
        deadline = time.monotonic() + self.wait_seconds
        ttl_ms = max(1, math.ceil(self.timeout_seconds * 1000))
        while True:
            acquired = await self.redis.set(self.key, owner, nx=True, px=ttl_ms)
            if bool(acquired):
                self.owner = owner
                self.acquired = True
                return True
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise YeastarLockTimeoutError(
                    "Phone-system authentication is already in progress."
                )
            await asyncio.sleep(min(0.05, remaining))

    async def owned(self) -> bool:
        if not self.acquired or self.owner is None:
            return False
        current = await self.redis.get(self.key)
        if current is None:
            return False
        return secrets.compare_digest(_as_text(current), self.owner)

    async def release(self) -> bool:
        if not self.acquired or self.owner is None:
            return False
        owner = self.owner
        try:
            removed = await self.redis.eval(_RELEASE_SCRIPT, 1, self.key, owner)
            return bool(removed)
        finally:
            self.acquired = False
            self.owner = None

    async def __aenter__(self) -> "RedisOwnerLock":
        await self.acquire()
        return self

    async def __aexit__(self, *_: object) -> None:
        await self.release()


class YeastarTokenStore:
    def __init__(
        self,
        redis: Redis,
        app_secret_key: str,
        *,
        configuration_fingerprint: str | None = None,
        state_key: str = YEASTAR_TOKEN_STATE_KEY,
        lock_key: str = YEASTAR_TOKEN_LOCK_KEY,
        lock_timeout_seconds: float = 180.0,
        lock_wait_seconds: float = 15.0,
    ) -> None:
        if not app_secret_key:
            raise ValueError("APP_SECRET_KEY is required for encrypted Yeastar token state")
        self.redis = redis
        self.configuration_fingerprint = configuration_fingerprint
        self.state_key = state_key
        self.lock_key = lock_key
        self.lock_timeout_seconds = lock_timeout_seconds
        self.lock_wait_seconds = lock_wait_seconds
        derived = HKDF(
            algorithm=hashes.SHA256(),
            length=32,
            salt=b"yeastar-call-analyzer:token-state:v1",
            info=b"encrypted-shared-yeastar-token-state",
        ).derive(app_secret_key.encode("utf-8"))
        self._fernet = Fernet(base64.urlsafe_b64encode(derived))

    def owner_lock(self) -> RedisOwnerLock:
        return RedisOwnerLock(
            self.redis,
            self.lock_key,
            timeout_seconds=self.lock_timeout_seconds,
            wait_seconds=self.lock_wait_seconds,
        )

    @staticmethod
    def _serialized(state: TokenState) -> bytes:
        payload = {
            "access_token": state.access_token_value,
            "access_token_expires_at": state.access_token_expires_at.astimezone(UTC).isoformat(),
            "refresh_token": state.refresh_token_value,
            "refresh_token_expires_at": state.refresh_token_expires_at.astimezone(UTC).isoformat(),
            "issued_at": state.issued_at.astimezone(UTC).isoformat(),
            "generation": state.generation,
            "configuration_fingerprint": state.configuration_fingerprint,
        }
        return json.dumps(payload, separators=(",", ":"), sort_keys=True).encode("utf-8")

    async def read(self) -> TokenState | None:
        encrypted = await self.redis.get(self.state_key)
        if encrypted is None:
            return None
        try:
            plaintext = self._fernet.decrypt(
                encrypted if isinstance(encrypted, bytes) else encrypted.encode("utf-8")
            )
            state = TokenState.model_validate_json(plaintext)
        except (InvalidToken, ValueError, TypeError):
            await self.redis.delete(self.state_key)
            return None
        register_secret(state.access_token_value)
        register_secret(state.refresh_token_value)
        return state

    async def write(self, state: TokenState) -> None:
        register_secret(state.access_token_value)
        register_secret(state.refresh_token_value)
        remaining = (
            state.refresh_token_expires_at.astimezone(UTC) - datetime.now(UTC)
        ).total_seconds()
        ttl = max(1, math.ceil(remaining))
        encrypted = self._fernet.encrypt(self._serialized(state))
        await self.redis.set(self.state_key, encrypted, ex=ttl)

    async def clear(self) -> None:
        await self.redis.delete(self.state_key)

    async def invalidate_access_token(
        self, expected_access_token: str | None = None
    ) -> bool:
        async with self.owner_lock():
            state = await self.read()
            if state is None:
                return False
            if expected_access_token is not None and not secrets.compare_digest(
                state.access_token_value, expected_access_token
            ):
                return False
            invalidated = state.model_copy(
                update={"access_token_expires_at": datetime.now(UTC)}
            )
            await self.write(invalidated)
            return True


__all__ = [
    "RedisOwnerLock",
    "YEASTAR_TOKEN_LOCK_KEY",
    "YEASTAR_TOKEN_STATE_KEY",
    "YeastarTokenStore",
]
