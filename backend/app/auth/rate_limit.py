from __future__ import annotations

import hashlib

from fastapi import HTTPException, status
from redis.asyncio import Redis

from app.core.config import Settings


class LoginRateLimiter:
    def __init__(self, redis: Redis, settings: Settings) -> None:
        self.redis = redis
        self.settings = settings

    @staticmethod
    def _key(ip_address: str, email: str) -> str:
        identity = f"{ip_address}|{email.strip().lower()}".encode()
        return "yca:login-rate:" + hashlib.sha256(identity).hexdigest()

    async def check(self, ip_address: str, email: str) -> None:
        key = self._key(ip_address, email)
        attempts = await self.redis.get(key)
        if attempts is not None and int(attempts) >= self.settings.login_rate_count:
            ttl = max(1, await self.redis.ttl(key))
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail="Too many sign-in attempts. Try again later.",
                headers={"Retry-After": str(ttl)},
            )

    async def failure(self, ip_address: str, email: str) -> None:
        key = self._key(ip_address, email)
        count = await self.redis.incr(key)
        if count == 1:
            await self.redis.expire(key, self.settings.login_rate_window_seconds)

    async def success(self, ip_address: str, email: str) -> None:
        await self.redis.delete(self._key(ip_address, email))
