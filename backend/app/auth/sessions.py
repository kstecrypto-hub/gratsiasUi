from __future__ import annotations

import json
import secrets
from dataclasses import dataclass
from datetime import UTC, datetime
from hmac import compare_digest
from typing import Any
from uuid import UUID

from fastapi import HTTPException, Request, Response, status
from itsdangerous import BadData, SignatureExpired, URLSafeTimedSerializer
from redis.asyncio import Redis

from app.core.config import Settings, get_settings
from app.core.redis import get_redis


SESSION_COOKIE = "yca_session"
CSRF_COOKIE = "yca_csrf"
SESSION_PREFIX = "yca:session:"


@dataclass(frozen=True)
class SessionData:
    session_id: str
    user_id: UUID
    csrf_token: str
    created_at: datetime


class SessionManager:
    def __init__(self, redis: Redis | None = None, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self.redis = redis or get_redis()
        if not self.settings.APP_SECRET_KEY:
            self.serializer: URLSafeTimedSerializer | None = None
        else:
            self.serializer = URLSafeTimedSerializer(
                self.settings.APP_SECRET_KEY,
                salt="yeastar-call-analyzer-session-v1",
            )

    def _require_serializer(self) -> URLSafeTimedSerializer:
        if self.serializer is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Application security is not configured.",
            )
        return self.serializer

    def sign(self, value: str) -> str:
        return self._require_serializer().dumps(value)

    def unsign(self, value: str, max_age: int | None = None) -> str | None:
        try:
            payload = self._require_serializer().loads(
                value, max_age=max_age or self.settings.SESSION_TTL_SECONDS
            )
        except (BadData, SignatureExpired):
            return None
        return payload if isinstance(payload, str) else None

    async def create(self, user_id: UUID) -> tuple[str, str]:
        session_id = secrets.token_urlsafe(32)
        csrf_token = secrets.token_urlsafe(32)
        payload = {
            "user_id": str(user_id),
            "csrf_token": csrf_token,
            "created_at": datetime.now(UTC).isoformat(),
        }
        await self.redis.setex(
            SESSION_PREFIX + session_id,
            self.settings.SESSION_TTL_SECONDS,
            json.dumps(payload, separators=(",", ":")),
        )
        return self.sign(session_id), csrf_token

    async def read(self, signed_session_id: str | None, *, touch: bool = True) -> SessionData | None:
        if not signed_session_id:
            return None
        session_id = self.unsign(signed_session_id)
        if not session_id:
            return None
        key = SESSION_PREFIX + session_id
        raw = await self.redis.get(key)
        if raw is None:
            return None
        try:
            payload: dict[str, Any] = json.loads(raw)
            data = SessionData(
                session_id=session_id,
                user_id=UUID(payload["user_id"]),
                csrf_token=str(payload["csrf_token"]),
                created_at=datetime.fromisoformat(payload["created_at"]),
            )
        except (KeyError, TypeError, ValueError, json.JSONDecodeError):
            await self.redis.delete(key)
            return None
        if touch:
            await self.redis.expire(key, self.settings.SESSION_TTL_SECONDS)
        return data

    async def destroy(self, signed_session_id: str | None) -> None:
        if not signed_session_id:
            return
        session_id = self.unsign(signed_session_id)
        if session_id:
            await self.redis.delete(SESSION_PREFIX + session_id)

    def set_auth_cookies(
        self, response: Response, signed_session_id: str, csrf_token: str
    ) -> None:
        common = {
            "secure": self.settings.cookie_secure,
            "samesite": "strict",
            "path": "/",
            "max_age": self.settings.SESSION_TTL_SECONDS,
        }
        response.set_cookie(SESSION_COOKIE, signed_session_id, httponly=True, **common)
        self.set_csrf_cookie(response, csrf_token)

    def set_csrf_cookie(self, response: Response, csrf_token: str) -> None:
        response.set_cookie(
            CSRF_COOKIE,
            self.sign(csrf_token),
            httponly=False,
            secure=self.settings.cookie_secure,
            samesite="strict",
            path="/",
            max_age=self.settings.SESSION_TTL_SECONDS,
        )

    def set_preauth_csrf_cookie(self, response: Response) -> str:
        raw = secrets.token_urlsafe(32)
        response.set_cookie(
            CSRF_COOKIE,
            self.sign(raw),
            httponly=False,
            secure=self.settings.cookie_secure,
            samesite="strict",
            path="/",
            max_age=3600,
        )
        return raw

    def clear_cookies(self, response: Response) -> None:
        response.delete_cookie(SESSION_COOKIE, path="/", secure=self.settings.cookie_secure)
        response.delete_cookie(CSRF_COOKIE, path="/", secure=self.settings.cookie_secure)

    def clear_session_cookie(self, response: Response) -> None:
        response.delete_cookie(SESSION_COOKIE, path="/", secure=self.settings.cookie_secure)

    async def validate_csrf(self, request: Request) -> bool:
        signed_csrf = request.cookies.get(CSRF_COOKIE)
        submitted = request.headers.get("X-CSRF-Token")
        if not signed_csrf or not submitted:
            return False
        cookie_value = self.unsign(signed_csrf, max_age=self.settings.SESSION_TTL_SECONDS)
        if not cookie_value or not compare_digest(cookie_value, submitted):
            return False
        signed_session = request.cookies.get(SESSION_COOKIE)
        if signed_session:
            session = await self.read(signed_session, touch=False)
            if not session or not compare_digest(session.csrf_token, submitted):
                return False
        return True


def session_manager() -> SessionManager:
    return SessionManager()
