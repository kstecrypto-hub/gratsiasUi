from __future__ import annotations

from uuid import UUID

from pydantic import EmailStr, Field

from app.schemas.common import APIModel


class LoginRequest(APIModel):
    email: EmailStr
    password: str = Field(min_length=1, max_length=1024)
    csrf_token: str | None = Field(default=None, max_length=512)


class UserResponse(APIModel):
    id: UUID
    email: EmailStr


class LoginResponse(APIModel):
    user: UserResponse
    csrf_token: str


class CsrfResponse(APIModel):
    csrf_token: str
