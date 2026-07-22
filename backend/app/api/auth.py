from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Request, Response, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.auth.dependencies import CurrentUser
from app.auth.passwords import hash_password, verify_password
from app.auth.rate_limit import LoginRateLimiter
from app.auth.sessions import SESSION_COOKIE, session_manager
from app.core.config import Settings, get_settings
from app.core.redis import get_redis
from app.core.time import utc_now
from app.database.session import get_db
from app.models import User
from app.schemas.auth import CsrfResponse, LoginRequest, LoginResponse, UserResponse
from app.schemas.common import MessageResponse
from app.services.audit import audit, request_ip


router = APIRouter(prefix="/auth", tags=["authentication"])
_DUMMY_HASH = hash_password("not-a-real-user-password")


@router.get("/csrf", response_model=CsrfResponse)
async def csrf(request: Request, response: Response) -> CsrfResponse:
    manager = session_manager()
    signed_session = request.cookies.get(SESSION_COOKIE)
    if signed_session:
        existing = await manager.read(signed_session)
        if existing is not None:
            manager.set_csrf_cookie(response, existing.csrf_token)
            return CsrfResponse(csrf_token=existing.csrf_token)
        # An expired Redis session must not leave the browser unable to perform
        # a pre-authenticated login protected by the double-submit token.
        manager.clear_session_cookie(response)
    token = manager.set_preauth_csrf_cookie(response)
    return CsrfResponse(csrf_token=token)


@router.post("/login", response_model=LoginResponse)
async def login(
    payload: LoginRequest,
    request: Request,
    response: Response,
    db: Annotated[AsyncSession, Depends(get_db)],
    settings: Annotated[Settings, Depends(get_settings)],
) -> LoginResponse:
    if not settings.admin_configured:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Administrator account is not configured.",
        )
    ip = request_ip(request) or "unknown"
    email = str(payload.email).strip().lower()
    limiter = LoginRateLimiter(get_redis(), settings)
    await limiter.check(ip, email)
    user = await db.scalar(select(User).where(User.email == email, User.is_active.is_(True)))
    valid, needs_rehash = verify_password(user.password_hash if user else _DUMMY_HASH, payload.password)
    if user is None or not valid:
        await limiter.failure(ip, email)
        await audit(
            db,
            action="login",
            request=request,
            outcome="failure",
            details={"email_hash": __import__("hashlib").sha256(email.encode()).hexdigest()},
        )
        await db.commit()
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Email or password is incorrect.",
        )
    await limiter.success(ip, email)
    if needs_rehash:
        user.password_hash = hash_password(payload.password)
    user.last_login_at = utc_now()
    signed_session, csrf_token = await session_manager().create(user.id)
    session_manager().set_auth_cookies(response, signed_session, csrf_token)
    await audit(db, action="login", request=request, user=user)
    await db.commit()
    return LoginResponse(user=UserResponse.model_validate(user), csrf_token=csrf_token)


@router.post("/logout", response_model=MessageResponse)
async def logout(
    request: Request,
    response: Response,
    user: CurrentUser,
    db: Annotated[AsyncSession, Depends(get_db)],
) -> MessageResponse:
    await session_manager().destroy(request.cookies.get(SESSION_COOKIE))
    session_manager().clear_cookies(response)
    await audit(db, action="logout", request=request, user=user)
    await db.commit()
    return MessageResponse(message="Signed out.")


@router.get("/me", response_model=UserResponse)
async def me(user: CurrentUser) -> UserResponse:
    return UserResponse.model_validate(user)
