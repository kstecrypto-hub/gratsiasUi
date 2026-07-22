from __future__ import annotations

from typing import Annotated

from fastapi import Depends, HTTPException, Request, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.auth.sessions import SESSION_COOKIE, session_manager
from app.database.session import get_db
from app.models import User


async def get_current_user(
    request: Request,
    db: Annotated[AsyncSession, Depends(get_db)],
) -> User:
    session = await session_manager().read(request.cookies.get(SESSION_COOKIE))
    if session is None:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Sign in required.")
    user = await db.scalar(select(User).where(User.id == session.user_id, User.is_active.is_(True)))
    if user is None:
        await session_manager().destroy(request.cookies.get(SESSION_COOKIE))
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Sign in required.")
    return user


CurrentUser = Annotated[User, Depends(get_current_user)]
