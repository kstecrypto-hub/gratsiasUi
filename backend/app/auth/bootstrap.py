from __future__ import annotations

import logging

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.auth.passwords import hash_password
from app.core.config import Settings
from app.models import User


logger = logging.getLogger(__name__)


async def ensure_admin(session: AsyncSession, settings: Settings) -> bool:
    count = await session.scalar(select(func.count()).select_from(User))
    if count:
        return False
    if not settings.admin_configured:
        logger.warning("Administrator account is not configured; set ADMIN_EMAIL and ADMIN_PASSWORD")
        return False
    user = User(
        email=str(settings.ADMIN_EMAIL).strip().lower(),
        password_hash=hash_password(str(settings.ADMIN_PASSWORD)),
        is_active=True,
    )
    session.add(user)
    await session.commit()
    logger.info("Initial administrator account created")
    return True
