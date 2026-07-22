from __future__ import annotations

from typing import Any

from fastapi import Request
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.logging import redact_value
from app.core.time import utc_now
from app.models import AuditLog, User


def request_ip(request: Request) -> str | None:
    # Proxy headers are trusted only at the ASGI server boundary. Avoid parsing arbitrary chains here.
    return request.client.host if request.client else None


async def audit(
    session: AsyncSession,
    *,
    action: str,
    request: Request | None = None,
    user: User | None = None,
    resource_type: str | None = None,
    resource_id: str | None = None,
    outcome: str = "success",
    details: dict[str, Any] | None = None,
) -> AuditLog:
    entry = AuditLog(
        created_at=utc_now(),
        user_id=user.id if user else None,
        action=action,
        resource_type=resource_type,
        resource_id=resource_id,
        outcome=outcome,
        ip_address=request_ip(request) if request else None,
        user_agent=(request.headers.get("user-agent", "")[:512] if request else None),
        details=redact_value(details or {}),
    )
    session.add(entry)
    await session.flush()
    return entry
