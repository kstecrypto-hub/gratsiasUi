from __future__ import annotations

from datetime import datetime
from uuid import UUID

from app.schemas.common import APIModel


class OperatorResponse(APIModel):
    id: UUID
    yeastar_extension_id: str
    extension_number: str
    display_name: str
    email: str | None
    mobile_number: str | None
    presence_status: str | None
    provider_active: bool
    enabled: bool
    last_synced_at: datetime


class OperatorUpdate(APIModel):
    enabled: bool


class OperatorSyncResponse(APIModel):
    created: int
    updated: int
    total: int
    synchronized_at: datetime
