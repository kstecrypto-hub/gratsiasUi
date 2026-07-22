from __future__ import annotations

from typing import Any
from uuid import UUID
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import Settings
from app.models import ApplicationSetting


ALLOWED_SETTINGS = {
    "default_language",
    "transcript_retention_days",
    "delete_audio_after_transcription",
    "max_parallel_transcriptions",
    "company_vocabulary",
    "default_timezone",
}


def environment_defaults(settings: Settings) -> dict[str, Any]:
    return {
        "default_language": settings.TRANSCRIPTION_LANGUAGE,
        "transcript_retention_days": settings.TRANSCRIPT_RETENTION_DAYS,
        "delete_audio_after_transcription": settings.DELETE_AUDIO_AFTER_TRANSCRIPTION,
        "max_parallel_transcriptions": settings.MAX_PARALLEL_TRANSCRIPTIONS,
        "company_vocabulary": "",
        "default_timezone": settings.APP_TIMEZONE,
    }


async def load_application_settings(session: AsyncSession, settings: Settings) -> dict[str, Any]:
    result = environment_defaults(settings)
    rows = (await session.scalars(select(ApplicationSetting))).all()
    for row in rows:
        if row.key in ALLOWED_SETTINGS:
            result[row.key] = row.value
    return result


async def update_application_settings(
    session: AsyncSession,
    settings: Settings,
    values: dict[str, Any],
    user_id: UUID,
) -> dict[str, Any]:
    values = {key: value for key, value in values.items() if key in ALLOWED_SETTINGS and value is not None}
    if "default_timezone" in values:
        try:
            ZoneInfo(str(values["default_timezone"]))
        except ZoneInfoNotFoundError as exc:
            raise ValueError("Unknown timezone") from exc
    existing = {
        row.key: row
        for row in (await session.scalars(select(ApplicationSetting))).all()
    }
    for key, value in values.items():
        row = existing.get(key)
        if row is None:
            row = ApplicationSetting(key=key, value=value, updated_by_id=user_id)
            session.add(row)
        else:
            row.value = value
            row.updated_by_id = user_id
    await session.flush()
    return await load_application_settings(session, settings)
