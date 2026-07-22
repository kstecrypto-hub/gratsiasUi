from __future__ import annotations

from datetime import UTC, datetime
from zoneinfo import ZoneInfo


def utc_now() -> datetime:
    return datetime.now(UTC)


def ensure_utc(value: datetime, source_timezone: str = "Europe/Athens") -> datetime:
    if value.tzinfo is None:
        value = value.replace(tzinfo=ZoneInfo(source_timezone))
    return value.astimezone(UTC)


def to_local(value: datetime, timezone: str = "Europe/Athens") -> datetime:
    if value.tzinfo is None:
        value = value.replace(tzinfo=UTC)
    return value.astimezone(ZoneInfo(timezone))
