from __future__ import annotations

from typing import Annotated

from fastapi import Depends, HTTPException, status

from app.core.config import Settings, get_settings
from app.core.redis import get_redis
from app.services.transcription.configuration_store import load_effective_openai_settings
from app.services.yeastar.configuration_store import load_effective_yeastar_settings


async def get_effective_yeastar_settings(
    settings: Annotated[Settings, Depends(get_settings)],
) -> Settings:
    """Resolve the local UI override without contacting the phone system."""
    try:
        return await load_effective_yeastar_settings(get_redis(), settings)
    except Exception as exc:
        # Configuration storage is security-sensitive. Never fall back to a
        # potentially different environment configuration when Redis cannot be
        # read or an encrypted value fails authentication.
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Phone-system configuration is temporarily unavailable.",
        ) from exc


EffectiveYeastarSettings = Annotated[
    Settings,
    Depends(get_effective_yeastar_settings),
]


async def get_effective_openai_settings(
    settings: Annotated[Settings, Depends(get_settings)],
) -> Settings:
    """Resolve the local UI OpenAI override without contacting OpenAI."""
    try:
        return await load_effective_openai_settings(get_redis(), settings)
    except Exception as exc:
        # As with the phone-system override, a failed encrypted-state read must
        # never silently fall back to an environment key for a different account.
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="OpenAI configuration is temporarily unavailable.",
        ) from exc


EffectiveOpenAISettings = Annotated[
    Settings,
    Depends(get_effective_openai_settings),
]


async def get_effective_runtime_settings(
    settings: Annotated[Settings, Depends(get_settings)],
) -> Settings:
    """Resolve both UI-managed integrations for a request that needs both."""
    effective_yeastar = await get_effective_yeastar_settings(settings)
    return await get_effective_openai_settings(effective_yeastar)


EffectiveRuntimeSettings = Annotated[
    Settings,
    Depends(get_effective_runtime_settings),
]


__all__ = [
    "EffectiveOpenAISettings",
    "EffectiveRuntimeSettings",
    "EffectiveYeastarSettings",
    "get_effective_openai_settings",
    "get_effective_runtime_settings",
    "get_effective_yeastar_settings",
]
