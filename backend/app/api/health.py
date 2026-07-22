from __future__ import annotations

import asyncio
from typing import Annotated

from fastapi import APIRouter, Depends, Response, status
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from app.auth.dependencies import CurrentUser
from app.api.dependencies import (
    EffectiveOpenAISettings,
    EffectiveRuntimeSettings,
    EffectiveYeastarSettings,
)
from app.core.redis import get_redis
from app.database.session import get_db
from app.schemas.common import IntegrationState
from app.models.enums import YeastarConnectionStatus
from app.schemas.configuration import ConfigurationResponse, HealthResponse, YeastarHealthResponse
from app.services.transcription import OpenAITranscriptionClient
from app.services.yeastar.integration import (
    configuration_validation,
    get_integration_status,
    local_connection_status,
    status_response,
)
from app.services.yeastar.circuit_breaker import YeastarCircuitBreaker
from app.workers.celery_app import celery_app


router = APIRouter(tags=["health"])


async def _processing_worker_ready() -> bool:
    try:
        replies = await asyncio.wait_for(
            asyncio.to_thread(celery_app.control.ping, timeout=0.75),
            timeout=1.0,
        )
    except Exception:
        return False
    # Celery replies are keyed by private worker hostnames. Collapse them to a
    # boolean so configuration responses never expose infrastructure identity.
    return bool(replies)


@router.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    return HealthResponse(status="ok")


@router.get("/health/database", response_model=HealthResponse)
async def health_database(
    response: Response, db: Annotated[AsyncSession, Depends(get_db)]
) -> HealthResponse:
    try:
        await db.execute(text("SELECT 1"))
        return HealthResponse(status="ready", service="database")
    except Exception:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
        return HealthResponse(status="unavailable", service="database")


@router.get("/health/redis", response_model=HealthResponse)
async def health_redis(response: Response) -> HealthResponse:
    try:
        await get_redis().ping()
        return HealthResponse(status="ready", service="redis")
    except Exception:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
        return HealthResponse(status="unavailable", service="redis")


@router.get("/health/yeastar", response_model=YeastarHealthResponse)
async def health_yeastar(
    db: Annotated[AsyncSession, Depends(get_db)],
    settings: EffectiveYeastarSettings,
) -> YeastarHealthResponse:
    # Local state only: health checks never authenticate, resolve DNS, or call PBX APIs.
    configured = configuration_validation(settings).valid
    if not configured:
        return YeastarHealthResponse(status=YeastarConnectionStatus.NOT_CONFIGURED)
    row = await get_integration_status(db, create=False)
    fingerprint_matches = bool(
        row is not None
        and row.configuration_fingerprint == settings.yeastar_configuration_fingerprint
    )
    circuit_state = await YeastarCircuitBreaker(get_redis()).get_state()
    return YeastarHealthResponse(
        status=local_connection_status(
            row,
            configured=True,
            fingerprint_matches=fingerprint_matches,
            circuit_state=circuit_state,
        ),
        last_successful_connection_at=(
            row.last_successful_connection_at if row is not None else None
        ),
    )


@router.get("/health/openai", response_model=HealthResponse)
async def health_openai(
    response: Response, settings: EffectiveOpenAISettings
) -> HealthResponse:
    if not settings.openai_configured:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
        return HealthResponse(
            status="not_configured",
            service="openai",
            configured=False,
            message="OpenAI integration is not configured. Add an API key in Settings.",
        )
    try:
        async with OpenAITranscriptionClient(settings=settings) as client:
            await client.test_connection()
        return HealthResponse(status="connected", service="openai", configured=True)
    except Exception:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
        return HealthResponse(status="unavailable", service="openai", configured=True)


@router.get("/configuration", response_model=ConfigurationResponse)
async def configuration(
    user: CurrentUser,
    db: Annotated[AsyncSession, Depends(get_db)],
    settings: EffectiveRuntimeSettings,
) -> ConfigurationResponse:
    circuit_state = None
    try:
        redis = get_redis()
        redis_ready = bool(await redis.ping())
        if redis_ready:
            circuit_state = await YeastarCircuitBreaker(redis).get_state()
    except Exception:
        redis_ready = False
    worker_ready = await _processing_worker_ready() if redis_ready else False
    processing_ready = redis_ready and worker_ready
    yeastar_configured = configuration_validation(settings).valid
    integration = await get_integration_status(db, create=False)
    fingerprint_matches = bool(
        yeastar_configured
        and integration is not None
        and integration.configuration_fingerprint
        == settings.yeastar_configuration_fingerprint
    )
    if not yeastar_configured:
        yeastar_status = YeastarConnectionStatus.NOT_CONFIGURED
        yeastar_message = "Add the connection details in Settings."
    elif not fingerprint_matches:
        yeastar_status = YeastarConnectionStatus.NOT_TESTED
        yeastar_message = "Test the phone-system connection in Settings."
    else:
        yeastar_status = local_connection_status(
            integration,
            configured=True,
            fingerprint_matches=fingerprint_matches,
            circuit_state=circuit_state,
        )
        yeastar_message = (
            None
            if yeastar_status == YeastarConnectionStatus.CONNECTED
            else status_response(
                integration,
                configured=True,
                fingerprint_matches=fingerprint_matches,
                circuit_state=circuit_state,
            ).message
        )
    return ConfigurationResponse(
        administrator_configured=settings.admin_configured,
        application_security_configured=bool(settings.APP_SECRET_KEY),
        yeastar=IntegrationState(
            configured=yeastar_configured,
            status=yeastar_status.value,
            message=yeastar_message,
        ),
        openai=IntegrationState(
            configured=settings.openai_configured,
            status="Ready" if settings.openai_configured else "Not configured",
            message=(
                None
                if settings.openai_configured
                else "OpenAI integration is not configured. Add an API key in Settings."
            ),
        ),
        database=IntegrationState(configured=True, status="Ready"),
        processing=IntegrationState(
            configured=processing_ready,
            status="Ready" if processing_ready else "Unavailable",
            message=(
                None
                if processing_ready
                else "The background processing service is unavailable."
            ),
        ),
    )
