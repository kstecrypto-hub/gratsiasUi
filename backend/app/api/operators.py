from __future__ import annotations

from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Request, status
from redis.exceptions import RedisError
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.auth.dependencies import CurrentUser
from app.api.dependencies import EffectiveYeastarSettings
from app.core.redis import get_redis
from app.database.session import get_db
from app.models import Operator
from app.schemas.common import Page
from app.schemas.operators import OperatorResponse, OperatorSyncResponse, OperatorUpdate
from app.services.audit import audit
from app.services.yeastar import YeastarClient, YeastarConfigurationError
from app.services.yeastar.circuit_breaker import YeastarCircuitBreaker
from app.services.yeastar.errors import YeastarCircuitOpenError, YeastarError
from app.services.yeastar.integration import (
    clear_shared_token_and_require_test,
    is_connected_configuration,
    reconcile_configuration_fingerprint,
)
from app.services.yeastar.schemas import AuthTrigger
from app.services.yeastar.synchronization import synchronize_operators


router = APIRouter(prefix="/operators", tags=["operators"])


@router.get("", response_model=Page[OperatorResponse])
async def list_operators(
    user: CurrentUser,
    db: Annotated[AsyncSession, Depends(get_db)],
    page: int = 1,
    page_size: int = 100,
) -> Page[OperatorResponse]:
    page = max(1, page)
    page_size = min(500, max(1, page_size))
    condition = Operator.deleted_at.is_(None)
    total = await db.scalar(select(func.count()).select_from(Operator).where(condition)) or 0
    rows = (
        await db.scalars(
            select(Operator)
            .where(condition)
            .order_by(Operator.extension_number)
            .offset((page - 1) * page_size)
            .limit(page_size)
        )
    ).all()
    return Page(
        items=[OperatorResponse.model_validate(row) for row in rows],
        total=total,
        page=page,
        page_size=page_size,
    )


@router.post("/sync", response_model=OperatorSyncResponse)
async def sync_operators(
    request: Request,
    user: CurrentUser,
    db: Annotated[AsyncSession, Depends(get_db)],
    settings: EffectiveYeastarSettings,
) -> OperatorSyncResponse:
    redis = get_redis()
    integration, configured, changed_from_existing = await reconcile_configuration_fingerprint(
        db,
        settings,
        clear_token_state=lambda: clear_shared_token_and_require_test(redis),
    )
    await db.commit()
    if not configured:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Phone system not configured.",
        )
    if changed_from_existing:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Phone-system settings changed. Test the connection in Settings.",
        )
    try:
        circuit_state = await YeastarCircuitBreaker(redis).get_state()
    except RedisError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Phone-system connection status is temporarily unavailable.",
        ) from exc
    if not is_connected_configuration(
        integration,
        settings,
        circuit_state=circuit_state,
    ):
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Phone-system connection is not ready. Test the connection in Settings.",
        )
    try:
        async with YeastarClient(
            settings=settings,
            redis=redis,
            trigger=AuthTrigger.OPERATOR_SYNC,
        ) as client:
            result = await synchronize_operators(db, client)
    except YeastarConfigurationError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=(
                "Yeastar integration is not configured. "
                "Add the connection details in Settings."
            ),
        ) from exc
    except YeastarCircuitOpenError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Phone-system connection attempts are paused. Test the connection in Settings.",
        ) from exc
    except YeastarError as exc:
        raise HTTPException(
            status_code=(
                status.HTTP_401_UNAUTHORIZED
                if exc.opens_circuit
                else status.HTTP_503_SERVICE_UNAVAILABLE
            ),
            detail=str(exc),
        ) from exc
    await audit(db, action="operators.sync", request=request, user=user, details={"total": result.total})
    await db.commit()
    return OperatorSyncResponse(**result.__dict__)


@router.patch("/{operator_id}", response_model=OperatorResponse)
async def update_operator(
    operator_id: UUID,
    payload: OperatorUpdate,
    request: Request,
    user: CurrentUser,
    db: Annotated[AsyncSession, Depends(get_db)],
) -> OperatorResponse:
    operator = await db.get(Operator, operator_id)
    if operator is None or operator.deleted_at is not None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Operator not found.")
    operator.enabled = payload.enabled
    await audit(
        db,
        action="operator.enabled" if payload.enabled else "operator.disabled",
        request=request,
        user=user,
        resource_type="operator",
        resource_id=str(operator.id),
    )
    await db.commit()
    await db.refresh(operator)
    return OperatorResponse.model_validate(operator)
