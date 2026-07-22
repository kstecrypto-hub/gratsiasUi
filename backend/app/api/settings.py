from __future__ import annotations

import logging
from datetime import UTC, datetime
from typing import Annotated

from fastapi import APIRouter, Body, Depends, HTTPException, Request, status
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse
from pydantic import ValidationError
from redis.asyncio import Redis
from redis.exceptions import RedisError
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.auth.dependencies import CurrentUser
from app.api.dependencies import EffectiveYeastarSettings
from app.core.config import Settings, get_settings
from app.core.redis import get_redis
from app.database.session import get_db
from app.models import ProcessingJob, ProcessingJobItem
from app.models.enums import ItemStatus, JobStatus, YeastarConnectionStatus
from app.schemas.configuration import (
    OpenAIConfigurationUpdate,
    OpenAIConfigurationValidationError,
    OpenAIConfigurationValidationResponse,
    OpenAIConnectionTestResponse,
    OpenAIConnectionTestResult,
    SanitizedOpenAIConfiguration,
    SanitizedYeastarConnectionConfig,
    YeastarConfigurationValidationError,
    YeastarConfigurationValidationResponse,
    YeastarConnectionConfigurationUpdate,
    YeastarConnectionTestResponse,
    YeastarConnectionTestResult,
    YeastarStatusResponse,
)
from app.schemas.settings import ApplicationSettingsResponse, ApplicationSettingsUpdate
from app.services.application_settings import load_application_settings, update_application_settings
from app.services.audit import audit
from app.services.transcription.client import OpenAITranscriptionClient, TranscriptionError
from app.services.transcription.configuration_store import (
    OpenAIConfiguration,
    OpenAIConfigurationStateError,
    OpenAIConfigurationStore,
    load_effective_openai_settings,
)
from app.services.yeastar.client import YeastarClient
from app.services.yeastar.configuration_store import (
    YeastarConfigurationStore,
    load_effective_yeastar_settings,
)
from app.services.yeastar.errors import (
    YeastarConfigurationStateError,
    YeastarError,
    YeastarLockTimeoutError,
)
from app.services.yeastar.integration import (
    clear_shared_token_and_require_test,
    clear_detected_metadata,
    connection_status_for_error,
    configuration_validation,
    get_integration_status,
    record_connection_failure,
    record_connection_success,
    reconcile_configuration_fingerprint,
    safe_configuration,
    status_response,
)
from app.services.yeastar.schemas import AuthTrigger, CircuitBreakerState, ConnectionState
from app.services.yeastar.circuit_breaker import YEASTAR_CIRCUIT_KEY, YeastarCircuitBreaker
from app.services.yeastar.token_manager import YeastarTokenManager
from app.services.yeastar.token_store import (
    RedisOwnerLock,
    YEASTAR_TOKEN_STATE_KEY,
)


router = APIRouter(prefix="/settings", tags=["settings"])
logger = logging.getLogger(__name__)
YEASTAR_MANUAL_TEST_LOCK = "yeastar:manual-connection-test:lock:v1"
OPENAI_CONFIGURATION_ACTION_LOCK = "openai:configuration:action:lock:v1"
_PRESERVE_CREDENTIAL_MARKERS = {
    "",
    "[CONFIGURED]",
    "[NOT CONFIGURED]",
    "[REDACTED]",
}
_PRESERVE_OPENAI_KEY_MARKERS = {
    "",
    "[CONFIGURED]",
    "[NOT CONFIGURED]",
    "[REDACTED]",
}


def _manual_action_lock_timeout(settings: Settings) -> float:
    # system/information, extension/list and conditional CDR verification can
    # each consume an initial attempt plus the one bounded transport retry.
    return max(
        120.0,
        8
        * (
            settings.yeastar_connect_timeout_seconds
            + settings.yeastar_read_timeout_seconds
        )
        + 60.0,
    )


async def _clear_shared_token_state() -> None:
    await clear_shared_token_and_require_test(get_redis())


async def _load_effective_configuration(redis: Redis, base: Settings) -> Settings:
    try:
        return await load_effective_yeastar_settings(redis, base)
    except (RedisError, YeastarConfigurationStateError) as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Phone-system configuration is temporarily unavailable.",
        ) from exc


async def _load_effective_openai_configuration(redis: Redis, base: Settings) -> Settings:
    try:
        return await load_effective_openai_settings(redis, base)
    except (RedisError, OpenAIConfigurationStateError) as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="OpenAI configuration is temporarily unavailable.",
        ) from exc


def _safe_openai_configuration(settings: Settings) -> SanitizedOpenAIConfiguration:
    return SanitizedOpenAIConfiguration(
        api_key="[CONFIGURED]" if settings.openai_configured else "[NOT CONFIGURED]"
    )


def _openai_validation_response(
    settings: Settings,
    *,
    message: str | None = None,
) -> OpenAIConfigurationValidationResponse:
    errors = (
        []
        if settings.openai_configured and message is None
        else [
            OpenAIConfigurationValidationError(
                field="api_key",
                message=message or "Enter an OpenAI API key.",
            )
        ]
    )
    return OpenAIConfigurationValidationResponse(
        valid=not errors,
        errors=errors,
        configuration=_safe_openai_configuration(settings),
    )


def _openai_payload_validation_response(
    existing: Settings,
    _error: ValidationError,
) -> OpenAIConfigurationValidationResponse:
    # Do not include rejected input or Pydantic's context: either could contain
    # a submitted API key.
    return _openai_validation_response(existing, message="Enter a valid OpenAI API key.")


def _configuration_update_settings(
    base: Settings,
    existing: Settings,
    payload: YeastarConnectionConfigurationUpdate,
) -> Settings:
    existing_config = existing.yeastar_connection_config
    submitted_client_id = payload.Settings.ClientId.strip()
    submitted_secret = payload.Settings.ClientSecret.get_secret_value()
    client_id = (
        existing_config.Settings.ClientId
        if submitted_client_id.upper() in _PRESERVE_CREDENTIAL_MARKERS
        else submitted_client_id
    )
    client_secret = (
        existing_config.Settings.ClientSecret.get_secret_value()
        if submitted_secret == "" or submitted_secret.upper() in _PRESERVE_CREDENTIAL_MARKERS
        else submitted_secret
    )
    return base.model_copy(
        update={
            "YEASTAR_NAME": payload.Name.strip(),
            "YEASTAR_BASE_URL": payload.Settings.BaseUrl.strip().rstrip("/"),
            "YEASTAR_CLIENT_ID": client_id,
            "YEASTAR_CLIENT_SECRET": client_secret,
            "YEASTAR_DATE_FORMAT": payload.Settings.DateFormat.strip(),
            "YEASTAR_PAGE_SIZE": payload.Settings.PageSize,
            "YEASTAR_IGNORE_SSL_ERRORS": payload.Settings.IgnoreSslErrors,
        }
    )


_CONFIGURATION_FIELD_MESSAGES = {
    "Name": "Enter a valid phone-system name.",
    "Settings": "Use the required phone-system settings format.",
    "Settings.BaseUrl": "A valid phone-system URL is required.",
    "Settings.ClientId": "A valid phone-system Client ID is required.",
    "Settings.ClientSecret": "A valid phone-system Client Secret is required.",
    "Settings.DateFormat": "Enter a valid phone-system date format.",
    "Settings.PageSize": "Page size must be between 1 and 10000.",
    "Settings.IgnoreSslErrors": "Choose a valid certificate setting.",
}


def _payload_validation_response(
    existing: Settings,
    error: ValidationError,
) -> YeastarConfigurationValidationResponse:
    errors: list[YeastarConfigurationValidationError] = []
    seen: set[str] = set()
    for item in error.errors(
        include_url=False,
        include_context=False,
        include_input=False,
    ):
        field = ".".join(str(part) for part in item.get("loc", ())) or "Configuration"
        if field in seen:
            continue
        seen.add(field)
        errors.append(
            YeastarConfigurationValidationError(
                field=field,
                message=_CONFIGURATION_FIELD_MESSAGES.get(
                    field,
                    "Enter a valid phone-system configuration value.",
                ),
            )
        )
    if not errors:
        errors.append(
            YeastarConfigurationValidationError(
                field="Configuration",
                message="Use the required phone-system configuration format.",
            )
        )
    return YeastarConfigurationValidationResponse(
        valid=False,
        errors=errors,
        configuration=safe_configuration(existing),
    )


def _strict_type_validation_response(
    existing: Settings,
    update: YeastarConnectionConfigurationUpdate,
) -> YeastarConfigurationValidationResponse | None:
    errors: list[YeastarConfigurationValidationError] = []
    if type(update.Settings.PageSize) is not int:
        errors.append(
            YeastarConfigurationValidationError(
                field="Settings.PageSize",
                message=_CONFIGURATION_FIELD_MESSAGES["Settings.PageSize"],
            )
        )
    if type(update.Settings.IgnoreSslErrors) is not bool:
        errors.append(
            YeastarConfigurationValidationError(
                field="Settings.IgnoreSslErrors",
                message=_CONFIGURATION_FIELD_MESSAGES["Settings.IgnoreSslErrors"],
            )
        )
    if not errors:
        return None
    return YeastarConfigurationValidationResponse(
        valid=False,
        errors=errors,
        configuration=safe_configuration(existing),
    )


@router.get("", response_model=ApplicationSettingsResponse)
async def get_application_settings(
    user: CurrentUser,
    db: Annotated[AsyncSession, Depends(get_db)],
    settings: Annotated[Settings, Depends(get_settings)],
) -> ApplicationSettingsResponse:
    return ApplicationSettingsResponse(**(await load_application_settings(db, settings)))


@router.patch("", response_model=ApplicationSettingsResponse)
async def patch_application_settings(
    payload: ApplicationSettingsUpdate,
    request: Request,
    user: CurrentUser,
    db: Annotated[AsyncSession, Depends(get_db)],
    settings: Annotated[Settings, Depends(get_settings)],
) -> ApplicationSettingsResponse:
    changes = payload.model_dump(
        exclude_none=True, exclude={"maximum_simultaneous_transcriptions"}
    )
    try:
        result = await update_application_settings(db, settings, changes, user.id)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)) from exc
    await audit(
        db,
        action="settings.update",
        request=request,
        user=user,
        resource_type="application_settings",
        details={"changed_keys": sorted(changes)},
    )
    await db.commit()
    return ApplicationSettingsResponse(**result)


@router.get(
    "/openai/configuration",
    response_model=SanitizedOpenAIConfiguration,
)
async def get_openai_configuration(
    user: CurrentUser,
    settings: Annotated[Settings, Depends(get_settings)],
) -> SanitizedOpenAIConfiguration:
    effective = await _load_effective_openai_configuration(get_redis(), settings)
    return _safe_openai_configuration(effective)


@router.put(
    "/openai/configuration",
    response_model=OpenAIConfigurationValidationResponse,
)
async def update_openai_configuration(
    payload: Annotated[object, Body()],
    request: Request,
    user: CurrentUser,
    db: Annotated[AsyncSession, Depends(get_db)],
    settings: Annotated[Settings, Depends(get_settings)],
) -> OpenAIConfigurationValidationResponse | JSONResponse:
    """Save a write-only encrypted OpenAI key without contacting OpenAI."""

    redis = get_redis()
    action_lock = RedisOwnerLock(
        redis,
        key=OPENAI_CONFIGURATION_ACTION_LOCK,
        timeout_seconds=30.0,
        wait_seconds=0,
    )
    try:
        await action_lock.acquire()
    except YeastarLockTimeoutError as exc:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="An OpenAI configuration action is already running.",
        ) from exc

    try:
        existing = await _load_effective_openai_configuration(redis, settings)
        try:
            update = (
                payload
                if isinstance(payload, OpenAIConfigurationUpdate)
                else OpenAIConfigurationUpdate.model_validate(payload)
            )
        except ValidationError as exc:
            invalid = _openai_payload_validation_response(existing, exc)
            return JSONResponse(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                content=jsonable_encoder(invalid),
            )

        submitted_key = update.api_key.get_secret_value().strip()
        if submitted_key.upper() in _PRESERVE_OPENAI_KEY_MARKERS:
            validation = _openai_validation_response(existing)
            if validation.valid:
                return validation
            return JSONResponse(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                content=jsonable_encoder(validation),
            )
        if not settings.APP_SECRET_KEY:
            raise OpenAIConfigurationStateError(
                "Application security is required to save OpenAI configuration."
            )

        candidate = OpenAIConfiguration(submitted_key)
        configuration_changed = (existing.OPENAI_API_KEY or "") != submitted_key
        store = OpenAIConfigurationStore(redis, settings.APP_SECRET_KEY)
        await store.write(candidate)
        # The in-memory candidate has the same key that was persisted. Avoid a
        # second Redis read while constructing the safe response.
        current = existing.model_copy(update={"OPENAI_API_KEY": submitted_key})
        validation = _openai_validation_response(current)
        await audit(
            db,
            action="openai.configuration.update",
            request=request,
            user=user,
            resource_type="integration_configuration",
            resource_id="openai",
            details={
                "configuration_changed": configuration_changed,
                "configured": True,
            },
        )
        await db.commit()
        return validation
    except (RedisError, OpenAIConfigurationStateError) as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="OpenAI configuration could not be saved safely.",
        ) from exc
    finally:
        await action_lock.release()


@router.post(
    "/openai/test",
    response_model=OpenAIConnectionTestResponse | OpenAIConfigurationValidationResponse,
)
async def test_openai_configuration(
    request: Request,
    user: CurrentUser,
    db: Annotated[AsyncSession, Depends(get_db)],
    settings: Annotated[Settings, Depends(get_settings)],
) -> OpenAIConnectionTestResponse | JSONResponse:
    """Verify the saved credential without uploading audio or returning the key."""

    redis = get_redis()
    action_lock = RedisOwnerLock(
        redis,
        key=OPENAI_CONFIGURATION_ACTION_LOCK,
        timeout_seconds=300.0,
        wait_seconds=0,
    )
    try:
        await action_lock.acquire()
    except YeastarLockTimeoutError as exc:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="An OpenAI configuration action is already running.",
        ) from exc

    try:
        effective = await _load_effective_openai_configuration(redis, settings)
        validation = _openai_validation_response(effective)
        if not validation.valid:
            return JSONResponse(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                content=jsonable_encoder(validation),
            )
        try:
            async with OpenAITranscriptionClient(settings=effective) as client:
                await client.test_connection()
        except TranscriptionError as exc:
            category = exc.category
            await audit(
                db,
                action="openai.connection.test",
                request=request,
                user=user,
                resource_type="integration_configuration",
                resource_id="openai",
                outcome="failure",
                details={"category": category},
            )
            await db.commit()
            raise HTTPException(
                status_code=(
                    status.HTTP_401_UNAUTHORIZED
                    if category == "openai_authentication"
                    else status.HTTP_503_SERVICE_UNAVAILABLE
                ),
                detail=(
                    "OpenAI credentials were rejected."
                    if category == "openai_authentication"
                    else "OpenAI connection test could not be completed."
                ),
            ) from exc
        except Exception as exc:
            await audit(
                db,
                action="openai.connection.test",
                request=request,
                user=user,
                resource_type="integration_configuration",
                resource_id="openai",
                outcome="failure",
                details={"category": "openai_unexpected"},
            )
            await db.commit()
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="OpenAI connection test could not be completed.",
            ) from exc

        await audit(
            db,
            action="openai.connection.test",
            request=request,
            user=user,
            resource_type="integration_configuration",
            resource_id="openai",
            details={"status": "connected"},
        )
        await db.commit()
        return OpenAIConnectionTestResponse(
            configurationAccepted=True,
            configuration=validation.configuration,
            connection=OpenAIConnectionTestResult(status="connected"),
        )
    finally:
        await action_lock.release()


@router.get(
    "/yeastar/configuration",
    response_model=SanitizedYeastarConnectionConfig,
)
async def get_yeastar_configuration(
    user: CurrentUser,
    settings: EffectiveYeastarSettings,
) -> SanitizedYeastarConnectionConfig:
    return safe_configuration(settings)


@router.put(
    "/yeastar/configuration",
    response_model=YeastarConfigurationValidationResponse,
)
async def update_yeastar_configuration(
    payload: Annotated[object, Body()],
    request: Request,
    user: CurrentUser,
    db: Annotated[AsyncSession, Depends(get_db)],
    settings: Annotated[Settings, Depends(get_settings)],
) -> YeastarConfigurationValidationResponse | JSONResponse:
    """Save an encrypted local override without contacting the phone system."""

    redis = get_redis()
    action_lock = RedisOwnerLock(
        redis,
        key=YEASTAR_MANUAL_TEST_LOCK,
        timeout_seconds=_manual_action_lock_timeout(settings),
        wait_seconds=0,
    )
    try:
        await action_lock.acquire()
    except YeastarLockTimeoutError as exc:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="A phone-system connection action is already running.",
        ) from exc

    try:
        existing = await load_effective_yeastar_settings(redis, settings)
        try:
            update = (
                payload
                if isinstance(payload, YeastarConnectionConfigurationUpdate)
                else YeastarConnectionConfigurationUpdate.model_validate(payload)
            )
        except ValidationError as exc:
            invalid = _payload_validation_response(existing, exc)
            return JSONResponse(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                content=jsonable_encoder(invalid),
            )
        strict_type_error = _strict_type_validation_response(existing, update)
        if strict_type_error is not None:
            return JSONResponse(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                content=jsonable_encoder(strict_type_error),
            )
        candidate = _configuration_update_settings(settings, existing, update)
        validation = configuration_validation(candidate)
        if not validation.valid:
            return JSONResponse(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                content=jsonable_encoder(validation),
            )

        configuration = candidate.yeastar_connection_config
        store = YeastarConfigurationStore(redis, settings.APP_SECRET_KEY or "")
        row = await get_integration_status(db)
        assert row is not None
        candidate_fingerprint = candidate.yeastar_configuration_fingerprint
        configuration_changed = (
            existing.yeastar_configuration_fingerprint != candidate_fingerprint
        )
        requires_retest = row.configuration_fingerprint != candidate_fingerprint

        if requires_retest:
            # The auth owner lock excludes token rotation. A single Redis
            # transaction then replaces configuration, opens the manual gate,
            # and deletes the old token without a partial transition.
            async with RedisOwnerLock(redis, wait_seconds=5, timeout_seconds=30):
                encrypted = store.encrypt(configuration)
                circuit = CircuitBreakerState(
                    opened_at=datetime.now(UTC),
                    reason=ConnectionState.NOT_TESTED,
                ).model_dump_json()
                async with redis.pipeline(transaction=True) as transaction:
                    transaction.mset(
                        {
                            store.state_key: encrypted,
                            YEASTAR_CIRCUIT_KEY: circuit,
                        }
                    )
                    transaction.delete(YEASTAR_TOKEN_STATE_KEY)
                    await transaction.execute()
            clear_detected_metadata(row)
            row.status = YeastarConnectionStatus.NOT_TESTED
        else:
            # Re-encrypt/persist an equivalent override without disconnecting a
            # healthy, fingerprint-matched integration.
            await store.write(configuration)

        row.configured_date_format = str(candidate.YEASTAR_DATE_FORMAT or "")
        row.configuration_fingerprint = candidate_fingerprint
        await audit(
            db,
            action="yeastar.configuration.update",
            request=request,
            user=user,
            resource_type="integration_status",
            resource_id="yeastar",
            details={
                "configuration_changed": configuration_changed,
                "retest_required": requires_retest,
                "status": row.status.value,
            },
        )
        await db.commit()
        return validation
    except YeastarLockTimeoutError as exc:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Phone-system authentication is busy. Try saving again.",
        ) from exc
    except (RedisError, YeastarConfigurationStateError) as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Phone-system configuration could not be saved safely.",
        ) from exc
    finally:
        await action_lock.release()


@router.get(
    "/yeastar/configuration/validate",
    response_model=YeastarConfigurationValidationResponse,
)
async def validate_yeastar_configuration(
    user: CurrentUser,
    settings: EffectiveYeastarSettings,
) -> YeastarConfigurationValidationResponse:
    # This endpoint reads only local encrypted state. It performs no PBX, DNS,
    # token, or outbound HTTP operation.
    return configuration_validation(settings)


@router.get("/yeastar/status", response_model=YeastarStatusResponse)
async def get_yeastar_status(
    user: CurrentUser,
    db: Annotated[AsyncSession, Depends(get_db)],
    settings: EffectiveYeastarSettings,
) -> YeastarStatusResponse:
    circuit_state: CircuitBreakerState | None = None
    try:
        row, configured, _ = await reconcile_configuration_fingerprint(
            db,
            settings,
            clear_token_state=_clear_shared_token_state,
        )
        circuit_state = await YeastarCircuitBreaker(get_redis()).get_state()
    except Exception:
        # A token-cache outage must not turn this local status read into a PBX
        # connection attempt. Fingerprint checks in TokenManager remain fail-safe.
        row = await get_integration_status(db)
        assert row is not None
        configured = configuration_validation(settings).valid
        row.status = (
            YeastarConnectionStatus.NOT_TESTED
            if configured
            else YeastarConnectionStatus.NOT_CONFIGURED
        )
    await db.commit()
    return status_response(
        row,
        configured=configured,
        circuit_state=circuit_state,
    )


@router.post(
    "/yeastar/test",
    response_model=YeastarConnectionTestResponse | YeastarConfigurationValidationResponse,
)
async def test_yeastar_connection(
    request: Request,
    user: CurrentUser,
    db: Annotated[AsyncSession, Depends(get_db)],
    settings: Annotated[Settings, Depends(get_settings)],
) -> YeastarConnectionTestResponse | JSONResponse:
    base_settings = settings
    redis = get_redis()
    settings = await _load_effective_configuration(redis, base_settings)
    validation = configuration_validation(settings)
    if not validation.valid:
        # Return the exact safe validation contract and stop before constructing
        # token/HTTP clients or resolving the PBX host.
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            content=jsonable_encoder(validation),
        )

    lock = RedisOwnerLock(
        redis,
        key=YEASTAR_MANUAL_TEST_LOCK,
        timeout_seconds=_manual_action_lock_timeout(settings),
        wait_seconds=0,
    )
    try:
        await lock.acquire()
    except YeastarLockTimeoutError as exc:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="A phone-system connection test is already running.",
        ) from exc

    client: YeastarClient | None = None
    try:
        # Save/Test/Reset share this lock. Reload after acquisition so a Save
        # that completed during dependency resolution cannot leave stale
        # credentials in this manual test.
        settings = await _load_effective_configuration(redis, base_settings)
        validation = configuration_validation(settings)
        if not validation.valid:
            return JSONResponse(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                content=jsonable_encoder(validation),
            )
        row, _, _ = await reconcile_configuration_fingerprint(
            db,
            settings,
            clear_token_state=_clear_shared_token_state,
        )
        client = YeastarClient(
            settings=settings,
            redis=redis,
            trigger=AuthTrigger.MANUAL_CONNECTION_TEST,
        )
        information, capabilities = await client.inspect_connection()
        if capabilities.state == "supported":
            await client.token_manager.mark_connection_successful()
        else:
            await client.token_manager.open_circuit(
                ConnectionState.UNSUPPORTED_FIRMWARE
            )
        record_connection_success(row, information, capabilities)
        await audit(
            db,
            action="yeastar.connection.test",
            request=request,
            user=user,
            resource_type="integration_status",
            resource_id="yeastar",
            details={"status": row.status.value},
        )
        waiting_items = []
        waiting_jobs = []
        if row.status == YeastarConnectionStatus.CONNECTED:
            waiting_items = list(
                (
                    await db.scalars(
                        select(ProcessingJobItem).where(
                            ProcessingJobItem.status
                            == ItemStatus.WAITING_FOR_CONNECTION
                        )
                    )
                ).all()
            )
        waiting_item_job_ids = {item.job_id for item in waiting_items}
        for item in waiting_items:
            item.status = ItemStatus.QUEUED
            item.stage = "queued"
            item.error_category = None
            item.error_message = None
        if row.status == YeastarConnectionStatus.CONNECTED:
            waiting_jobs = list(
                (
                    await db.scalars(
                        select(ProcessingJob).where(
                            ProcessingJob.status == JobStatus.WAITING_FOR_CONNECTION
                        )
                    )
                ).all()
            )
        discovery_job_ids = []
        for waiting_job in waiting_jobs:
            if waiting_job.id in waiting_item_job_ids:
                waiting_job.status = JobStatus.DOWNLOADING_RECORDINGS
                waiting_job.current_stage = "Preparing recordings"
            else:
                waiting_job.status = JobStatus.QUEUED
                waiting_job.current_stage = "Queued"
                discovery_job_ids.append(waiting_job.id)
            waiting_job.last_error_category = None
            waiting_job.last_error_message = None
        await db.commit()
        try:
            from app.workers.tasks import process_analysis_job, process_job_item

            for waiting_job_id in discovery_job_ids:
                process_analysis_job.delay(str(waiting_job_id))
            for waiting_item in waiting_items:
                process_job_item.delay(str(waiting_item.id))
        except Exception:
            logger.exception("Could not resume locally paused phone-system jobs")
        return YeastarConnectionTestResponse(
            configurationAccepted=True,
            configuration=validation.configuration,
            connection=YeastarConnectionTestResult(
                status=row.status,
                model=information.model_name,
                firmwareVersion=information.firmware_version,
            ),
        )
    except Exception as exc:
        breaker = YeastarCircuitBreaker(redis)
        if await breaker.get_state() is None:
            failure_status = connection_status_for_error(exc)
            await breaker.open(ConnectionState(failure_status.value))
        row = await get_integration_status(db)
        assert row is not None
        record_connection_failure(row, exc)
        await audit(
            db,
            action="yeastar.connection.test",
            request=request,
            user=user,
            resource_type="integration_status",
            resource_id="yeastar",
            outcome="failure",
            details={"category": row.last_error_category},
        )
        await db.commit()
        safe_status = status_response(row, configured=True)
        raise HTTPException(
            status_code=(
                status.HTTP_401_UNAUTHORIZED
                if isinstance(exc, YeastarError) and exc.opens_circuit
                else status.HTTP_503_SERVICE_UNAVAILABLE
            ),
            detail={
                "status": safe_status.status,
                "message": safe_status.message,
                "last_error_reference": safe_status.last_error_reference,
            },
        ) from exc
    finally:
        if client is not None:
            await client.aclose()
        await lock.release()


@router.post("/yeastar/reset", response_model=YeastarStatusResponse)
async def reset_yeastar_connection(
    request: Request,
    user: CurrentUser,
    db: Annotated[AsyncSession, Depends(get_db)],
    settings: Annotated[Settings, Depends(get_settings)],
) -> YeastarStatusResponse:
    base_settings = settings
    redis = get_redis()
    lock = RedisOwnerLock(
        redis,
        key=YEASTAR_MANUAL_TEST_LOCK,
        timeout_seconds=_manual_action_lock_timeout(settings),
        wait_seconds=0,
    )
    try:
        await lock.acquire()
    except YeastarLockTimeoutError as exc:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="A phone-system connection action is already running.",
        ) from exc
    try:
        settings = await _load_effective_configuration(redis, base_settings)
        validation = configuration_validation(settings)
        auth_lock = RedisOwnerLock(
            redis,
            timeout_seconds=_manual_action_lock_timeout(settings),
            wait_seconds=5,
        )
        try:
            await auth_lock.acquire()
        except YeastarLockTimeoutError as exc:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="Phone-system authentication is busy. Try resetting again.",
            ) from exc
        try:
            await YeastarCircuitBreaker(redis).open(
                ConnectionState.NOT_TESTED
                if validation.valid
                else ConnectionState.NOT_CONFIGURED
            )
            if validation.valid:
                manager = YeastarTokenManager(settings, redis)
                try:
                    await manager.revoke_current_token(owner_lock=auth_lock)
                except YeastarLockTimeoutError:
                    raise
                except YeastarError:
                    # Revocation is attempted exactly once. Local invalidation still
                    # completes so Reset never loops against a rejecting PBX.
                    pass
                finally:
                    await manager.aclose()
            else:
                await redis.delete(YEASTAR_TOKEN_STATE_KEY)
            row = await get_integration_status(db)
            assert row is not None
            clear_detected_metadata(row)
            row.configured_date_format = str(settings.YEASTAR_DATE_FORMAT or "")
            row.configuration_fingerprint = (
                settings.yeastar_configuration_fingerprint if validation.valid else None
            )
            row.status = (
                YeastarConnectionStatus.NOT_TESTED
                if validation.valid
                else YeastarConnectionStatus.NOT_CONFIGURED
            )
            await audit(
                db,
                action="yeastar.connection.reset",
                request=request,
                user=user,
                resource_type="integration_status",
                resource_id="yeastar",
            )
            await db.commit()
            return status_response(row, configured=validation.valid)
        finally:
            await auth_lock.release()
    finally:
        await lock.release()
