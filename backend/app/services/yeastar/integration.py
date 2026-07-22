from __future__ import annotations

from collections.abc import Awaitable, Callable, Mapping
from datetime import UTC, datetime
from typing import Literal

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from redis.asyncio import Redis

from app.core.config import Settings
from app.models import IntegrationStatus
from app.models.enums import YeastarConnectionStatus
from app.schemas.configuration import (
    SanitizedYeastarConnectionConfig,
    YeastarCapabilitiesResponse,
    YeastarConfigurationValidationError,
    YeastarConfigurationValidationResponse,
    YeastarStatusResponse,
)
from app.services.yeastar.errors import YeastarError
from app.services.yeastar.schemas import (
    CapabilityProfile,
    CircuitBreakerState,
    ConnectionState,
    SystemInformation,
    sanitized_configuration,
    validate_local_configuration,
)
from app.services.yeastar.circuit_breaker import YeastarCircuitBreaker
from app.services.yeastar.token_store import (
    RedisOwnerLock,
    YEASTAR_TOKEN_STATE_KEY,
)


YEASTAR_PROVIDER = "yeastar"


def runtime_cdr_api_version(
    capabilities_json: Mapping[str, object] | None,
) -> Literal["v1", "v2"]:
    """Select the persisted CDR mode, preserving the historical v2 default.

    A v1 mode is selected only when a successful connection test explicitly
    recorded it.  Older integration rows did not have this field, and must
    continue using the existing v2 path rather than silently switching modes.
    """

    if capabilities_json and capabilities_json.get("cdr_api_version") == "v1":
        return "v1"
    return "v2"


def reported_cdr_api_version(
    capabilities_json: Mapping[str, object] | None,
) -> Literal["v1", "v2"] | None:
    """Return only a verified CDR mode for the read-only status response."""

    if not capabilities_json:
        return None
    value = capabilities_json.get("cdr_api_version")
    if value == "v1":
        return "v1"
    if value == "v2":
        return "v2"
    return None


def safe_configuration(settings: Settings) -> SanitizedYeastarConnectionConfig:
    return SanitizedYeastarConnectionConfig.model_validate(
        sanitized_configuration(settings.yeastar_connection_config)
    )


def configuration_validation(
    settings: Settings,
) -> YeastarConfigurationValidationResponse:
    errors = validate_local_configuration(settings)
    return YeastarConfigurationValidationResponse(
        valid=not errors,
        errors=[
            YeastarConfigurationValidationError(field=item.field, message=item.message)
            for item in errors
        ],
        configuration=safe_configuration(settings),
    )


def is_connected_configuration(
    row: IntegrationStatus | None,
    settings: Settings,
    *,
    circuit_state: CircuitBreakerState | None = None,
) -> bool:
    """Return whether the live status belongs to this exact tested configuration."""

    fingerprint = settings.yeastar_configuration_fingerprint
    fingerprint_matches = bool(
        fingerprint and row is not None and row.configuration_fingerprint == fingerprint
    )
    return local_connection_status(
        row,
        configured=settings.yeastar_configured,
        fingerprint_matches=fingerprint_matches,
        circuit_state=circuit_state,
    ) == YeastarConnectionStatus.CONNECTED


async def get_integration_status(
    session: AsyncSession, *, create: bool = True
) -> IntegrationStatus | None:
    row = await session.scalar(
        select(IntegrationStatus).where(IntegrationStatus.provider == YEASTAR_PROVIDER)
    )
    if row is None and create:
        row = IntegrationStatus(
            provider=YEASTAR_PROVIDER,
            status=YeastarConnectionStatus.NOT_TESTED,
            capabilities_json={},
        )
        session.add(row)
        await session.flush()
    return row


def clear_detected_metadata(row: IntegrationStatus) -> None:
    row.device_name = None
    row.model_name = None
    row.firmware_version = None
    row.system_time = None
    row.system_date_format = None
    row.system_time_format = None
    row.provider_timestamp = None
    row.capabilities_json = {}
    row.last_tested_at = None
    row.last_successful_connection_at = None
    row.last_error_category = None
    row.last_error_reference = None


async def reconcile_configuration_fingerprint(
    session: AsyncSession,
    settings: Settings,
    *,
    clear_token_state: Callable[[], Awaitable[None]],
) -> tuple[IntegrationStatus, bool, bool]:
    """Reconcile local state without authenticating or constructing HTTP transport."""
    validation = configuration_validation(settings)
    row = await get_integration_status(session)
    assert row is not None
    row.configured_date_format = str(settings.YEASTAR_DATE_FORMAT or "")
    if not validation.valid:
        changed_from_existing = row.configuration_fingerprint is not None
        if changed_from_existing:
            await clear_token_state()
            clear_detected_metadata(row)
        row.configuration_fingerprint = None
        row.status = YeastarConnectionStatus.NOT_CONFIGURED
        return row, False, changed_from_existing

    fingerprint = settings.yeastar_configuration_fingerprint
    if fingerprint is None:
        # APP_SECRET_KEY is part of local validation, so this is defensive.
        row.status = YeastarConnectionStatus.NOT_CONFIGURED
        return row, False, False
    if row.configuration_fingerprint != fingerprint:
        changed_from_existing = row.configuration_fingerprint is not None
        if changed_from_existing:
            await clear_token_state()
        clear_detected_metadata(row)
        row.configuration_fingerprint = fingerprint
        row.status = YeastarConnectionStatus.NOT_TESTED
        return row, True, changed_from_existing
    return row, True, False


async def clear_shared_token_and_require_test(redis: Redis) -> None:
    """Atomically invalidate local auth state after a known configuration rotation."""
    await YeastarCircuitBreaker(redis).open(ConnectionState.NOT_TESTED)
    async with RedisOwnerLock(redis, wait_seconds=5, timeout_seconds=30):
        await redis.delete(YEASTAR_TOKEN_STATE_KEY)


def connection_status_for_error(error: BaseException) -> YeastarConnectionStatus:
    state = getattr(error, "connection_state", ConnectionState.TEMPORARILY_UNAVAILABLE)
    try:
        value = state.value if isinstance(state, ConnectionState) else str(state)
        return YeastarConnectionStatus(value)
    except ValueError:
        return YeastarConnectionStatus.TEMPORARILY_UNAVAILABLE


def local_connection_status(
    row: IntegrationStatus | None,
    *,
    configured: bool,
    fingerprint_matches: bool,
    circuit_state: CircuitBreakerState | None = None,
) -> YeastarConnectionStatus:
    """Return the fail-safe local status without contacting the phone system."""
    if not configured:
        return YeastarConnectionStatus.NOT_CONFIGURED
    if not fingerprint_matches or row is None:
        return YeastarConnectionStatus.NOT_TESTED
    if circuit_state is not None:
        status = YeastarConnectionStatus(circuit_state.reason.value)
        # An open breaker can never represent a usable connection. Normalize
        # impossible/stale values to the manual test gate.
        if status in {
            YeastarConnectionStatus.CONNECTED,
            YeastarConnectionStatus.NOT_CONFIGURED,
        }:
            return YeastarConnectionStatus.NOT_TESTED
        return status
    return row.status


def record_connection_failure(
    row: IntegrationStatus,
    error: BaseException,
    *,
    tested_at: datetime | None = None,
) -> None:
    row.status = connection_status_for_error(error)
    row.last_tested_at = tested_at or datetime.now(UTC)
    row.last_error_category = str(getattr(error, "category", "unexpected"))[:128]
    errcode = getattr(error, "errcode", None)
    row.last_error_reference = (
        f"YS-{int(errcode)}" if isinstance(errcode, int) else row.last_error_category
    )[:128]


def record_connection_success(
    row: IntegrationStatus,
    information: SystemInformation,
    capabilities: CapabilityProfile,
    *,
    tested_at: datetime | None = None,
) -> None:
    now = tested_at or datetime.now(UTC)
    row.status = (
        YeastarConnectionStatus.CONNECTED
        if capabilities.state == "supported"
        else YeastarConnectionStatus.UNSUPPORTED_FIRMWARE
    )
    row.device_name = information.device_name
    row.model_name = information.model_name
    row.firmware_version = information.firmware_version
    row.system_time = information.system_time
    row.system_date_format = information.system_date_format
    row.system_time_format = information.system_time_format
    try:
        row.provider_timestamp = (
            int(information.timestamp) if information.timestamp is not None else None
        )
    except (TypeError, ValueError):
        row.provider_timestamp = None
    row.capabilities_json = capabilities.model_dump(mode="json")
    row.last_tested_at = now
    row.last_successful_connection_at = now
    row.last_error_category = None
    row.last_error_reference = None


_STATUS_MESSAGES = {
    YeastarConnectionStatus.NOT_CONFIGURED: "Phone system not configured",
    YeastarConnectionStatus.NOT_TESTED: "Connection has not been tested",
    YeastarConnectionStatus.CONNECTED: "Phone system connected",
    YeastarConnectionStatus.AUTH_REJECTED: "The connection details were rejected",
    YeastarConnectionStatus.TOKEN_REFRESH_FAILED: (
        "Connection attempts have been paused for safety"
    ),
    YeastarConnectionStatus.IP_NOT_ALLOWED: (
        "This server is not permitted to access the phone system"
    ),
    YeastarConnectionStatus.IP_BLOCKED: "The phone system has blocked this server",
    YeastarConnectionStatus.API_DISABLED: "The phone-system API is not enabled",
    YeastarConnectionStatus.PERMISSION_DENIED: (
        "The phone-system account does not have permission"
    ),
    YeastarConnectionStatus.UNSUPPORTED_API_VERSION: (
        "The installed phone-system version is not supported"
    ),
    YeastarConnectionStatus.UNSUPPORTED_FIRMWARE: (
        "The installed phone-system version is not supported"
    ),
    YeastarConnectionStatus.NETWORK_UNAVAILABLE: "Could not reach the phone system",
    YeastarConnectionStatus.TEMPORARILY_UNAVAILABLE: (
        "The phone system is temporarily unavailable"
    ),
}


def status_response(
    row: IntegrationStatus | None,
    *,
    configured: bool,
    fingerprint_matches: bool = True,
    circuit_state: CircuitBreakerState | None = None,
) -> YeastarStatusResponse:
    status = local_connection_status(
        row,
        configured=configured,
        fingerprint_matches=fingerprint_matches,
        circuit_state=circuit_state,
    )
    raw_capabilities = row.capabilities_json if row is not None else {}

    def optional_boolean(key: str) -> bool | None:
        value = raw_capabilities.get(key)
        return value if isinstance(value, bool) else None

    message = _STATUS_MESSAGES[status]
    if status == YeastarConnectionStatus.UNSUPPORTED_FIRMWARE:
        capability_message = raw_capabilities.get("message")
        if isinstance(capability_message, str) and capability_message:
            message = capability_message
    last_error_reference = row.last_error_reference if row is not None else None
    if circuit_state is not None and configured and fingerprint_matches:
        last_error_reference = (
            f"YS-{circuit_state.last_errcode}"
            if circuit_state.last_errcode is not None
            else (
                status.value
                if status
                not in {
                    YeastarConnectionStatus.NOT_CONFIGURED,
                    YeastarConnectionStatus.NOT_TESTED,
                }
                else None
            )
        )
    return YeastarStatusResponse(
        status=status,
        configured=configured,
        last_tested_at=row.last_tested_at if row is not None else None,
        last_successful_connection_at=(
            row.last_successful_connection_at if row is not None else None
        ),
        model_name=row.model_name if row is not None else None,
        firmware_version=row.firmware_version if row is not None else None,
        capabilities=YeastarCapabilitiesResponse(
            extensions=optional_boolean("extensions"),
            cdr_v2=optional_boolean("cdr_v2"),
            cdr_api_version=reported_cdr_api_version(raw_capabilities),
            recordings=optional_boolean("recordings"),
        ),
        message=message,
        last_error_reference=last_error_reference,
    )


def is_authentication_failure(error: BaseException) -> bool:
    return isinstance(error, YeastarError) and bool(error.opens_circuit)


__all__ = [
    "YEASTAR_PROVIDER",
    "clear_detected_metadata",
    "clear_shared_token_and_require_test",
    "configuration_validation",
    "connection_status_for_error",
    "get_integration_status",
    "local_connection_status",
    "is_connected_configuration",
    "record_connection_failure",
    "record_connection_success",
    "reconcile_configuration_fingerprint",
    "reported_cdr_api_version",
    "runtime_cdr_api_version",
    "safe_configuration",
    "status_response",
]
