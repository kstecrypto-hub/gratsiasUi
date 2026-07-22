from __future__ import annotations

from app.services.yeastar.schemas import ConnectionState


class YeastarError(Exception):
    category = "yeastar_error"
    connection_state = ConnectionState.TEMPORARILY_UNAVAILABLE
    retryable = False
    opens_circuit = False

    def __init__(self, message: str, errcode: int | str | None = None) -> None:
        super().__init__(message)
        try:
            self.errcode = int(errcode) if errcode is not None else None
        except (TypeError, ValueError):
            self.errcode = None


class YeastarConfigurationError(YeastarError):
    category = "not_configured"
    connection_state = ConnectionState.NOT_CONFIGURED


class YeastarConfigurationStateError(YeastarConfigurationError):
    """The encrypted, UI-managed phone-system configuration cannot be trusted."""

    category = "configuration_state"


class YeastarAuthenticationError(YeastarError):
    category = "auth_rejected"
    connection_state = ConnectionState.AUTH_REJECTED
    opens_circuit = True


class YeastarAuthenticationRequiredError(YeastarAuthenticationError):
    category = "not_tested"
    connection_state = ConnectionState.NOT_TESTED
    opens_circuit = False


class YeastarTokenExpiredError(YeastarError):
    category = "token_expired"


class YeastarTokenRefreshError(YeastarError):
    category = "token_refresh_failed"
    connection_state = ConnectionState.TOKEN_REFRESH_FAILED
    opens_circuit = True


class YeastarPermissionError(YeastarError):
    category = "permission_denied"
    connection_state = ConnectionState.PERMISSION_DENIED
    opens_circuit = True


class YeastarIpBlockedError(YeastarError):
    category = "ip_blocked"
    connection_state = ConnectionState.IP_BLOCKED
    opens_circuit = True


class YeastarIpForbiddenError(YeastarError):
    category = "ip_not_allowed"
    connection_state = ConnectionState.IP_NOT_ALLOWED
    opens_circuit = True


class YeastarApiDisabledError(YeastarError):
    category = "api_disabled"
    connection_state = ConnectionState.API_DISABLED
    opens_circuit = True


class YeastarUnsupportedVersionError(YeastarError):
    category = "unsupported_api_version"
    connection_state = ConnectionState.UNSUPPORTED_API_VERSION
    opens_circuit = True


class YeastarUnsupportedFirmwareError(YeastarError):
    category = "unsupported_firmware"
    connection_state = ConnectionState.UNSUPPORTED_FIRMWARE
    opens_circuit = True


class YeastarRecordingDisabledError(YeastarError):
    category = "recording_disabled"


class YeastarRecordingDownloadLimitError(YeastarError):
    category = "recording_download_limit"
    retryable = True


class YeastarParameterError(YeastarError):
    category = "invalid_parameter"


class YeastarDataNotFoundError(YeastarError):
    category = "data_not_found"


class YeastarOperationError(YeastarError):
    category = "operation_failed"


class YeastarConnectionError(YeastarError):
    category = "connection"
    connection_state = ConnectionState.NETWORK_UNAVAILABLE
    retryable = True


class YeastarNetworkError(YeastarConnectionError):
    category = "network_unavailable"


class YeastarTemporarilyUnavailableError(YeastarConnectionError):
    category = "temporarily_unavailable"
    connection_state = ConnectionState.TEMPORARILY_UNAVAILABLE


class YeastarAPIError(YeastarError):
    category = "provider_api"


class YeastarResponseError(YeastarAPIError):
    category = "invalid_response"


class YeastarSecurityError(YeastarError):
    category = "security"


class YeastarCircuitOpenError(YeastarError):
    def __init__(self, state: ConnectionState, errcode: int | None = None) -> None:
        super().__init__("Phone-system connection attempts are paused for safety.", errcode)
        self.connection_state = state
        self.category = state.value


class YeastarLockTimeoutError(YeastarError):
    category = "authentication_busy"
    retryable = True


class YeastarTokenStateError(YeastarError):
    category = "token_state"
    connection_state = ConnectionState.TOKEN_REFRESH_FAILED
    opens_circuit = True


__all__ = [name for name in globals() if name.startswith("Yeastar")]
