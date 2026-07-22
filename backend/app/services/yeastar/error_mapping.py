from __future__ import annotations

from app.services.yeastar.errors import (
    YeastarAPIError,
    YeastarApiDisabledError,
    YeastarAuthenticationError,
    YeastarDataNotFoundError,
    YeastarError,
    YeastarIpBlockedError,
    YeastarIpForbiddenError,
    YeastarOperationError,
    YeastarParameterError,
    YeastarPermissionError,
    YeastarRecordingDisabledError,
    YeastarRecordingDownloadLimitError,
    YeastarTemporarilyUnavailableError,
    YeastarTokenExpiredError,
    YeastarUnsupportedVersionError,
)


CIRCUIT_OPEN_CODES = frozenset({10002, 10003, 10005, 70004, 70087, 70123, 70656, 80010})


def normalize_errcode(value: int | str | None) -> int | None:
    try:
        return int(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def map_yeastar_error(
    errcode: int | str | None,
    errmsg: str | None = None,
) -> YeastarError:
    """Map a provider code to a safe typed error without reflecting ``errmsg``."""
    code = normalize_errcode(errcode)
    mapping: dict[int, tuple[type[YeastarError], str]] = {
        10001: (
            YeastarUnsupportedVersionError,
            "The requested phone-system function is not available.",
        ),
        10002: (
            YeastarUnsupportedVersionError,
            "The configured phone-system API version is no longer supported.",
        ),
        10003: (
            YeastarUnsupportedVersionError,
            "The installed phone-system version does not support the required call API.",
        ),
        10004: (YeastarTokenExpiredError, "The phone-system session has expired."),
        10005: (
            YeastarAuthenticationError,
            "The phone-system connection details were rejected. Connection attempts have been paused for safety.",
        ),
        10009: (YeastarOperationError, "The phone system could not complete the request."),
        40001: (YeastarParameterError, "The phone system rejected a request parameter."),
        40002: (YeastarParameterError, "The phone system rejected a request parameter."),
        60001: (YeastarDataNotFoundError, "The requested phone-system data was not found."),
        70004: (
            YeastarIpBlockedError,
            "The phone system has blocked this server. Ask your IT administrator to remove the block.",
        ),
        70087: (
            YeastarIpForbiddenError,
            "This server is not allowed to access the phone system. Ask your IT administrator to check the API IP allowlist.",
        ),
        70123: (YeastarApiDisabledError, "The phone-system API is unavailable."),
        70131: (
            YeastarRecordingDisabledError,
            "Call recording is not enabled on the phone system.",
        ),
        70651: (
            YeastarRecordingDownloadLimitError,
            "The phone system is busy preparing another recording download.",
        ),
        70656: (
            YeastarApiDisabledError,
            "The required phone-system API function is not enabled.",
        ),
        80010: (
            YeastarPermissionError,
            "The phone-system account does not have permission for this operation.",
        ),
    }
    if code in mapping:
        error_type, message = mapping[code]
        return error_type(message, code)
    return YeastarAPIError("The phone system rejected the request.", code)


def map_http_status(status_code: int) -> YeastarError:
    if status_code == 401:
        return YeastarAuthenticationError(
            "The phone-system connection details were rejected.", status_code
        )
    if status_code == 403:
        return YeastarPermissionError(
            "The phone-system account does not have permission for this operation.",
            status_code,
        )
    if status_code in {429, 502, 503, 504}:
        return YeastarTemporarilyUnavailableError(
            "The phone system is temporarily unavailable.", status_code
        )
    return YeastarAPIError("The phone system returned an unexpected HTTP response.", status_code)


def opens_authentication_circuit(error: BaseException) -> bool:
    return isinstance(error, YeastarError) and bool(error.opens_circuit)


__all__ = [
    "CIRCUIT_OPEN_CODES",
    "map_http_status",
    "map_yeastar_error",
    "normalize_errcode",
    "opens_authentication_circuit",
]
