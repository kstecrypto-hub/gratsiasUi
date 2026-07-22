from __future__ import annotations

import hashlib
import hmac
import json
from datetime import UTC, datetime
from enum import StrEnum
from typing import Literal, Protocol
from urllib.parse import unquote, urlparse

from pydantic import BaseModel, ConfigDict, Field, SecretStr, field_validator


class AuthTrigger(StrEnum):
    MANUAL_CONNECTION_TEST = "manual_connection_test"
    OPERATOR_SYNC = "operator_sync"
    CALL_ANALYSIS = "call_analysis"
    TOKEN_RENEWAL = "token_renewal"


class ConnectionState(StrEnum):
    NOT_CONFIGURED = "not_configured"
    NOT_TESTED = "not_tested"
    CONNECTED = "connected"
    AUTH_REJECTED = "auth_rejected"
    TOKEN_REFRESH_FAILED = "token_refresh_failed"
    IP_BLOCKED = "ip_blocked"
    IP_NOT_ALLOWED = "ip_not_allowed"
    API_DISABLED = "api_disabled"
    PERMISSION_DENIED = "permission_denied"
    UNSUPPORTED_API_VERSION = "unsupported_api_version"
    UNSUPPORTED_FIRMWARE = "unsupported_firmware"
    NETWORK_UNAVAILABLE = "network_unavailable"
    TEMPORARILY_UNAVAILABLE = "temporarily_unavailable"


class YeastarEdition(StrEnum):
    CLOUD = "cloud"
    SOFTWARE = "software"
    APPLIANCE = "appliance"
    UNKNOWN = "unknown"


class YeastarConnectionSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    BaseUrl: str
    ClientId: str
    ClientSecret: SecretStr
    DateFormat: str = "MM/dd/yyyy HH:mm:ss"
    PageSize: int = Field(default=500, ge=1, le=10_000)
    IgnoreSslErrors: bool = True


class YeastarConnectionConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    Name: str = ""
    Settings: YeastarConnectionSettings


class ConfigurationValidationError(BaseModel):
    model_config = ConfigDict(extra="forbid")

    field: str
    message: str


class TokenState(BaseModel):
    model_config = ConfigDict(extra="forbid")

    access_token: SecretStr
    access_token_expires_at: datetime
    refresh_token: SecretStr
    refresh_token_expires_at: datetime
    issued_at: datetime
    generation: int = Field(ge=1)
    configuration_fingerprint: str | None = None

    @field_validator(
        "access_token_expires_at", "refresh_token_expires_at", "issued_at", mode="after"
    )
    @classmethod
    def require_aware_datetime(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("Token timestamps must include a timezone")
        return value.astimezone(UTC)

    @property
    def access_token_value(self) -> str:
        return self.access_token.get_secret_value()

    @property
    def refresh_token_value(self) -> str:
        return self.refresh_token.get_secret_value()


class TokenResponse(BaseModel):
    model_config = ConfigDict(extra="ignore")

    errcode: int = 0
    errmsg: str = "SUCCESS"
    access_token_expire_time: int = Field(gt=0)
    access_token: SecretStr
    refresh_token_expire_time: int = Field(gt=0)
    refresh_token: SecretStr

    @field_validator("access_token", "refresh_token")
    @classmethod
    def require_token(cls, value: SecretStr) -> SecretStr:
        if not value.get_secret_value():
            raise ValueError("Token cannot be empty")
        return value


class CircuitBreakerState(BaseModel):
    model_config = ConfigDict(extra="forbid")

    state: Literal["open"] = "open"
    opened_at: datetime
    reason: ConnectionState
    last_errcode: int | None = None
    manual_reset_required: bool = True

    @field_validator("opened_at", mode="after")
    @classmethod
    def normalize_opened_at(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("Circuit timestamp must include a timezone")
        return value.astimezone(UTC)


class CapabilityProfile(BaseModel):
    model_config = ConfigDict(extra="forbid")

    edition: YeastarEdition
    state: Literal["supported", "unsupported", "unknown"]
    extensions: bool
    cdr_v2: bool | None
    # The endpoint family selected for CDR discovery.  ``cdr_v2`` is retained
    # for the existing public contract; this explicit discriminator lets the
    # worker select the legacy adapter without treating it as V2 support.
    cdr_api_version: Literal["v1", "v2"] | None = None
    recordings: bool
    firmware_version: str
    minimum_cdr_v2_version: str | None = None
    message: str


class SystemInformation(BaseModel):
    """Non-secret fields from ``system/information`` used for capability checks."""

    model_config = ConfigDict(extra="ignore")

    device_name: str | None = None
    model_name: str | None = None
    firmware_version: str | None = None
    system_time: str | None = None
    system_date_format: str | None = None
    system_time_format: str | None = None
    timestamp: int | None = None

    @field_validator("timestamp", mode="before")
    @classmethod
    def normalize_timestamp(cls, value: object) -> int | None:
        if value in (None, ""):
            return None
        try:
            return int(value)  # type: ignore[arg-type]
        except (TypeError, ValueError) as exc:
            raise ValueError("Invalid phone-system timestamp") from exc


class YeastarEnvelope(BaseModel):
    model_config = ConfigDict(extra="allow")

    errcode: int = 0
    errmsg: str = "SUCCESS"

    @field_validator("errcode", mode="before")
    @classmethod
    def normalize_errcode(cls, value: object) -> int:
        try:
            return int(value or 0)
        except (TypeError, ValueError) as exc:
            raise ValueError("Invalid Yeastar error code") from exc


class _SettingsLike(Protocol):
    APP_SECRET_KEY: str | None
    YEASTAR_NAME: str
    YEASTAR_BASE_URL: str | None
    YEASTAR_CLIENT_ID: str | None
    YEASTAR_CLIENT_SECRET: str | None
    YEASTAR_DATE_FORMAT: str
    YEASTAR_PAGE_SIZE: int | str
    YEASTAR_IGNORE_SSL_ERRORS: bool | str
    YEASTAR_ALLOW_HTTP: bool | str
    YEASTAR_USER_AGENT: str
    YEASTAR_AUTH_API_PATH: str
    YEASTAR_SYSTEM_API_PATH: str
    YEASTAR_EXTENSION_API_PATH: str
    YEASTAR_CDR_API_PATH: str
    YEASTAR_RECORDING_API_PATH: str
    YEASTAR_CONNECT_TIMEOUT_SECONDS: float | str
    YEASTAR_READ_TIMEOUT_SECONDS: float | str
    YEASTAR_DOWNLOAD_TIMEOUT_SECONDS: float | str
    YEASTAR_TOKEN_REFRESH_SKEW_SECONDS: int | str
    YEASTAR_TRANSIENT_RETRY_COUNT: int | str
    yeastar_page_size: int
    yeastar_ignore_ssl_errors: bool
    yeastar_allow_http: bool


def connection_config_from_settings(settings: _SettingsLike) -> YeastarConnectionConfig:
    return YeastarConnectionConfig(
        Name=settings.YEASTAR_NAME.strip(),
        Settings=YeastarConnectionSettings(
            BaseUrl=str(settings.YEASTAR_BASE_URL or "").strip().rstrip("/"),
            ClientId=str(settings.YEASTAR_CLIENT_ID or "").strip(),
            ClientSecret=SecretStr(str(settings.YEASTAR_CLIENT_SECRET or "")),
            DateFormat=str(settings.YEASTAR_DATE_FORMAT or "").strip(),
            PageSize=settings.yeastar_page_size,
            IgnoreSslErrors=settings.yeastar_ignore_ssl_errors,
        ),
    )


def sanitized_configuration(config: YeastarConnectionConfig) -> dict[str, object]:
    client_id = config.Settings.ClientId.strip()
    secret = config.Settings.ClientSecret.get_secret_value()
    raw_base_url = config.Settings.BaseUrl
    parsed_base_url = urlparse(raw_base_url)
    valid_port = True
    try:
        _ = parsed_base_url.port
    except ValueError:
        valid_port = False
    if raw_base_url and (
        parsed_base_url.scheme not in {"http", "https"}
        or not parsed_base_url.hostname
        or not valid_port
        or parsed_base_url.username is not None
        or parsed_base_url.password is not None
        or parsed_base_url.path not in {"", "/"}
        or bool(parsed_base_url.query)
        or bool(parsed_base_url.fragment)
    ):
        safe_base_url = "[INVALID]"
    else:
        safe_base_url = raw_base_url
    return {
        "Name": config.Name,
        "Settings": {
            "BaseUrl": safe_base_url,
            "ClientId": "[CONFIGURED]" if client_id else "[NOT CONFIGURED]",
            "ClientSecret": "[REDACTED]" if secret else "[NOT CONFIGURED]",
            "DateFormat": config.Settings.DateFormat,
            "PageSize": config.Settings.PageSize,
            "IgnoreSslErrors": config.Settings.IgnoreSslErrors,
        },
    }


def configuration_fingerprint(config: YeastarConnectionConfig, app_secret_key: str) -> str:
    if not app_secret_key:
        raise ValueError("APP_SECRET_KEY is required for a configuration fingerprint")
    key = hashlib.sha256(
        b"yeastar-configuration-fingerprint:v1\x00" + app_secret_key.encode("utf-8")
    ).digest()
    secret_digest = hmac.new(
        key,
        config.Settings.ClientSecret.get_secret_value().encode("utf-8"),
        hashlib.sha256,
    ).hexdigest()
    canonical = json.dumps(
        {
            "Name": config.Name,
            "BaseUrl": config.Settings.BaseUrl.rstrip("/"),
            "ClientId": config.Settings.ClientId,
            "DateFormat": config.Settings.DateFormat,
            "PageSize": config.Settings.PageSize,
            "IgnoreSslErrors": config.Settings.IgnoreSslErrors,
            "ClientSecretDigest": secret_digest,
        },
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hmac.new(key, canonical, hashlib.sha256).hexdigest()


def _integer_error(
    value: int | str,
    *,
    field: str,
    minimum: int,
    maximum: int,
    message: str,
) -> ConfigurationValidationError | None:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return ConfigurationValidationError(field=field, message=message)
    if not minimum <= parsed <= maximum:
        return ConfigurationValidationError(field=field, message=message)
    return None


def _float_error(
    value: float | str,
    *,
    field: str,
    minimum: float,
    maximum: float,
) -> ConfigurationValidationError | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        parsed = -1
    if not minimum <= parsed <= maximum:
        return ConfigurationValidationError(
            field=field,
            message="Enter a valid phone-system timeout value.",
        )
    return None


def validate_local_configuration(settings: _SettingsLike) -> list[ConfigurationValidationError]:
    """Validate Yeastar settings without constructing a network client."""
    from app.services.yeastar.datetime_formatter import YeastarDateTimeFormatter

    errors: list[ConfigurationValidationError] = []
    base_url = str(settings.YEASTAR_BASE_URL or "").strip()
    if not base_url:
        errors.append(
            ConfigurationValidationError(
                field="Settings.BaseUrl", message="A valid phone-system URL is required."
            )
        )
    else:
        parsed = urlparse(base_url)
        valid_port = True
        try:
            _ = parsed.port
        except ValueError:
            valid_port = False
        if (
            parsed.scheme not in {"http", "https"}
            or not parsed.hostname
            or not valid_port
            or parsed.username is not None
            or parsed.password is not None
            or parsed.path not in {"", "/"}
            or bool(parsed.params)
            or bool(parsed.query)
            or bool(parsed.fragment)
        ):
            errors.append(
                ConfigurationValidationError(
                    field="Settings.BaseUrl",
                    message="Use only the phone-system scheme, host and optional port.",
                )
            )
        elif parsed.scheme == "http" and settings.YEASTAR_ALLOW_HTTP is not True:
            errors.append(
                ConfigurationValidationError(
                    field="Settings.BaseUrl",
                    message="HTTP requires explicit approval for a trusted internal network.",
                )
            )

    if not str(settings.YEASTAR_CLIENT_ID or "").strip():
        errors.append(
            ConfigurationValidationError(
                field="Settings.ClientId", message="A phone-system Client ID is required."
            )
        )
    if not str(settings.YEASTAR_CLIENT_SECRET or "").strip():
        errors.append(
            ConfigurationValidationError(
                field="Settings.ClientSecret", message="A phone-system Client Secret is required."
            )
        )
    if not settings.APP_SECRET_KEY:
        errors.append(
            ConfigurationValidationError(
                field="ApplicationSecurity",
                message="Application security must be configured before connecting.",
            )
        )

    user_agent = str(settings.YEASTAR_USER_AGENT or "")
    if not user_agent.strip() or "\n" in user_agent or "\r" in user_agent or len(user_agent) > 255:
        errors.append(
            ConfigurationValidationError(
                field="UserAgent", message="A valid phone-system User-Agent is required."
            )
        )

    api_paths = {
        "AuthApiPath": settings.YEASTAR_AUTH_API_PATH,
        "SystemApiPath": settings.YEASTAR_SYSTEM_API_PATH,
        "ExtensionApiPath": settings.YEASTAR_EXTENSION_API_PATH,
        "CdrApiPath": settings.YEASTAR_CDR_API_PATH,
        "RecordingApiPath": settings.YEASTAR_RECORDING_API_PATH,
    }
    for field, raw_path in api_paths.items():
        path = str(raw_path or "")
        decoded_path = path
        for _ in range(3):
            next_value = unquote(decoded_path)
            if next_value == decoded_path:
                break
            decoded_path = next_value
        parsed_path = urlparse(decoded_path)
        if (
            not decoded_path.startswith("/")
            or decoded_path.startswith("//")
            or ".." in decoded_path.replace("\\", "/").split("/")
            or "\\" in decoded_path
            or parsed_path.scheme
            or parsed_path.netloc
            or parsed_path.query
            or parsed_path.fragment
        ):
            errors.append(
                ConfigurationValidationError(
                    field=field, message="Use a valid relative phone-system API path."
                )
            )

    try:
        YeastarDateTimeFormatter.dotnet_to_strftime(str(settings.YEASTAR_DATE_FORMAT or ""))
    except ValueError:
        errors.append(
            ConfigurationValidationError(
                field="Settings.DateFormat",
                message="The configured phone-system date format is not supported.",
            )
        )

    page_error = _integer_error(
        settings.YEASTAR_PAGE_SIZE,
        field="Settings.PageSize",
        minimum=1,
        maximum=10_000,
        message="Page size must be between 1 and 10000.",
    )
    if page_error:
        errors.append(page_error)

    if not isinstance(settings.YEASTAR_IGNORE_SSL_ERRORS, bool):
        errors.append(
            ConfigurationValidationError(
                field="Settings.IgnoreSslErrors", message="Choose a valid certificate setting."
            )
        )
    if not isinstance(settings.YEASTAR_ALLOW_HTTP, bool):
        errors.append(
            ConfigurationValidationError(
                field="AllowHttp", message="Choose a valid HTTP policy setting."
            )
        )

    for value, field, minimum, maximum in (
        (settings.YEASTAR_CONNECT_TIMEOUT_SECONDS, "ConnectTimeout", 0.1, 300.0),
        (settings.YEASTAR_READ_TIMEOUT_SECONDS, "ReadTimeout", 0.1, 900.0),
        (settings.YEASTAR_DOWNLOAD_TIMEOUT_SECONDS, "DownloadTimeout", 1.0, 3600.0),
    ):
        error = _float_error(value, field=field, minimum=minimum, maximum=maximum)
        if error:
            errors.append(error)

    skew_error = _integer_error(
        settings.YEASTAR_TOKEN_REFRESH_SKEW_SECONDS,
        field="TokenRefreshSkew",
        minimum=0,
        maximum=1800,
        message="Token refresh skew must be between 0 and 1800 seconds.",
    )
    if skew_error:
        errors.append(skew_error)
    retry_error = _integer_error(
        settings.YEASTAR_TRANSIENT_RETRY_COUNT,
        field="TransientRetryCount",
        minimum=0,
        maximum=1,
        message="Transient retry count must be zero or one.",
    )
    if retry_error:
        errors.append(retry_error)
    return errors


__all__ = [
    "AuthTrigger",
    "CapabilityProfile",
    "CircuitBreakerState",
    "ConfigurationValidationError",
    "ConnectionState",
    "SystemInformation",
    "TokenResponse",
    "TokenState",
    "YeastarConnectionConfig",
    "YeastarConnectionSettings",
    "YeastarEdition",
    "YeastarEnvelope",
    "configuration_fingerprint",
    "connection_config_from_settings",
    "sanitized_configuration",
    "validate_local_configuration",
]
