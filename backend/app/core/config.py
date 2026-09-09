from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Literal
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from pydantic import BeforeValidator, EmailStr, Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

if TYPE_CHECKING:
    from app.services.yeastar.schemas import YeastarConnectionConfig


def _empty_to_none(value: object) -> object:
    if isinstance(value, str) and not value.strip():
        return None
    return value


OptionalSecret = Annotated[str | None, BeforeValidator(_empty_to_none)]
OptionalEmail = Annotated[EmailStr | None, BeforeValidator(_empty_to_none)]


def _int_or_text(value: object) -> object:
    if isinstance(value, bool):
        return str(value)
    try:
        return int(str(value).strip())
    except (TypeError, ValueError):
        return value


def _float_or_text(value: object) -> object:
    if isinstance(value, bool):
        return str(value)
    try:
        return float(str(value).strip())
    except (TypeError, ValueError):
        return value


def _bool_or_text(value: object) -> object:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "on"}:
            return True
        if normalized in {"0", "false", "no", "off"}:
            return False
    return value


LooseInt = Annotated[int | str, BeforeValidator(_int_or_text)]
LooseFloat = Annotated[float | str, BeforeValidator(_float_or_text)]
LooseBool = Annotated[bool | str, BeforeValidator(_bool_or_text)]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="ignore",
    )

    APP_ENV: Literal["development", "test", "production"] = "production"
    APP_SECRET_KEY: OptionalSecret = None
    APP_TIMEZONE: str = "Europe/Athens"
    FRONTEND_ORIGIN: str = "http://localhost:3000"
    APP_ORIGINS: str | None = None
    TRUSTED_HOSTS: str = "localhost,127.0.0.1,backend"
    SECURE_COOKIES: bool | None = None
    STORAGE_ROOT: Path = Path("/app/storage")
    MAX_REQUEST_BYTES: int = Field(default=2 * 1024 * 1024, ge=1024, le=20 * 1024 * 1024)
    MAX_RECORDING_BYTES: int = Field(default=500 * 1024 * 1024, ge=1024 * 1024)
    MAX_UPLOAD_BYTES: int = Field(
        default=24 * 1024 * 1024, ge=1024 * 1024, le=24 * 1024 * 1024
    )
    MAX_TRANSCRIPTION_UPLOAD_BYTES: int | None = Field(
        default=None, ge=1024 * 1024, le=24 * 1024 * 1024
    )

    ADMIN_EMAIL: OptionalEmail = None
    ADMIN_PASSWORD: OptionalSecret = None

    # Yeastar values remain permissive at process startup. Strict local
    # validation is exposed separately so a bad PBX value can be corrected in
    # the UI without preventing FastAPI from starting.
    YEASTAR_NAME: str = ""
    YEASTAR_BASE_URL: OptionalSecret = None
    YEASTAR_CLIENT_ID: OptionalSecret = None
    YEASTAR_CLIENT_SECRET: OptionalSecret = None
    YEASTAR_DATE_FORMAT: str = "MM/dd/yyyy HH:mm:ss"
    YEASTAR_PAGE_SIZE: LooseInt = 500
    YEASTAR_IGNORE_SSL_ERRORS: LooseBool = True
    YEASTAR_ALLOW_HTTP: LooseBool = False
    YEASTAR_USER_AGENT: str = "YeastarCallAnalyzer/1.0"

    YEASTAR_AUTH_API_PATH: str = "/openapi/v1.0"
    YEASTAR_SYSTEM_API_PATH: str = "/openapi/v1.0"
    YEASTAR_EXTENSION_API_PATH: str = "/openapi/v1.0"
    YEASTAR_CDR_API_PATH: str = "/openapi/v2.0"
    YEASTAR_RECORDING_API_PATH: str = "/openapi/v1.0"

    YEASTAR_CONNECT_TIMEOUT_SECONDS: LooseFloat = 10.0
    YEASTAR_READ_TIMEOUT_SECONDS: LooseFloat = 60.0
    YEASTAR_DOWNLOAD_TIMEOUT_SECONDS: LooseFloat = 300.0
    YEASTAR_TOKEN_REFRESH_SKEW_SECONDS: LooseInt = 120
    YEASTAR_TRANSIENT_RETRY_COUNT: LooseInt = 1
    YEASTAR_TOKEN_LOCK_TIMEOUT_SECONDS: LooseFloat = 180.0
    YEASTAR_TOKEN_LOCK_WAIT_SECONDS: LooseFloat = 15.0

    # Retained only as an internal routing compatibility setting. TLS and date
    # behavior are controlled exclusively by the new non-ambiguous settings.
    YEASTAR_API_VERSION: str = "v2"

    OPENAI_API_KEY: OptionalSecret = None
    OPENAI_TRANSCRIPTION_MODEL: str = "gpt-4o-transcribe"
    OPENAI_DIARIZATION_MODEL: str = "gpt-4o-transcribe-diarize"
    TRANSCRIPTION_LANGUAGE: str = "el"
    TRANSCRIPTION_PIPELINE_DEFAULT: Literal["legacy-v1", "pipeline-v2"] = "legacy-v1"
    TRANSCRIPTION_PIPELINE_V2_ENABLED: bool = False

    DATABASE_URL: str = "postgresql+psycopg://app:app@postgres:5432/yeastar"
    REDIS_URL: str = "redis://redis:6379/0"

    DELETE_AUDIO_AFTER_TRANSCRIPTION: bool = True
    TRANSCRIPT_RETENTION_DAYS: int = Field(default=90, ge=1, le=3650)
    MAX_PARALLEL_TRANSCRIPTIONS: int = Field(default=3, ge=1, le=16)
    SESSION_TTL_SECONDS: int = Field(default=8 * 60 * 60, ge=300, le=7 * 24 * 60 * 60)
    LOGIN_RATE_LIMIT: str = "5/15minutes"
    LOGIN_RATE_WINDOW_SECONDS: int = Field(default=15 * 60, ge=60, le=24 * 60 * 60)

    @field_validator("APP_TIMEZONE")
    @classmethod
    def valid_timezone(cls, value: str) -> str:
        try:
            ZoneInfo(value)
        except ZoneInfoNotFoundError as exc:
            raise ValueError("APP_TIMEZONE must be an IANA timezone") from exc
        return value

    @field_validator("YEASTAR_API_VERSION")
    @classmethod
    def normalize_api_version(cls, value: str) -> str:
        normalized = value.lower().strip().removeprefix("v").removesuffix(".0")
        if normalized not in {"1", "2"}:
            raise ValueError("YEASTAR_API_VERSION must be v1 or v2")
        return f"v{normalized}"

    @model_validator(mode="after")
    def production_secrets(self) -> "Settings":
        if self.APP_ENV == "production" and self.APP_SECRET_KEY is not None:
            if len(self.APP_SECRET_KEY) < 32:
                raise ValueError("APP_SECRET_KEY must contain at least 32 characters")
        return self

    @property
    def yeastar_configured(self) -> bool:
        if not all((self.YEASTAR_BASE_URL, self.YEASTAR_CLIENT_ID, self.YEASTAR_CLIENT_SECRET)):
            return False
        return not self.yeastar_configuration_errors

    @property
    def yeastar_configuration_errors(self) -> list[dict[str, str]]:
        from app.services.yeastar.schemas import validate_local_configuration

        return [item.model_dump() for item in validate_local_configuration(self)]

    @property
    def yeastar_connection_config(self) -> "YeastarConnectionConfig":
        from app.services.yeastar.schemas import connection_config_from_settings

        return connection_config_from_settings(self)

    @property
    def yeastar_configuration_fingerprint(self) -> str | None:
        if not self.APP_SECRET_KEY:
            return None
        from app.services.yeastar.schemas import configuration_fingerprint

        return configuration_fingerprint(self.yeastar_connection_config, self.APP_SECRET_KEY)

    @staticmethod
    def _bounded_int(value: int | str, *, default: int, minimum: int, maximum: int) -> int:
        try:
            parsed = int(value)
        except (TypeError, ValueError):
            return default
        return parsed if minimum <= parsed <= maximum else default

    @staticmethod
    def _bounded_float(
        value: float | str, *, default: float, minimum: float, maximum: float
    ) -> float:
        try:
            parsed = float(value)
        except (TypeError, ValueError):
            return default
        return parsed if minimum <= parsed <= maximum else default

    @staticmethod
    def _safe_bool(value: bool | str, *, default: bool) -> bool:
        return value if isinstance(value, bool) else default

    @property
    def yeastar_page_size(self) -> int:
        return self._bounded_int(self.YEASTAR_PAGE_SIZE, default=500, minimum=1, maximum=10_000)

    @property
    def yeastar_ignore_ssl_errors(self) -> bool:
        return self._safe_bool(self.YEASTAR_IGNORE_SSL_ERRORS, default=True)

    @property
    def yeastar_allow_http(self) -> bool:
        return self._safe_bool(self.YEASTAR_ALLOW_HTTP, default=False)

    @property
    def yeastar_verify_ssl(self) -> bool:
        return not self.yeastar_ignore_ssl_errors

    @property
    def yeastar_connect_timeout_seconds(self) -> float:
        return self._bounded_float(
            self.YEASTAR_CONNECT_TIMEOUT_SECONDS, default=10.0, minimum=0.1, maximum=300.0
        )

    @property
    def yeastar_read_timeout_seconds(self) -> float:
        return self._bounded_float(
            self.YEASTAR_READ_TIMEOUT_SECONDS, default=60.0, minimum=0.1, maximum=900.0
        )

    @property
    def yeastar_download_timeout_seconds(self) -> float:
        return self._bounded_float(
            self.YEASTAR_DOWNLOAD_TIMEOUT_SECONDS,
            default=300.0,
            minimum=1.0,
            maximum=3600.0,
        )

    @property
    def yeastar_token_refresh_skew_seconds(self) -> int:
        return self._bounded_int(
            self.YEASTAR_TOKEN_REFRESH_SKEW_SECONDS,
            default=120,
            minimum=0,
            maximum=1800,
        )

    @property
    def yeastar_transient_retry_count(self) -> int:
        return self._bounded_int(
            self.YEASTAR_TRANSIENT_RETRY_COUNT, default=1, minimum=0, maximum=1
        )

    @property
    def yeastar_token_lock_timeout_seconds(self) -> float:
        configured = self._bounded_float(
            self.YEASTAR_TOKEN_LOCK_TIMEOUT_SECONDS,
            default=180.0,
            minimum=1.0,
            maximum=900.0,
        )
        longest_request = (
            self.yeastar_connect_timeout_seconds + self.yeastar_read_timeout_seconds
        ) * (self.yeastar_transient_retry_count + 1)
        return max(configured, longest_request + 15.0)

    @property
    def yeastar_token_lock_wait_seconds(self) -> float:
        return self._bounded_float(
            self.YEASTAR_TOKEN_LOCK_WAIT_SECONDS,
            default=15.0,
            minimum=0.1,
            maximum=120.0,
        )

    # Compatibility aliases consumed by the existing all-in-one client while
    # higher-level Yeastar services migrate to the dedicated core components.
    @property
    def YEASTAR_VERIFY_SSL(self) -> bool:  # noqa: N802
        return self.yeastar_verify_ssl

    @property
    def YEASTAR_DATETIME_FORMAT(self) -> str:  # noqa: N802
        from app.services.yeastar.datetime_formatter import YeastarDateTimeFormatter

        try:
            return YeastarDateTimeFormatter.dotnet_to_strftime(self.YEASTAR_DATE_FORMAT)
        except ValueError:
            return "%m/%d/%Y %H:%M:%S"

    @property
    def YEASTAR_CONNECT_TIMEOUT(self) -> float:  # noqa: N802
        return self.yeastar_connect_timeout_seconds

    @property
    def YEASTAR_READ_TIMEOUT(self) -> float:  # noqa: N802
        return self.yeastar_read_timeout_seconds

    @property
    def openai_configured(self) -> bool:
        return bool(self.OPENAI_API_KEY)

    @property
    def admin_configured(self) -> bool:
        return bool(self.ADMIN_EMAIL and self.ADMIN_PASSWORD)

    @property
    def cookie_secure(self) -> bool:
        if self.SECURE_COOKIES is not None:
            return self.SECURE_COOKIES
        return self.APP_ENV == "production"

    @property
    def origins(self) -> list[str]:
        configured = self.APP_ORIGINS or self.FRONTEND_ORIGIN
        return [item.strip().rstrip("/") for item in configured.split(",") if item.strip()]

    @property
    def max_transcription_upload_bytes(self) -> int:
        return self.MAX_TRANSCRIPTION_UPLOAD_BYTES or self.MAX_UPLOAD_BYTES

    @property
    def login_rate_count(self) -> int:
        value = self.LOGIN_RATE_LIMIT.strip().lower()
        if value.isdigit():
            return max(1, min(100, int(value)))
        try:
            count = int(value.split("/", 1)[0])
        except (ValueError, IndexError) as exc:
            raise ValueError("LOGIN_RATE_LIMIT must look like 5/15minutes") from exc
        return max(1, min(100, count))

    @property
    def login_rate_window_seconds(self) -> int:
        value = self.LOGIN_RATE_LIMIT.strip().lower()
        if "/" not in value:
            return self.LOGIN_RATE_WINDOW_SECONDS
        window = value.split("/", 1)[1]
        digits = "".join(character for character in window if character.isdigit())
        amount = int(digits or "1")
        if "hour" in window:
            return amount * 3600
        if "minute" in window:
            return amount * 60
        return amount

    @property
    def trusted_hosts(self) -> list[str]:
        return [item.strip() for item in self.TRUSTED_HOSTS.split(",") if item.strip()]

    @property
    def api_path(self) -> str:
        return "openapi/v2.0" if self.YEASTAR_API_VERSION == "v2" else "openapi/v1.0"


@lru_cache
def get_settings() -> Settings:
    return Settings()
