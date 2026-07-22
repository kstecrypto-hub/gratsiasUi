from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import Field, SecretStr

from app.models.enums import YeastarConnectionStatus
from app.schemas.common import APIModel, IntegrationState


class ConfigurationResponse(APIModel):
    administrator_configured: bool
    application_security_configured: bool
    yeastar: IntegrationState
    openai: IntegrationState
    database: IntegrationState
    processing: IntegrationState


class HealthResponse(APIModel):
    status: str
    service: str | None = None
    configured: bool | None = None
    message: str | None = None


class SanitizedYeastarConnectionSettings(APIModel):
    BaseUrl: str
    ClientId: str
    ClientSecret: str
    DateFormat: str
    PageSize: int = Field(ge=1, le=10000)
    IgnoreSslErrors: bool


class SanitizedYeastarConnectionConfig(APIModel):
    Name: str
    Settings: SanitizedYeastarConnectionSettings


class YeastarConnectionSettingsUpdate(APIModel):
    """Write contract kept permissive enough for safe local error reporting."""

    BaseUrl: str
    ClientId: str
    ClientSecret: SecretStr
    DateFormat: str
    PageSize: object
    IgnoreSslErrors: object


class YeastarConnectionConfigurationUpdate(APIModel):
    Name: str
    Settings: YeastarConnectionSettingsUpdate


class YeastarConfigurationValidationError(APIModel):
    field: str
    message: str


class YeastarConfigurationValidationResponse(APIModel):
    valid: bool
    errors: list[YeastarConfigurationValidationError]
    configuration: SanitizedYeastarConnectionConfig


class YeastarCapabilitiesResponse(APIModel):
    extensions: bool | None
    cdr_v2: bool | None
    # The concrete read-only CDR API selected after the connection test.  This
    # is separate from ``cdr_v2`` so legacy appliance installations can report
    # their usable v1 mode without being mistaken for an unsupported PBX.
    cdr_api_version: Literal["v1", "v2"] | None
    recordings: bool | None


class YeastarStatusResponse(APIModel):
    status: YeastarConnectionStatus
    configured: bool
    last_tested_at: datetime | None
    last_successful_connection_at: datetime | None
    model_name: str | None
    firmware_version: str | None
    capabilities: YeastarCapabilitiesResponse
    message: str
    last_error_reference: str | None = None


class YeastarConnectionTestResult(APIModel):
    status: YeastarConnectionStatus
    model: str | None
    firmwareVersion: str | None


class YeastarConnectionTestResponse(APIModel):
    configurationAccepted: bool
    configuration: SanitizedYeastarConnectionConfig
    connection: YeastarConnectionTestResult


class YeastarHealthResponse(APIModel):
    status: YeastarConnectionStatus
    last_successful_connection_at: datetime | None = None


class SanitizedOpenAIConfiguration(APIModel):
    """A write-only OpenAI credential marker safe to send to the browser."""

    api_key: str


class OpenAIConfigurationUpdate(APIModel):
    """The OpenAI API key may be blank to retain the existing effective key."""

    api_key: SecretStr = Field(max_length=1024)


class OpenAIConfigurationValidationError(APIModel):
    field: str
    message: str


class OpenAIConfigurationValidationResponse(APIModel):
    valid: bool
    errors: list[OpenAIConfigurationValidationError]
    configuration: SanitizedOpenAIConfiguration


class OpenAIConnectionTestResult(APIModel):
    status: Literal["connected"]


class OpenAIConnectionTestResponse(APIModel):
    configurationAccepted: bool
    configuration: SanitizedOpenAIConfiguration
    connection: OpenAIConnectionTestResult
