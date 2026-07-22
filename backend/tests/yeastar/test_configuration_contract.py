from __future__ import annotations

import os
from datetime import UTC, datetime
from zoneinfo import ZoneInfo

import httpx
import pytest
from pydantic import SecretStr, ValidationError

from app.core.config import Settings
from app.services.yeastar.capabilities import (
    build_capability_profile,
    detect_edition,
    firmware_at_least,
    parse_firmware_version,
    supports_legacy_cdr_v1,
)
from app.services.yeastar.datetime_formatter import YeastarDateTimeFormatter
from app.services.yeastar.http_client import YeastarHttpClient
from app.services.yeastar.schemas import (
    YeastarConnectionConfig,
    YeastarConnectionSettings,
    sanitized_configuration,
)


def configured_settings(**overrides: object) -> Settings:
    values: dict[str, object] = {
        "APP_ENV": "test",
        "APP_SECRET_KEY": "s" * 32,
        "YEASTAR_BASE_URL": "https://pbx.internal:8088",
        "YEASTAR_CLIENT_ID": "client-id",
        "YEASTAR_CLIENT_SECRET": "client-secret",
    }
    values.update(overrides)
    return Settings(_env_file=None, **values)


def test_exact_external_configuration_contract_and_defaults() -> None:
    config = YeastarConnectionConfig.model_validate(
        {
            "Name": "Office",
            "Settings": {
                "BaseUrl": "https://pbx.internal",
                "ClientId": "id",
                "ClientSecret": "secret",
            },
        }
    )

    assert config.Name == "Office"
    assert config.Settings.DateFormat == "MM/dd/yyyy HH:mm:ss"
    assert config.Settings.PageSize == 500
    assert config.Settings.IgnoreSslErrors is True
    assert config.Settings.ClientSecret.get_secret_value() == "secret"
    assert "secret" not in config.model_dump_json()

    with pytest.raises(ValidationError):
        YeastarConnectionConfig.model_validate(
            {"Name": "", "Settings": {"BaseUrl": "", "ClientId": "", "ClientSecret": "", "Extra": 1}}
        )


def test_invalid_nonempty_values_do_not_crash_settings_and_block_network_config() -> None:
    settings = configured_settings(
        YEASTAR_PAGE_SIZE="many",
        YEASTAR_IGNORE_SSL_ERRORS="maybe",
        YEASTAR_CONNECT_TIMEOUT_SECONDS="forever",
        YEASTAR_AUTH_API_PATH="/%252e%252e/private",
    )

    fields = {error["field"] for error in settings.yeastar_configuration_errors}
    assert "Settings.PageSize" in fields
    assert "Settings.IgnoreSslErrors" in fields
    assert "ConnectTimeout" in fields
    assert "AuthApiPath" in fields
    assert settings.yeastar_configured is False


def test_http_requires_explicit_opt_in_and_old_verify_setting_is_ignored() -> None:
    denied = configured_settings(
        YEASTAR_BASE_URL="http://pbx.internal",
        YEASTAR_ALLOW_HTTP=False,
        YEASTAR_VERIFY_SSL=False,
    )
    allowed = configured_settings(
        YEASTAR_BASE_URL="http://pbx.internal",
        YEASTAR_ALLOW_HTTP=True,
        YEASTAR_VERIFY_SSL=False,
    )

    assert denied.yeastar_configured is False
    assert any(error["field"] == "Settings.BaseUrl" for error in denied.yeastar_configuration_errors)
    assert allowed.yeastar_configured is True
    assert allowed.YEASTAR_IGNORE_SSL_ERRORS is True
    assert allowed.YEASTAR_VERIFY_SSL is False


@pytest.mark.asyncio
async def test_ignore_ssl_errors_only_controls_yeastar_transport(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: list[dict[str, object]] = []
    baseline_tls_environment = {
        key: os.environ.get(key)
        for key in ("SSL_CERT_FILE", "SSL_CERT_DIR", "PYTHONHTTPSVERIFY")
    }

    class FakeAsyncClient:
        async def aclose(self) -> None:
            return None

    def factory(*args: object, **kwargs: object) -> FakeAsyncClient:
        del args
        captured.append(kwargs)
        return FakeAsyncClient()

    monkeypatch.setattr(httpx, "AsyncClient", factory)
    ignoring = configured_settings(
        YEASTAR_IGNORE_SSL_ERRORS=True,
        OPENAI_API_KEY="unrelated-openai-secret",
    )
    verifying = configured_settings(
        YEASTAR_IGNORE_SSL_ERRORS=False,
        OPENAI_API_KEY="unrelated-openai-secret",
    )

    await YeastarHttpClient(ignoring)._ensure_client()
    await YeastarHttpClient(verifying)._ensure_client()

    assert [item["verify"] for item in captured] == [False, True]
    assert ignoring.OPENAI_API_KEY == verifying.OPENAI_API_KEY == "unrelated-openai-secret"
    assert {
        key: os.environ.get(key)
        for key in ("SSL_CERT_FILE", "SSL_CERT_DIR", "PYTHONHTTPSVERIFY")
    } == baseline_tls_environment


@pytest.mark.parametrize(
    "base_url",
    [
        "https://" + "user:password@" + "pbx.internal",
        "https://pbx.internal?access_token=secret",
        "https://pbx.internal:invalid",
        "javascript://pbx.internal",
    ],
)
def test_sanitized_configuration_never_reflects_invalid_base_url(base_url: str) -> None:
    config = YeastarConnectionConfig(
        Name="Office",
        Settings=YeastarConnectionSettings(
            BaseUrl=base_url,
            ClientId="secret-client-id",
            ClientSecret=SecretStr("secret-client-secret"),
        ),
    )

    result = sanitized_configuration(config)

    assert result["Settings"]["BaseUrl"] == "[INVALID]"  # type: ignore[index]
    rendered = repr(result)
    assert "secret-client-id" not in rendered
    assert "secret-client-secret" not in rendered


def test_dotnet_tokens_are_case_sensitive_and_timezone_aware() -> None:
    assert (
        YeastarDateTimeFormatter.dotnet_to_strftime("MM/dd/yyyy HH:mm:ss")
        == "%m/%d/%Y %H:%M:%S"
    )
    instant = datetime(2026, 3, 29, 0, 30, tzinfo=UTC)
    rendered = YeastarDateTimeFormatter.format_pattern(
        instant,
        "yyyy-MM-dd HH:mm:ss",
        ZoneInfo("Europe/Athens"),
    )
    assert rendered == "2026-03-29 02:30:00"

    assert (
        YeastarDateTimeFormatter.dotnet_to_strftime("mm/dd/yyyy HH:MM:ss")
        == "%M/%d/%Y %H:%m:%S"
    )
    with pytest.raises(ValueError):
        YeastarDateTimeFormatter.dotnet_to_strftime("yyyy-M-dd")


def test_firmware_comparison_is_numeric_not_lexical() -> None:
    assert parse_firmware_version("84.23.0.123") == (84, 23, 0, 123)
    assert firmware_at_least("84.100.0.1", (84, 23, 0, 123)) is True
    assert firmware_at_least("84.9.99.999", (84, 23, 0, 123)) is False
    assert parse_firmware_version("84.23.0") is None


@pytest.mark.parametrize("model_name", ["Yeastar P550", "Yeastar P560", "Yeastar P570"])
def test_known_legacy_appliances_select_cdr_v1(model_name: str) -> None:
    profile = build_capability_profile(model_name, "37.20.0.78")

    assert detect_edition(model_name).value == "appliance"
    assert profile.state == "supported"
    assert profile.cdr_api_version == "v1"
    assert profile.cdr_v2 is False
    assert profile.minimum_cdr_v2_version == "37.23.0.123"


def test_legacy_appliance_range_is_known_and_does_not_override_v2() -> None:
    assert supports_legacy_cdr_v1("Yeastar P560", "37.7.0.16") is True
    assert supports_legacy_cdr_v1("Yeastar P560", "37.20.0.78") is True
    assert supports_legacy_cdr_v1("Yeastar P560", "37.23.0.123") is False
    assert supports_legacy_cdr_v1("Yeastar P560", "37.7.0.15") is False
    assert supports_legacy_cdr_v1("P-Series Appliance", "37.20.0.78") is False

    profile = build_capability_profile("Yeastar P560", "37.23.0.123")
    assert profile.state == "supported"
    assert profile.cdr_api_version == "v2"
    assert profile.cdr_v2 is True
