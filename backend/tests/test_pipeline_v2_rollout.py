from __future__ import annotations

import pytest

from app.core.config import Settings
from app.models import ProcessingJobItem
from app.workers.pipeline import (
    LEGACY_PIPELINE_VERSION,
    PIPELINE_V2_VERSION,
    ReprocessStateError,
    _effective_pipeline_version,
)


def _settings(*, default: str = "legacy-v1", v2_enabled: bool = False) -> Settings:
    return Settings(
        APP_ENV="test",
        TRANSCRIPTION_PIPELINE_DEFAULT=default,
        TRANSCRIPTION_PIPELINE_V2_ENABLED=v2_enabled,
    )


def test_settings_default_to_legacy_with_v2_disabled() -> None:
    settings = Settings(APP_ENV="test")
    assert settings.TRANSCRIPTION_PIPELINE_DEFAULT == "legacy-v1"
    assert settings.TRANSCRIPTION_PIPELINE_V2_ENABLED is False


def test_effective_version_keeps_legacy_default() -> None:
    settings = _settings()
    item = ProcessingJobItem(requested_pipeline_version=None)
    assert _effective_pipeline_version(item, settings) == LEGACY_PIPELINE_VERSION


def test_effective_version_uses_v2_default_only_when_enabled() -> None:
    enabled = _settings(default=PIPELINE_V2_VERSION, v2_enabled=True)
    disabled = _settings(default=PIPELINE_V2_VERSION, v2_enabled=False)
    item = ProcessingJobItem(requested_pipeline_version=None)
    assert _effective_pipeline_version(item, enabled) == PIPELINE_V2_VERSION
    assert _effective_pipeline_version(item, disabled) == LEGACY_PIPELINE_VERSION


def test_explicit_v2_is_rejected_while_disabled() -> None:
    settings = _settings(v2_enabled=False)
    item = ProcessingJobItem(requested_pipeline_version=PIPELINE_V2_VERSION)
    with pytest.raises(ReprocessStateError, match="V2 is disabled"):
        _effective_pipeline_version(item, settings)


def test_explicit_v2_is_allowed_while_enabled() -> None:
    settings = _settings(v2_enabled=True)
    item = ProcessingJobItem(requested_pipeline_version=PIPELINE_V2_VERSION)
    assert _effective_pipeline_version(item, settings) == PIPELINE_V2_VERSION


def test_explicit_legacy_is_always_allowed() -> None:
    settings = _settings(v2_enabled=False)
    item = ProcessingJobItem(requested_pipeline_version=LEGACY_PIPELINE_VERSION)
    assert _effective_pipeline_version(item, settings) == LEGACY_PIPELINE_VERSION


def test_unsupported_pipeline_version_is_rejected() -> None:
    settings = _settings(v2_enabled=True)
    item = ProcessingJobItem(requested_pipeline_version="future-unsafe")
    with pytest.raises(ReprocessStateError, match="not available"):
        _effective_pipeline_version(item, settings)
