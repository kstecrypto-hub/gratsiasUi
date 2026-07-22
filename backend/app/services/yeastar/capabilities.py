from __future__ import annotations

import re

from app.services.yeastar.schemas import CapabilityProfile, YeastarEdition


_VERSION = re.compile(r"^\s*(\d+)\.(\d+)\.(\d+)\.(\d+)\s*$")
_MINIMUM_CDR_V2 = {
    YeastarEdition.CLOUD: (84, 23, 0, 123),
    YeastarEdition.SOFTWARE: (83, 23, 0, 123),
    YeastarEdition.APPLIANCE: (37, 23, 0, 123),
}
_LEGACY_APPLIANCE_MODELS = re.compile(r"\bp(?:550|560|570)\b")
_MINIMUM_LEGACY_APPLIANCE_FIRMWARE = (37, 7, 0, 16)


def parse_firmware_version(value: str) -> tuple[int, int, int, int] | None:
    matched = _VERSION.fullmatch(value or "")
    if matched is None:
        return None
    first, second, third, fourth = matched.groups()
    return int(first), int(second), int(third), int(fourth)


def detect_edition(model_name: str) -> YeastarEdition:
    normalized = " ".join((model_name or "").casefold().split())
    if "p-series cloud edition" in normalized:
        return YeastarEdition.CLOUD
    if "p-series software edition" in normalized:
        return YeastarEdition.SOFTWARE
    if "p-series appliance" in normalized or "p-series pbx system" in normalized:
        return YeastarEdition.APPLIANCE
    # Appliance responses on older P-Series firmware identify the hardware
    # directly (for example, ``Yeastar P560``) rather than using the edition
    # label returned by newer versions.
    if _LEGACY_APPLIANCE_MODELS.search(normalized):
        return YeastarEdition.APPLIANCE
    return YeastarEdition.UNKNOWN


def firmware_at_least(value: str, minimum: tuple[int, int, int, int]) -> bool | None:
    parsed = parse_firmware_version(value)
    return None if parsed is None else parsed >= minimum


def supports_legacy_cdr_v1(model_name: str, firmware_version: str) -> bool:
    """Return whether a known appliance can use the timestamp-based CDR V1 API.

    This is deliberately a narrow allow-list.  A lower version alone is not
    enough to establish support: the appliance model must be recognized and
    its firmware must be in the appliance V1 range.  Versions at or above the
    V2 threshold use V2 instead.
    """

    normalized = " ".join((model_name or "").casefold().split())
    parsed = parse_firmware_version(firmware_version)
    return bool(
        _LEGACY_APPLIANCE_MODELS.search(normalized)
        and parsed is not None
        and _MINIMUM_LEGACY_APPLIANCE_FIRMWARE
        <= parsed
        < _MINIMUM_CDR_V2[YeastarEdition.APPLIANCE]
    )


def _version_text(value: tuple[int, int, int, int]) -> str:
    return ".".join(str(item) for item in value)


def build_capability_profile(model_name: str, firmware_version: str) -> CapabilityProfile:
    edition = detect_edition(model_name)
    if edition == YeastarEdition.UNKNOWN:
        return CapabilityProfile(
            edition=edition,
            state="unknown",
            extensions=True,
            cdr_v2=None,
            cdr_api_version=None,
            recordings=True,
            firmware_version=firmware_version,
            message=(
                "The phone-system edition could not be identified. "
                "Ask your IT administrator to confirm it."
            ),
        )
    minimum = _MINIMUM_CDR_V2[edition]
    supported = firmware_at_least(firmware_version, minimum)
    if supported is None:
        return CapabilityProfile(
            edition=edition,
            state="unknown",
            extensions=True,
            cdr_v2=None,
            cdr_api_version=None,
            recordings=True,
            firmware_version=firmware_version,
            minimum_cdr_v2_version=_version_text(minimum),
            message=(
                "The phone-system version could not be recognized. "
                "Ask your IT administrator to confirm it."
            ),
        )
    if not supported and supports_legacy_cdr_v1(model_name, firmware_version):
        return CapabilityProfile(
            edition=edition,
            state="supported",
            extensions=True,
            cdr_v2=False,
            cdr_api_version="v1",
            recordings=True,
            firmware_version=firmware_version,
            minimum_cdr_v2_version=_version_text(minimum),
            message="Phone system connected using the legacy call API.",
        )
    if not supported:
        return CapabilityProfile(
            edition=edition,
            state="unsupported",
            extensions=True,
            cdr_v2=False,
            cdr_api_version=None,
            recordings=True,
            firmware_version=firmware_version,
            minimum_cdr_v2_version=_version_text(minimum),
            message="The installed phone-system version does not support the required call API.",
        )
    return CapabilityProfile(
        edition=edition,
        state="supported",
        extensions=True,
        cdr_v2=True,
        cdr_api_version="v2",
        recordings=True,
        firmware_version=firmware_version,
        minimum_cdr_v2_version=_version_text(minimum),
        message="Phone system connected",
    )


__all__ = [
    "build_capability_profile",
    "detect_edition",
    "firmware_at_least",
    "parse_firmware_version",
    "supports_legacy_cdr_v1",
]
