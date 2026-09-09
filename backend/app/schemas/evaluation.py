"""Human reference contracts. These never inherit production transcript schemas."""

from typing import Annotated, Literal

from pydantic import BaseModel, BeforeValidator, ConfigDict, Field


def strict_channel(value):
    if value is not None and type(value) is not int:
        raise ValueError("Channel must be 0, 1, or null.")
    return value


Channel = Annotated[Literal[0, 1] | None, BeforeValidator(strict_channel)]
Quality = Literal["clean", "normal", "noisy", "very_noisy"]
ShortText = Annotated[str, Field(max_length=500)]


class ReferenceEntities(BaseModel):
    model_config = ConfigDict(extra="forbid")
    names: list[ShortText] = Field(default_factory=list, max_length=100)
    telephone_numbers: list[ShortText] = Field(default_factory=list, max_length=100)
    licence_plates: list[ShortText] = Field(default_factory=list, max_length=100)
    vehicle_models: list[ShortText] = Field(default_factory=list, max_length=100)


class HumanSegment(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    speaker: Literal["Operator", "Customer", "Other", "Unknown"] = "Unknown"
    channel: Channel = None
    start: float = Field(default=0, ge=0, strict=True)
    end: float = Field(default=0, ge=0, strict=True)
    text: str = Field(default="", max_length=20000)
    exclude_from_wer: bool = Field(default=False, strict=True)
    entities: ReferenceEntities = Field(default_factory=ReferenceEntities)


class ReferenceDraft(BaseModel):
    model_config = ConfigDict(extra="forbid")
    quality: Quality | None = None
    operator_channel: Channel = None
    operator_channel_answered: bool = Field(default=False, strict=True)
    expected_keywords: list[ShortText] = Field(default_factory=list, max_length=5000)
    segments: list[HumanSegment] = Field(default_factory=list, max_length=5000)


class SaveReference(ReferenceDraft):
    revision: str = Field(pattern=r"^[a-f0-9]{64}$")


class VerifyReference(BaseModel):
    model_config = ConfigDict(extra="forbid")
    revision: str = Field(pattern=r"^[a-f0-9]{64}$")
