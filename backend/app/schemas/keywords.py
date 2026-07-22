from __future__ import annotations

from datetime import datetime
from uuid import UUID

from pydantic import Field, field_validator

from app.models.enums import Severity
from app.schemas.common import APIModel


class KeywordVariantInput(APIModel):
    phrase: str = Field(min_length=1, max_length=500)

    @field_validator("phrase")
    @classmethod
    def trim_phrase(cls, value: str) -> str:
        value = " ".join(value.split())
        if not value:
            raise ValueError("Phrase cannot be empty")
        return value


class KeywordVariantResponse(APIModel):
    id: UUID
    phrase: str


class KeywordCategoryCreate(APIModel):
    name: str = Field(min_length=1, max_length=255)
    description: str | None = Field(default=None, max_length=5000)
    active: bool = True

    @field_validator("name")
    @classmethod
    def trim_name(cls, value: str) -> str:
        return " ".join(value.split())


class KeywordCategoryUpdate(APIModel):
    name: str | None = Field(default=None, min_length=1, max_length=255)
    description: str | None = Field(default=None, max_length=5000)
    active: bool | None = None


class KeywordCategoryResponse(APIModel):
    id: UUID
    name: str
    description: str | None
    active: bool
    created_at: datetime
    updated_at: datetime
    keyword_count: int = 0


class KeywordCreate(APIModel):
    category_id: UUID
    canonical_phrase: str = Field(min_length=1, max_length=500)
    variants: list[KeywordVariantInput] = Field(default_factory=list, max_length=100)
    accent_insensitive: bool = True
    whole_word: bool = True
    exact_phrase: bool = True
    fuzzy_match: bool = False
    fuzzy_threshold: float = Field(default=0.9, ge=0.5, le=1.0)
    active: bool = True
    severity: Severity = Severity.MEDIUM
    notes: str | None = Field(default=None, max_length=5000)

    @field_validator("variants", mode="before")
    @classmethod
    def accept_variant_strings(cls, value: object) -> object:
        if isinstance(value, list):
            return [{"phrase": item} if isinstance(item, str) else item for item in value]
        return value


class KeywordUpdate(APIModel):
    category_id: UUID | None = None
    canonical_phrase: str | None = Field(default=None, min_length=1, max_length=500)
    variants: list[KeywordVariantInput] | None = Field(default=None, max_length=100)
    accent_insensitive: bool | None = None
    whole_word: bool | None = None
    exact_phrase: bool | None = None
    fuzzy_match: bool | None = None
    fuzzy_threshold: float | None = Field(default=None, ge=0.5, le=1.0)
    active: bool | None = None
    severity: Severity | None = None
    notes: str | None = Field(default=None, max_length=5000)

    @field_validator("variants", mode="before")
    @classmethod
    def accept_variant_strings(cls, value: object) -> object:
        if isinstance(value, list):
            return [{"phrase": item} if isinstance(item, str) else item for item in value]
        return value


class KeywordResponse(APIModel):
    id: UUID
    category_id: UUID
    canonical_phrase: str
    variants: list[KeywordVariantResponse]
    accent_insensitive: bool
    whole_word: bool
    exact_phrase: bool
    fuzzy_match: bool
    fuzzy_threshold: float
    active: bool
    severity: Severity
    notes: str | None
    created_at: datetime
    updated_at: datetime
