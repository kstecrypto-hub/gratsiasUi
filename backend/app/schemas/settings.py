from __future__ import annotations

from pydantic import Field, model_validator

from app.schemas.common import APIModel


class ApplicationSettingsResponse(APIModel):
    default_language: str
    transcript_retention_days: int
    delete_audio_after_transcription: bool
    max_parallel_transcriptions: int
    maximum_simultaneous_transcriptions: int | None = None
    company_vocabulary: str
    default_timezone: str

    @model_validator(mode="after")
    def populate_legacy_parallel_name(self) -> "ApplicationSettingsResponse":
        self.maximum_simultaneous_transcriptions = self.max_parallel_transcriptions
        return self


class ApplicationSettingsUpdate(APIModel):
    default_language: str | None = Field(default=None, min_length=2, max_length=16)
    transcript_retention_days: int | None = Field(default=None, ge=1, le=3650)
    delete_audio_after_transcription: bool | None = None
    max_parallel_transcriptions: int | None = Field(default=None, ge=1, le=16)
    maximum_simultaneous_transcriptions: int | None = Field(default=None, ge=1, le=16)
    company_vocabulary: str | None = Field(default=None, max_length=10000)
    default_timezone: str | None = Field(default=None, min_length=1, max_length=100)

    @model_validator(mode="after")
    def normalize_parallel_name(self) -> "ApplicationSettingsUpdate":
        if self.max_parallel_transcriptions is None:
            self.max_parallel_transcriptions = self.maximum_simultaneous_transcriptions
        elif (
            self.maximum_simultaneous_transcriptions is not None
            and self.maximum_simultaneous_transcriptions != self.max_parallel_transcriptions
        ):
            raise ValueError("Parallel transcription settings conflict")
        return self
