from __future__ import annotations

import asyncio
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Protocol
from urllib.parse import unquote, urlsplit

from pydantic import BaseModel, ConfigDict, Field

from app.services.yeastar.errors import (
    YeastarAPIError,
    YeastarRecordingDownloadLimitError,
    YeastarSecurityError,
)
from app.services.yeastar.extensions import QueryValue, YeastarGetClient


class RecordingRecord(BaseModel):
    model_config = ConfigDict(extra="allow")

    id: str | int
    time: str | int | float | None = None
    uid: str
    call_from: str | None = None
    call_to: str | None = None
    duration: int | float | None = None
    size: int | None = None
    call_type: str | None = None
    file: str | None = None
    call_from_number: str | None = None
    call_to_number: str | None = None
    archive_status: str | int | None = None

    def provider_dict(self) -> dict[str, object]:
        result = self.model_dump()
        result["id"] = str(self.id)
        return result


class RecordingSearchPage(BaseModel):
    model_config = ConfigDict(extra="ignore")

    data: list[RecordingRecord] = Field(default_factory=list)
    total_number: int = Field(default=0, ge=0)


@dataclass(frozen=True)
class RecordingDownload:
    path: Path
    size_bytes: int
    content_type: str


class YeastarRecordingClient(YeastarGetClient, Protocol):
    async def request_recording_download_resource(
        self, recording_id: str
    ) -> tuple[Mapping[str, object], str]: ...

    async def stream_download(
        self,
        resource_path: str,
        *,
        recording_id: str,
        access_token: str,
        destination: Path,
    ) -> RecordingDownload: ...


def validate_download_resource_path(value: str) -> str:
    """Validate a PBX temporary resource while keeping its host fixed.

    Yeastar revisions issue the resource below ``/api/`` with either the
    documented ``/api/download/...`` form or an opaque temporary path.  The
    value is deliberately kept relative, so it cannot choose a host, port,
    query string, or fragment.  A legacy missing-leading-slash form is
    canonicalized before it is sent to the configured PBX.
    """
    if not value or any(ord(character) < 32 for character in value):
        raise YeastarSecurityError("Recording download path is invalid.")
    parsed = urlsplit(value)
    if parsed.scheme or parsed.netloc or parsed.query or parsed.fragment:
        raise YeastarSecurityError("Recording download path must be relative to the phone system.")

    resource_path = parsed.path
    if resource_path.startswith("api/"):
        resource_path = f"/{resource_path}"
    if not resource_path.startswith("/api/") or resource_path.startswith("//"):
        raise YeastarSecurityError("Recording download path is invalid.")
    if "\\" in resource_path:
        raise YeastarSecurityError("Recording download path is invalid.")

    decoded = resource_path
    for _ in range(4):
        next_value = unquote(decoded)
        if next_value == decoded:
            break
        decoded = next_value
    decoded_parts = urlsplit(decoded)
    normalized = decoded_parts.path.replace("\\", "/")
    if (
        decoded_parts.scheme
        or decoded_parts.netloc
        or decoded_parts.query
        or decoded_parts.fragment
        or not normalized.startswith("/api/")
        or normalized.startswith("//")
    ):
        raise YeastarSecurityError("Recording download path is invalid.")
    if ".." in normalized or any(part in {".", ".."} for part in normalized.split("/")):
        raise YeastarSecurityError("Recording download path contains traversal.")
    return resource_path


class YeastarRecordings:
    def __init__(
        self,
        client: YeastarRecordingClient,
        redis: object,
        storage_root: Path,
        *,
        page_size: int = 500,
        download_limit_retries: int = 1,
        download_lock_timeout: int = 3600,
    ) -> None:
        if not 1 <= page_size <= 10_000:
            raise ValueError("page_size must be between 1 and 10000")
        if not 0 <= download_limit_retries <= 5:
            raise ValueError("download_limit_retries must be between 0 and 5")
        self.client = client
        self.redis = redis
        self.storage_root = storage_root.resolve()
        self.page_size = page_size
        self.download_limit_retries = download_limit_retries
        self.download_lock_timeout = download_lock_timeout

    async def page(
        self,
        *,
        page: int,
        start_time: datetime,
        end_time: datetime,
        caller: str | None = None,
        callee: str | None = None,
        ids: list[str] | None = None,
    ) -> RecordingSearchPage:
        if page < 1:
            raise ValueError("Invalid recording pagination")
        params: dict[str, QueryValue] = {
            "page": page,
            "page_size": self.page_size,
            "sort_by": "id",
            "order_by": "asc",
            "start_time": int(start_time.timestamp()),
            "end_time": int(end_time.timestamp()),
        }
        if caller:
            params["caller"] = caller
        if callee:
            params["callee"] = callee
        if ids:
            params["ids"] = ",".join(ids)
        payload = await self.client.get("recording/search", version="v1", params=params)
        try:
            result = RecordingSearchPage.model_validate(payload)
        except ValueError as exc:
            raise YeastarAPIError("Phone system returned an invalid recording list.") from exc
        if result.total_number == 0 and result.data:
            result.total_number = len(result.data)
        return result

    async def search_all(
        self,
        start_time: datetime,
        end_time: datetime,
        *,
        caller: str | None = None,
        callee: str | None = None,
        ids: list[str] | None = None,
    ) -> list[RecordingRecord]:
        records: list[RecordingRecord] = []
        page_number = 1
        while True:
            page = await self.page(
                page=page_number,
                start_time=start_time,
                end_time=end_time,
                caller=caller,
                callee=callee,
                ids=ids,
            )
            records.extend(page.data)
            total = page.total_number or len(records)
            if len(records) >= total:
                return records
            if not page.data or page_number >= 10_000:
                raise YeastarAPIError("Phone system returned invalid recording pagination.")
            page_number += 1

    async def find_by_cdr_uid(
        self,
        uid: str,
        start_time: datetime,
        end_time: datetime,
    ) -> list[RecordingRecord]:
        return [
            item
            for item in await self.search_all(start_time, end_time)
            if item.uid == uid
        ]

    async def _download_resource(self, recording_id: str) -> tuple[str, str]:
        for attempt in range(self.download_limit_retries + 1):
            try:
                payload, access_token = (
                    await self.client.request_recording_download_resource(recording_id)
                )
                supplied = payload.get("download_resource_url")
                if not isinstance(supplied, str):
                    raise YeastarAPIError("Phone system did not provide a recording download.")
                return validate_download_resource_path(supplied), access_token
            except YeastarRecordingDownloadLimitError:
                if attempt >= self.download_limit_retries:
                    raise
                await asyncio.sleep(min(30, 2 ** (attempt + 1)))
        raise AssertionError("unreachable")

    def _safe_destination(self, destination: Path) -> Path:
        if destination.is_symlink():
            raise YeastarSecurityError("Recording destination cannot be a symbolic link.")
        resolved = destination.resolve()
        if not resolved.is_relative_to(self.storage_root):
            raise YeastarSecurityError("Recording destination is outside application storage.")
        resolved.parent.mkdir(parents=True, exist_ok=True)
        return resolved

    async def download(self, recording_id: str, destination: Path) -> RecordingDownload:
        if not recording_id or len(recording_id) > 255:
            raise ValueError("Invalid recording ID")
        destination = self._safe_destination(destination)
        temporary = destination.with_suffix(destination.suffix + ".part")
        # A single global lock is the safe default for MP3-capable systems. The
        # lock is shared by web and worker processes through Redis.
        lock = self.redis.lock(
            "yca:yeastar:recording-download:slot:0",
            timeout=self.download_lock_timeout,
            blocking_timeout=self.download_lock_timeout,
        )
        async with lock:
            temporary.unlink(missing_ok=True)
            resource_path, access_token = await self._download_resource(recording_id)
            try:
                result = await self.client.stream_download(
                    resource_path,
                    recording_id=recording_id,
                    access_token=access_token,
                    destination=temporary,
                )
                if result.size_bytes <= 0:
                    raise YeastarAPIError("Phone system returned an empty recording.")
                temporary.replace(destination)
                return RecordingDownload(
                    path=destination,
                    size_bytes=result.size_bytes,
                    content_type=result.content_type,
                )
            except Exception:
                temporary.unlink(missing_ok=True)
                raise
