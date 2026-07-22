from __future__ import annotations

import mimetypes
import os
import json
from collections.abc import Mapping, Sequence
from datetime import datetime
from pathlib import Path
from typing import Literal, cast
from zoneinfo import ZoneInfo

import httpx
from redis.asyncio import Redis

from app.core.config import Settings, get_settings
from app.core.redis import get_redis
from app.core.time import ensure_utc
from app.services.yeastar.capabilities import build_capability_profile
from app.services.yeastar.cdr import CDRSummary, YeastarCDR
from app.services.yeastar.datetime_formatter import YeastarDateTimeFormatter
from app.services.yeastar.errors import (
    YeastarAPIError,
    YeastarConfigurationError,
    YeastarError,
    YeastarOperationError,
    YeastarResponseError,
    YeastarSecurityError,
    YeastarTokenExpiredError,
    YeastarTokenRefreshError,
)
from app.services.yeastar.error_mapping import map_yeastar_error
from app.services.yeastar.extensions import QueryValue, YeastarExtensions
from app.services.yeastar.http_client import JsonObject, JsonValue, YeastarHttpClient
from app.services.yeastar.recordings import (
    RecordingDownload,
    YeastarRecordings,
    validate_download_resource_path,
)
from app.services.yeastar.schemas import (
    AuthTrigger,
    CapabilityProfile,
    ConnectionState,
    SystemInformation,
)
from app.services.yeastar.token_manager import YeastarTokenManager


class YeastarClient:
    """Domain facade sharing the Redis-backed token lifecycle across all operations."""

    def __init__(
        self,
        settings: Settings | None = None,
        redis: Redis | None = None,
        http_client: httpx.AsyncClient | None = None,
        *,
        trigger: AuthTrigger = AuthTrigger.CALL_ANALYSIS,
        system_date_format: str | None = None,
        system_time_format: str | None = None,
        cdr_api_version: Literal["v1", "v2"] = "v2",
    ) -> None:
        if cdr_api_version not in {"v1", "v2"}:
            raise ValueError("cdr_api_version must be v1 or v2")
        self.settings = settings or get_settings()
        self.redis = redis or get_redis()
        self.trigger = trigger
        self.system_date_format = system_date_format
        self.system_time_format = system_time_format
        self.cdr_api_version = cdr_api_version
        # V1 has no documented per-UID detail endpoint.  Preserve every
        # normalized source row so its synthetic detail retains duplicate
        # legacy call legs rather than silently dropping them.
        self._cdr_summaries: dict[str, list[CDRSummary]] = {}
        self.http = YeastarHttpClient(self.settings, http_client=http_client)
        self.token_manager = YeastarTokenManager(
            self.settings,
            self.redis,
            http_client=self.http,
        )

    async def __aenter__(self) -> "YeastarClient":
        # Client construction and context entry intentionally perform no I/O.
        return self

    async def __aexit__(self, *_: object) -> None:
        await self.aclose()

    async def aclose(self) -> None:
        await self.http.aclose()

    def _api_path(self, endpoint: str, version: str) -> str:
        normalized = endpoint.strip().lstrip("/")
        if normalized.startswith("system/"):
            prefix = self.settings.YEASTAR_SYSTEM_API_PATH
        elif normalized.startswith("extension/"):
            prefix = self.settings.YEASTAR_EXTENSION_API_PATH
        elif normalized.startswith("cdr/"):
            prefix = (
                self.settings.YEASTAR_CDR_API_PATH
                if version in {"2", "v2", "v2.0"}
                else self.settings.YEASTAR_SYSTEM_API_PATH
            )
        elif normalized.startswith("recording/"):
            prefix = self.settings.YEASTAR_RECORDING_API_PATH
        elif normalized.startswith("autorecord/"):
            prefix = self.settings.YEASTAR_RECORDING_API_PATH
        else:
            prefix = (
                self.settings.YEASTAR_CDR_API_PATH
                if version in {"2", "v2", "v2.0"}
                else self.settings.YEASTAR_SYSTEM_API_PATH
            )
        return f"{str(prefix).rstrip('/')}/{normalized}"

    def _cdr(self, *, api_version: Literal["v1", "v2"] | None = None) -> YeastarCDR:
        return YeastarCDR(
            self,
            self.settings.yeastar_page_size,
            api_version=api_version or self.cdr_api_version,
        )

    async def _authorized_get(
        self,
        endpoint: str,
        *,
        version: str = "v1",
        params: Mapping[str, QueryValue] | None = None,
    ) -> JsonObject:
        return await self.http.protected_request(
            "GET",
            self._api_path(endpoint, version),
            token_manager=self.token_manager,
            trigger=self.trigger,
            params=cast(dict[str, JsonValue], dict(params or {})),
        )

    async def get(
        self,
        endpoint: str,
        *,
        version: str = "v1",
        params: Mapping[str, QueryValue] | None = None,
    ) -> Mapping[str, object]:
        return await self._authorized_get(endpoint, version=version, params=params)

    async def get_with_token(
        self,
        endpoint: str,
        *,
        access_token: str,
        version: str = "v1",
        params: Mapping[str, QueryValue] | None = None,
    ) -> Mapping[str, object]:
        return await self.http.get_with_token(
            self._api_path(endpoint, version),
            access_token,
            params=cast(dict[str, JsonValue], dict(params or {})),
        )

    async def access_token(self) -> str:
        return await self.token_manager.get_access_token(self.trigger)

    async def authenticate(self, *, force: bool = False) -> str:
        if force:
            return await self.token_manager.refresh_access_token()
        return await self.access_token()

    async def refresh_access_token(self) -> str:
        return await self.token_manager.refresh_access_token()

    async def clear_local_token_state(self) -> None:
        await self.token_manager.clear_local_token_state()

    async def open_circuit(
        self, reason: ConnectionState, *, last_errcode: int | None = None
    ) -> None:
        await self.token_manager.open_circuit(reason, last_errcode=last_errcode)

    async def system_information(self) -> SystemInformation:
        response = await self._authorized_get("system/information", version="v1")
        data = response.get("data")
        if not isinstance(data, dict):
            raise YeastarResponseError("Phone system returned invalid system information.")
        try:
            return SystemInformation.model_validate(data)
        except ValueError as exc:
            raise YeastarResponseError(
                "Phone system returned invalid system information."
            ) from exc

    async def inspect_connection(
        self,
    ) -> tuple[SystemInformation, CapabilityProfile]:
        information = await self.system_information()
        capabilities = build_capability_profile(
            information.model_name or "", information.firmware_version or ""
        )
        detected_format_present = bool(
            information.system_date_format or information.system_time_format
        )
        # Legacy CDR V1 receives UTC epoch bounds, so it never parses the PBX
        # display format.  Some legacy appliances advertise a 12-hour pattern
        # without AM/PM; that must not block a safe timestamp-based path.
        if detected_format_present and capabilities.cdr_api_version != "v1":
            try:
                if not information.system_date_format or not information.system_time_format:
                    raise ValueError("Incomplete detected format")
                YeastarDateTimeFormatter.dotnet_to_strftime(
                    f"{information.system_date_format} {information.system_time_format}"
                )
            except ValueError:
                capabilities = capabilities.model_copy(
                    update={
                        "state": "unknown",
                        "cdr_v2": None,
                        "cdr_api_version": None,
                        "message": (
                            "The phone-system date and time format is not supported. "
                            "Ask your IT administrator to confirm it."
                        ),
                    }
                )
        await YeastarExtensions(self, self.settings.yeastar_page_size).verify_read_access()
        if capabilities.cdr_api_version is not None:
            self.cdr_api_version = capabilities.cdr_api_version
            await self._cdr(
                api_version=capabilities.cdr_api_version
            ).verify_read_access()
        return information, capabilities

    async def test_connection(self) -> bool:
        await self.inspect_connection()
        return True

    async def stereo_separated_recording_enabled(self) -> bool:
        cache_key = "yca:yeastar:stereo-capability:v1"
        cached = await self.redis.get(cache_key)
        if cached in {b"0", b"1", "0", "1"}:
            return cached in {b"1", "1"}
        response = await self._authorized_get("autorecord/get", version="v1")
        data = response.get("data")
        top_level = response.get("auto_record")
        auto_record: object = top_level
        if not isinstance(auto_record, dict) and isinstance(data, dict):
            auto_record = data.get("auto_record", data)
        enabled = bool(
            isinstance(auto_record, dict)
            and str(auto_record.get("enb_channel_separate", "0")) == "1"
        )
        await self.redis.setex(cache_key, 300, "1" if enabled else "0")
        return enabled

    async def list_extensions(self) -> list[dict[str, object]]:
        records = await YeastarExtensions(
            self, self.settings.yeastar_page_size
        ).list_all()
        return [record.provider_dict() for record in records]

    def _format_cdr_datetime(self, value: datetime) -> str:
        formatter = YeastarDateTimeFormatter()
        timezone = ZoneInfo(self.settings.APP_TIMEZONE)
        if self.system_date_format is not None or self.system_time_format is not None:
            if not self.system_date_format or not self.system_time_format:
                raise YeastarConfigurationError(
                    "The detected phone-system date and time format is not supported."
                )
            try:
                return formatter.format_cdr_datetime(
                    ensure_utc(value),
                    self.system_date_format,
                    self.system_time_format,
                    timezone,
                )
            except ValueError as exc:
                raise YeastarConfigurationError(
                    "The detected phone-system date and time format is not supported."
                ) from exc
        try:
            return formatter.format_pattern(
                ensure_utc(value), self.settings.YEASTAR_DATE_FORMAT, timezone
            )
        except ValueError as exc:
            raise YeastarConfigurationError(
                "The configured phone-system date format is not supported."
            ) from exc

    def _cdr_time_bound(self, value: datetime) -> str | int:
        if self.cdr_api_version == "v1":
            # The legacy adapter queries CDR V1 using UTC epoch seconds and
            # filters locally.  This intentionally avoids its display-only
            # date/time format.
            return int(ensure_utc(value).timestamp())
        return self._format_cdr_datetime(value)

    def _cache_cdr_summaries(self, summaries: Sequence[CDRSummary]) -> None:
        for summary in summaries:
            uid = summary.uid.strip()
            if uid:
                self._cdr_summaries.setdefault(uid, []).append(summary)

    @staticmethod
    def _normalize_legacy_summaries(
        uid: str,
        *,
        summary: CDRSummary | Mapping[str, object] | None,
        summaries: Sequence[CDRSummary | Mapping[str, object]] | None,
    ) -> list[CDRSummary]:
        if summary is not None and summaries is not None:
            raise ValueError("Pass summary or summaries, not both")
        raw_summaries: Sequence[CDRSummary | Mapping[str, object]]
        if summaries is not None:
            raw_summaries = summaries
        elif summary is not None:
            raw_summaries = [summary]
        else:
            return []
        try:
            normalized = [
                item if isinstance(item, CDRSummary) else CDRSummary.model_validate(item)
                for item in raw_summaries
            ]
        except ValueError as exc:
            raise YeastarAPIError("Legacy call details could not be normalized.") from exc
        if not normalized or any(item.uid.strip() != uid for item in normalized):
            raise YeastarAPIError("Legacy call details do not match the requested call.")
        return normalized

    @staticmethod
    def _cdr_filters(filters: Mapping[str, QueryValue] | None) -> dict[str, QueryValue]:
        result = dict(filters or {})
        if "status" in result:
            result["last_status"] = result.pop("status")
        if "queue" in result:
            result["queue_list"] = result.pop("queue")
        return result

    async def search_cdrs(
        self,
        date_from: datetime,
        date_to: datetime,
        filters: dict[str, QueryValue] | None,
        page: int,
    ) -> dict[str, object]:
        if self.cdr_api_version == "v1":
            # V1's list endpoint has no server-side date filters.  Paging it
            # here would risk discovering calls outside the requested range.
            raise YeastarOperationError(
                "Legacy CDR discovery requires a bounded call search."
            )
        result = await self._cdr().page(
            page=page,
            time_begin=self._cdr_time_bound(date_from),
            time_end=self._cdr_time_bound(date_to),
            filters=self._cdr_filters(filters),
        )
        self._cache_cdr_summaries(result.data)
        return {
            "data": [item.provider_dict() for item in result.data],
            "total_number": result.total_number,
        }

    async def search_all_cdrs(
        self,
        date_from: datetime,
        date_to: datetime,
        filters: dict[str, QueryValue] | None = None,
    ) -> list[CDRSummary]:
        results = await self._cdr().search_all(
            time_begin=self._cdr_time_bound(date_from),
            time_end=self._cdr_time_bound(date_to),
            filters=self._cdr_filters(filters),
        )
        self._cache_cdr_summaries(results)
        return results

    async def get_cdr_detail(
        self,
        uid: str,
        *,
        summary: CDRSummary | Mapping[str, object] | None = None,
        summaries: Sequence[CDRSummary | Mapping[str, object]] | None = None,
    ) -> dict[str, object]:
        if self.cdr_api_version == "v1":
            normalized_uid = uid.strip()
            if not normalized_uid or len(normalized_uid) > 255:
                raise ValueError("Invalid CDR UID")
            provided_summaries = self._normalize_legacy_summaries(
                normalized_uid,
                summary=summary,
                summaries=summaries,
            )
            # Search results are authoritative because a legacy UID can span
            # multiple rows/legs.  A caller's representative summary only
            # fills a cache miss or adds a genuinely absent row.
            legacy_summaries = list(self._cdr_summaries.get(normalized_uid, []))
            if legacy_summaries:
                legacy_summaries.extend(
                    item for item in provided_summaries if item not in legacy_summaries
                )
            else:
                legacy_summaries = provided_summaries
            if not legacy_summaries:
                raise YeastarAPIError(
                    "Legacy call details require a previously discovered call summary."
                )
            detail = self._cdr().detail_from_summaries(legacy_summaries)
        else:
            detail = await self._cdr().detail(uid)
        return detail.provider_dict()

    async def search_recordings(
        self,
        start_time: datetime,
        end_time: datetime,
        caller: str | None = None,
        callee: str | None = None,
        ids: list[str] | None = None,
    ) -> list[dict[str, object]]:
        recordings = YeastarRecordings(
            self,
            self.redis,
            self.settings.STORAGE_ROOT,
            page_size=self.settings.yeastar_page_size,
        )
        records = await recordings.search_all(
            ensure_utc(start_time),
            ensure_utc(end_time),
            caller=caller,
            callee=callee,
            ids=ids,
        )
        return [record.provider_dict() for record in records]

    async def _open_circuit_for_error(self, error: YeastarError) -> None:
        if error.opens_circuit:
            await self.token_manager.open_circuit(
                error.connection_state,
                last_errcode=error.errcode,
            )

    async def request_recording_download_resource(
        self, recording_id: str
    ) -> tuple[Mapping[str, object], str]:
        """Get a temp path and the exact token that succeeded in step one."""
        if not recording_id or len(recording_id) > 255:
            raise ValueError("Invalid recording ID")
        try:
            try:
                token = await self.access_token()
                payload = await self.get_with_token(
                    "recording/download",
                    version="v1",
                    params={"id": recording_id},
                    access_token=token,
                )
            except YeastarTokenExpiredError:
                await self.token_manager.invalidate_access_token(token)
                token = await self.token_manager.refresh_access_token()
                try:
                    payload = await self.get_with_token(
                        "recording/download",
                        version="v1",
                        params={"id": recording_id},
                        access_token=token,
                    )
                except YeastarTokenExpiredError as exc:
                    await self.token_manager.open_circuit(
                        ConnectionState.TOKEN_REFRESH_FAILED,
                        last_errcode=10004,
                    )
                    raise YeastarTokenRefreshError(
                        "The phone-system session could not be renewed.", 10004
                    ) from exc
        except YeastarError as exc:
            await self._open_circuit_for_error(exc)
            raise
        return payload, token

    async def _stream_download_once(
        self,
        resource_path: str,
        *,
        access_token: str,
        destination: Path,
    ) -> RecordingDownload:
        total = 0
        content_type = "application/octet-stream"
        async with self.http.stream(
            "GET",
            resource_path,
            params={"access_token": access_token},
        ) as response:
            if 300 <= response.status_code < 400:
                raise YeastarSecurityError(
                    "Recording download attempted an unexpected redirect."
                )
            content_type = (
                response.headers.get("content-type", "").split(";", 1)[0].lower()
                or "application/octet-stream"
            )
            declared = response.headers.get("content-length")
            if declared and int(declared) > self.settings.MAX_RECORDING_BYTES:
                raise YeastarAPIError("Recording is larger than the configured limit.")

            chunks = response.aiter_bytes(64 * 1024)
            first = await anext(chunks, b"")
            looks_like_json = (
                "json" in content_type or first.lstrip().startswith((b"{", b"["))
            )
            if looks_like_json:
                body = bytearray(first)
                async for chunk in chunks:
                    body.extend(chunk)
                    if len(body) > 64 * 1024:
                        raise YeastarResponseError(
                            "Phone system returned an invalid recording response."
                        )
                try:
                    error_payload = json.loads(body)
                except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                    raise YeastarResponseError(
                        "Phone system returned an invalid recording response."
                    ) from exc
                if isinstance(error_payload, dict):
                    try:
                        errcode = int(error_payload.get("errcode", 0))
                    except (TypeError, ValueError) as exc:
                        raise YeastarResponseError(
                            "Phone system returned an invalid recording response."
                        ) from exc
                    if errcode != 0:
                        raise map_yeastar_error(errcode)
                raise YeastarResponseError(
                    "Phone system returned an invalid recording response."
                )

            with destination.open("xb") as handle:
                for chunk in (first,):
                    total += len(chunk)
                    handle.write(chunk)
                async for chunk in chunks:
                    total += len(chunk)
                    if total > self.settings.MAX_RECORDING_BYTES:
                        raise YeastarAPIError(
                            "Recording is larger than the configured limit."
                        )
                    handle.write(chunk)
                handle.flush()
                os.fsync(handle.fileno())
        return RecordingDownload(
            path=destination,
            size_bytes=total,
            content_type=(
                content_type
                or mimetypes.guess_type(destination.name)[0]
                or "application/octet-stream"
            ),
        )

    async def stream_download(
        self,
        resource_path: str,
        *,
        recording_id: str,
        access_token: str,
        destination: Path,
    ) -> RecordingDownload:
        resource_path = validate_download_resource_path(resource_path)
        try:
            try:
                return await self._stream_download_once(
                    resource_path,
                    access_token=access_token,
                    destination=destination,
                )
            except YeastarTokenExpiredError:
                destination.unlink(missing_ok=True)
                await self.token_manager.invalidate_access_token(access_token)
                refreshed = await self.token_manager.refresh_access_token()
                try:
                    payload = await self.get_with_token(
                        "recording/download",
                        version="v1",
                        params={"id": recording_id},
                        access_token=refreshed,
                    )
                    supplied = payload.get("download_resource_url")
                    if not isinstance(supplied, str):
                        raise YeastarResponseError(
                            "Phone system did not provide a recording download."
                        )
                    refreshed_resource = validate_download_resource_path(supplied)
                    return await self._stream_download_once(
                        refreshed_resource,
                        access_token=refreshed,
                        destination=destination,
                    )
                except YeastarTokenExpiredError as exc:
                    await self.token_manager.open_circuit(
                        ConnectionState.TOKEN_REFRESH_FAILED,
                        last_errcode=10004,
                    )
                    raise YeastarTokenRefreshError(
                        "The phone-system session could not be renewed.", 10004
                    ) from exc
        except YeastarError as exc:
            await self._open_circuit_for_error(exc)
            destination.unlink(missing_ok=True)
            raise
        except Exception:
            destination.unlink(missing_ok=True)
            raise

    async def download_recording(
        self, recording_id: str, destination: Path
    ) -> dict[str, object]:
        result = await YeastarRecordings(
            self,
            self.redis,
            self.settings.STORAGE_ROOT,
            page_size=self.settings.yeastar_page_size,
            download_limit_retries=0,
        ).download(recording_id, destination)
        return {
            "path": result.path,
            "size_bytes": result.size_bytes,
            "content_type": result.content_type,
        }

    async def get_ai_transcript(self, call_leg_ids: list[str]) -> dict[str, object]:
        clean_ids = [item for item in call_leg_ids if item and len(item) <= 255]
        if not clean_ids:
            return {}
        offset = 1
        limit = 100
        merged: dict[str, object] = {}
        seen_offsets: set[int] = set()
        while True:
            if offset in seen_offsets:
                raise YeastarAPIError("Phone system returned invalid transcript pagination.")
            seen_offsets.add(offset)
            response = await self._authorized_get(
                "cdr/getaicontext",
                version="v2",
                params={"cdr_ids": ",".join(clean_ids), "offset": offset, "limit": limit},
            )
            data = response.get("data")
            if not isinstance(data, dict):
                raise YeastarResponseError("Phone system returned invalid transcript data.")
            for leg_id, raw_leg_data in data.items():
                if not isinstance(raw_leg_data, dict):
                    merged[str(leg_id)] = raw_leg_data
                    continue
                existing = merged.get(str(leg_id))
                combined = dict(existing) if isinstance(existing, dict) else {}
                for key, value in raw_leg_data.items():
                    if key == "context" and isinstance(value, list):
                        current = combined.get(key)
                        combined[key] = (current if isinstance(current, list) else []) + value
                    else:
                        combined[key] = value
                merged[str(leg_id)] = combined
            try:
                next_offset = int(cast(int | str, response.get("offset", -1)))
            except (TypeError, ValueError) as exc:
                raise YeastarResponseError(
                    "Phone system returned invalid transcript pagination."
                ) from exc
            if next_offset == -1:
                return merged
            offset = next_offset
