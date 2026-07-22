from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from app.services.yeastar.errors import YeastarAPIError, YeastarOperationError
from app.services.yeastar.extensions import QueryValue, YeastarGetClient


CDRApiVersion = Literal["v1", "v2"]
_MAX_CDR_PAGES = 10_000


class _LegacyCDRPage(BaseModel):
    """The documented v1 list envelope before it is mapped to the v2-shaped model."""

    model_config = ConfigDict(extra="ignore")

    data: list[dict[str, object]] = Field(default_factory=list)
    total_number: int | None = Field(default=None, ge=0)


class CDRSummary(BaseModel):
    model_config = ConfigDict(extra="allow")

    uid: str
    id: str | int | None = None
    time: str | int | float | None = None
    call_type: str | None = None
    call_from: str | None = None
    call_to: str | None = None
    call_from_number: str | None = None
    call_to_number: str | None = None
    last_status: str | None = None
    call_duration: int = 0
    recording_type: int | str | None = None

    def provider_dict(self) -> dict[str, object]:
        return self.model_dump(exclude_unset=True)

    @classmethod
    def from_legacy(cls, value: Mapping[str, object]) -> "CDRSummary":
        """Map a CDR 1.0 row to the fields used by the analyzer.

        CDR 1.0 names the status and duration fields ``disposition`` and
        ``duration``.  It also provides a Unix timestamp, which is preferable
        to the PBX-rendered ``time`` string because legacy devices can expose
        date/time display patterns the application cannot parse safely.
        """
        candidate = dict(value)
        uid = _text(candidate.get("uid"))
        if not uid:
            raise ValueError("Legacy CDR is missing its UID")
        candidate["uid"] = uid

        if candidate.get("id") in (None, "") and candidate.get("new_id") not in (
            None,
            "",
        ):
            candidate["id"] = candidate["new_id"]
        timestamp = _legacy_timestamp(candidate.get("timestamp"))
        if timestamp is not None:
            candidate["timestamp"] = timestamp
            candidate["time"] = timestamp
        if candidate.get("last_status") in (None, ""):
            candidate["last_status"] = candidate.get("disposition")
        if candidate.get("call_duration") in (None, ""):
            candidate["call_duration"] = _legacy_duration(candidate.get("duration"))
        candidate["legacy_cdr"] = True
        return cls.model_validate(candidate)


class CDRSearchPage(BaseModel):
    model_config = ConfigDict(extra="ignore")

    data: list[CDRSummary] = Field(default_factory=list)
    total_number: int = Field(default=0, ge=0)


class CDRTimelineEntry(BaseModel):
    model_config = ConfigDict(extra="allow")

    leg: str | int | None = None
    transaction_id: str | int | None = None
    cdr_id: str | int | None = None
    time: str | int | float | None = None
    call_type: str | None = None
    call_from: str | None = None
    call_to: str | None = None
    call_from_number: str | None = None
    call_to_number: str | None = None
    call_from_ext_id: str | int | None = None
    call_to_ext_id: str | int | None = None
    status: str | None = None
    call_duration: int = 0
    ring_duration: int = 0
    talk_duration: int = 0
    hold_duration: int = 0
    event_list: list[dict[str, object]] = Field(default_factory=list)
    recording_id: str | int | None = None
    answer_time: str | int | float | None = None
    end_time: str | int | float | None = None

    @field_validator("event_list", mode="before")
    @classmethod
    def normalize_events(cls, value: object) -> object:
        return value if isinstance(value, list) else []

    def persistence_mapping(self, sequence_number: int) -> dict[str, object]:
        transaction_id = _text(self.transaction_id)
        cdr_id = _text(self.cdr_id)
        stable_id = ":".join(part for part in (transaction_id, cdr_id) if part)
        if not stable_id:
            stable_id = f"timeline-{sequence_number}"
        return {
            "yeastar_leg_id": stable_id,
            "transaction_id": transaction_id or None,
            "yeastar_cdr_id": cdr_id or None,
            "provider_leg": _text(self.leg) or None,
            "sequence_number": sequence_number,
            "caller_extension_id": _text(self.call_from_ext_id) or None,
            "callee_extension_id": _text(self.call_to_ext_id) or None,
            "call_from": self.call_from,
            "call_to": self.call_to,
            "caller_number": self.call_from_number,
            "callee_number": self.call_to_number,
            "duration_seconds": max(0, self.call_duration),
            "ring_duration_seconds": max(0, self.ring_duration),
            "talk_duration_seconds": max(0, self.talk_duration),
            "hold_duration_seconds": max(0, self.hold_duration),
            "call_type": self.call_type,
            "status": self.status,
            "event_list": self.event_list,
            "provider_recording_id": _text(self.recording_id) or None,
            "provider_start_time": self.time,
            "provider_answer_time": self.answer_time,
            "provider_end_time": self.end_time,
            "provider_payload": self.model_dump(),
        }


class CDRBasic(BaseModel):
    model_config = ConfigDict(extra="allow")

    uid: str | None = None


class CDRDetail(BaseModel):
    model_config = ConfigDict(extra="ignore")

    basic: CDRBasic = Field(default_factory=CDRBasic)
    timeline: list[CDRTimelineEntry] = Field(default_factory=list)

    def provider_dict(self) -> dict[str, object]:
        return {
            "basic": self.basic.model_dump(),
            "timeline": [entry.model_dump() for entry in self.timeline],
        }


def _text(value: object) -> str:
    return "" if value is None else str(value).strip()


def _legacy_timestamp(value: object) -> int | None:
    """Return a documented CDR 1.0 Unix timestamp, without guessing text dates."""
    if value in (None, "") or isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value) if value.is_integer() else None
    text = _text(value)
    if not text or (text[0] in "+-" and not text[1:].isdigit()) or (
        text[0] not in "+-" and not text.isdigit()
    ):
        return None
    try:
        return int(text)
    except ValueError:
        return None


def _legacy_epoch_bound(value: str | int | float, field: str) -> int:
    timestamp = _legacy_timestamp(value)
    if timestamp is None:
        raise ValueError(f"Legacy CDR {field} must be a Unix timestamp in seconds")
    return timestamp


def _legacy_duration(value: object) -> int:
    """Normalize documented integer duration fields without allowing negatives."""
    if value in (None, "") or isinstance(value, bool):
        return 0
    try:
        parsed = int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return 0
    return max(0, parsed)


class YeastarCDR:
    def __init__(
        self,
        client: YeastarGetClient,
        page_size: int = 500,
        *,
        api_version: CDRApiVersion = "v2",
    ) -> None:
        if not 1 <= page_size <= 10_000:
            raise ValueError("page_size must be between 1 and 10000")
        if api_version not in {"v1", "v2"}:
            raise ValueError("CDR API version must be v1 or v2")
        self.client = client
        self.page_size = page_size
        self.api_version = api_version

    async def page(
        self,
        *,
        page: int,
        time_begin: str | int | float | None = None,
        time_end: str | int | float | None = None,
        filters: Mapping[str, QueryValue] | None = None,
        page_size: int | None = None,
        order_by: str = "asc",
    ) -> CDRSearchPage:
        requested_size = page_size or self.page_size
        if page < 1 or not 1 <= requested_size <= 10_000:
            raise ValueError("Invalid CDR pagination")
        if self.api_version == "v1":
            if time_begin is not None or time_end is not None or filters:
                raise ValueError(
                    "Legacy CDR pagination does not support filters; use search_all."
                )
            return await self.legacy_page(
                page=page,
                page_size=requested_size,
                order_by=order_by,
            )
        allowed = {
            "call_from",
            "call_to",
            "call_type",
            "last_status",
            "queue_list",
            "recording_type",
            "uid",
        }
        params: dict[str, QueryValue] = {
            key: value
            for key, value in (filters or {}).items()
            if key in allowed and value not in (None, "")
        }
        params.update(
            {
                "page": page,
                "page_size": requested_size,
                "sort_by": "time",
                "order_by": "desc" if order_by == "desc" else "asc",
            }
        )
        if time_begin is not None:
            params["time_begin"] = time_begin
        if time_end is not None:
            params["time_end"] = time_end
        payload = await self.client.get("cdr/search", version="v2", params=params)
        try:
            result = CDRSearchPage.model_validate(payload)
        except ValueError as exc:
            raise YeastarAPIError("Phone system returned an invalid call list.") from exc
        if result.total_number == 0 and result.data:
            result.total_number = len(result.data)
        return result

    async def search_all(
        self,
        *,
        time_begin: str | int | float,
        time_end: str | int | float,
        filters: Mapping[str, QueryValue] | None = None,
    ) -> list[CDRSummary]:
        if self.api_version == "v1":
            return await self.legacy_search_all(
                time_begin=time_begin,
                time_end=time_end,
                filters=filters,
            )
        records: list[CDRSummary] = []
        page_number = 1
        while True:
            page = await self.page(
                page=page_number,
                time_begin=time_begin,
                time_end=time_end,
                filters=filters,
            )
            records.extend(page.data)
            total = page.total_number or len(records)
            if len(records) >= total:
                return records
            if not page.data or page_number >= _MAX_CDR_PAGES:
                raise YeastarAPIError("Phone system returned invalid call pagination.")
            page_number += 1

    async def verify_read_access(self) -> None:
        if self.api_version == "v1":
            await self.legacy_page(page=1, page_size=1, order_by="desc")
            return
        await self.page(page=1, page_size=1, order_by="desc")

    async def detail(
        self,
        uid: str,
        *,
        summary: CDRSummary | Mapping[str, object] | None = None,
        summaries: Sequence[CDRSummary | Mapping[str, object]] | None = None,
    ) -> CDRDetail:
        if not uid or len(uid) > 255:
            raise ValueError("Invalid CDR UID")
        if self.api_version == "v1":
            supplied: list[CDRSummary | Mapping[str, object]]
            if summaries is not None:
                if summary is not None:
                    raise ValueError("Supply either summary or summaries, not both")
                supplied = list(summaries)
            elif summary is not None:
                supplied = [summary]
            else:
                raise ValueError("Legacy CDR details require the listed CDR summaries")
            normalized = [self._legacy_summary(value) for value in supplied]
            if any(item.uid != uid for item in normalized):
                raise ValueError("Legacy CDR detail summaries must have the requested UID")
            return self.detail_from_summaries(normalized)
        payload = await self.client.get(
            "cdr/detail", version="v2", params={"uid": uid}
        )
        raw = payload.get("data")
        if not isinstance(raw, Mapping):
            raise YeastarAPIError("Phone system returned invalid call details.")
        # Official responses use data.basic + data.timeline. Accepting a direct
        # timeline keeps older saved doubles readable without flattening basic.
        candidate = dict(raw)
        if "basic" not in candidate:
            candidate["basic"] = {}
        try:
            return CDRDetail.model_validate(candidate)
        except ValueError as exc:
            raise YeastarAPIError("Phone system returned invalid call details.") from exc

    async def legacy_page(
        self,
        *,
        page: int,
        page_size: int | None = None,
        order_by: str = "desc",
    ) -> CDRSearchPage:
        """Retrieve one raw CDR 1.0 list page using only documented parameters.

        This method intentionally has no time or call filters.  CDR 1.0 list
        does not document the v2 query parameters, so legacy range filtering is
        handled locally by :meth:`legacy_search_all`.
        """
        result, _ = await self._legacy_page(
            page=page,
            page_size=page_size,
            order_by=order_by,
        )
        return result

    async def _legacy_page(
        self,
        *,
        page: int,
        page_size: int | None = None,
        order_by: str = "desc",
    ) -> tuple[CDRSearchPage, int | None]:
        requested_size = page_size or self.page_size
        if page < 1 or not 1 <= requested_size <= 10_000:
            raise ValueError("Invalid CDR pagination")
        payload = await self.client.get(
            "cdr/list",
            version="v1",
            params={
                "page": page,
                "page_size": requested_size,
                "sort_by": "time",
                "order_by": "desc" if order_by == "desc" else "asc",
            },
        )
        try:
            raw_page = _LegacyCDRPage.model_validate(payload)
            records = [CDRSummary.from_legacy(item) for item in raw_page.data]
        except ValueError as exc:
            raise YeastarAPIError("Phone system returned an invalid legacy call list.") from exc

        # A few older PBXs omit total_number.  Preserve a reported, coherent
        # total for pagination decisions while exposing the usual page model.
        reported_total = raw_page.total_number
        if reported_total is not None and reported_total < len(records):
            reported_total = None
        total_number = reported_total if reported_total is not None else len(records)
        return CDRSearchPage(data=records, total_number=total_number), reported_total

    async def legacy_search_all(
        self,
        *,
        time_begin: str | int | float,
        time_end: str | int | float,
        filters: Mapping[str, QueryValue] | None = None,
    ) -> list[CDRSummary]:
        """Read and locally filter legacy CDRs by documented Unix timestamps.

        CDR 1.0 list data is requested newest-first and only with documented
        list parameters.  The PBX-provided Unix ``timestamp`` avoids relying on
        its display-time format.  Filtering is evaluated at call (UID) scope so
        a matching transferred-call leg does not discard its related legs.
        """
        start_time = _legacy_epoch_bound(time_begin, "time_begin")
        end_time = _legacy_epoch_bound(time_end, "time_end")
        if start_time > end_time:
            raise ValueError("Legacy CDR time_begin cannot be after time_end")
        normalized_filters = self._legacy_filters(filters)

        grouped: dict[str, list[CDRSummary]] = {}
        page_number = 1
        records_seen = 0
        while True:
            page, reported_total = await self._legacy_page(
                page=page_number,
                order_by="desc",
            )
            raw_records = page.data
            if not raw_records:
                break
            records_seen += len(raw_records)

            timestamps: list[int] = []
            missing_timestamp = False
            for record in raw_records:
                timestamp = _legacy_timestamp(record.time)
                if timestamp is None:
                    missing_timestamp = True
                    continue
                timestamps.append(timestamp)
                if start_time <= timestamp <= end_time:
                    grouped.setdefault(record.uid, []).append(record)

            # ``sort_by=time&order_by=desc`` is documented for cdr/list.  We
            # only early-stop when every row carries a timestamp and is older
            # than the requested range; otherwise scan to a normal page end.
            if timestamps and not missing_timestamp and max(timestamps) < start_time:
                break
            if reported_total is not None and records_seen >= reported_total:
                break
            if len(raw_records) < self.page_size:
                break
            if page_number >= _MAX_CDR_PAGES:
                raise YeastarAPIError("Phone system returned invalid legacy call pagination.")
            page_number += 1

        result: list[CDRSummary] = []
        for summaries in grouped.values():
            if any(self._legacy_matches_filters(item, normalized_filters) for item in summaries):
                result.extend(summaries)
        return result

    @staticmethod
    def _legacy_summary(value: CDRSummary | Mapping[str, object]) -> CDRSummary:
        if isinstance(value, CDRSummary):
            return value
        try:
            return CDRSummary.from_legacy(value)
        except ValueError as exc:
            raise YeastarAPIError("Phone system returned an invalid legacy call list.") from exc

    @staticmethod
    def _legacy_filters(
        filters: Mapping[str, QueryValue] | None,
    ) -> dict[str, QueryValue]:
        normalized: dict[str, QueryValue] = {}
        aliases = {"status": "last_status"}
        allowed = {
            "call_from",
            "call_to",
            "call_type",
            "last_status",
            "recording_type",
            "uid",
        }
        for key, value in (filters or {}).items():
            if value in (None, ""):
                continue
            canonical = aliases.get(key, key)
            if canonical in {"queue", "queue_list"}:
                raise YeastarOperationError("Legacy CDR does not support queue filtering")
            if canonical not in allowed:
                raise YeastarOperationError(
                    f"Legacy CDR does not support the {key} filter"
                )
            normalized[canonical] = value
        return normalized

    @staticmethod
    def _legacy_matches_filters(
        record: CDRSummary,
        filters: Mapping[str, QueryValue],
    ) -> bool:
        raw = record.provider_dict()
        for key, expected in filters.items():
            if key == "call_from":
                candidates = (raw.get("call_from"), raw.get("call_from_number"))
            elif key == "call_to":
                candidates = (raw.get("call_to"), raw.get("call_to_number"))
            elif key == "call_type":
                candidates = (raw.get("call_type"),)
            elif key == "last_status":
                candidates = (raw.get("last_status"), raw.get("disposition"))
            elif key == "uid":
                candidates = (record.uid,)
            elif key == "recording_type":
                has_recording = bool(_text(raw.get("record_file")))
                try:
                    wanted = int(expected)
                except (TypeError, ValueError):
                    return False
                if wanted not in {1, 2} or has_recording != (wanted == 1):
                    return False
                continue
            else:  # Defensive guard for future changes to _legacy_filters.
                return False
            expected_text = _text(expected).casefold()
            if not any(_text(value).casefold() == expected_text for value in candidates):
                return False
        return True

    @staticmethod
    def detail_from_summary(summary: CDRSummary) -> CDRDetail:
        return YeastarCDR.detail_from_summaries([summary])

    @staticmethod
    def detail_from_summaries(summaries: Sequence[CDRSummary]) -> CDRDetail:
        """Create a conservative one-entry-per-v1-row timeline for one call."""
        if not summaries:
            raise ValueError("Legacy CDR details require at least one summary")
        uid = summaries[0].uid
        if not uid or any(item.uid != uid for item in summaries):
            raise ValueError("Legacy CDR detail summaries must share a non-empty UID")

        def sort_key(item: CDRSummary) -> int:
            return _legacy_timestamp(item.time) or 0

        timeline: list[CDRTimelineEntry] = []
        for position, summary in enumerate(sorted(summaries, key=sort_key), start=1):
            raw = summary.provider_dict()
            cdr_id = _text(raw.get("id") or raw.get("new_id")) or f"{uid}:{position}"
            timeline.append(
                CDRTimelineEntry(
                    transaction_id=_text(raw.get("call_id")) or None,
                    cdr_id=cdr_id,
                    time=summary.time,
                    call_type=summary.call_type,
                    call_from=summary.call_from,
                    call_to=summary.call_to,
                    call_from_number=summary.call_from_number,
                    call_to_number=summary.call_to_number,
                    status=summary.last_status,
                    call_duration=summary.call_duration,
                    ring_duration=_legacy_duration(raw.get("ring_duration")),
                    talk_duration=_legacy_duration(raw.get("talk_duration")),
                    legacy_cdr=True,
                    record_file=raw.get("record_file"),
                    provider_payload=raw,
                )
            )
        return CDRDetail(
            basic=CDRBasic(uid=uid, legacy_cdr=True),
            timeline=timeline,
        )
