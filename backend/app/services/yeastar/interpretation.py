from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from app.models.enums import ParticipantRole, SpeakerSource


@dataclass(frozen=True)
class InterpretedParticipant:
    operator_id: str
    provider_extension_id: str
    extension_number: str
    leg_id: str
    role: ParticipantRole
    was_caller: bool
    was_callee: bool
    answered: bool
    source: SpeakerSource = SpeakerSource.YEASTAR_EXTENSION


def _text(value: object) -> str:
    return "" if value is None else str(value)


def interpret_call_legs(
    detail: dict[str, Any], operators: list[dict[str, Any]]
) -> tuple[list[dict[str, Any]], list[InterpretedParticipant]]:
    """Interpret explicit Yeastar timeline identifiers without name-based guessing."""
    by_id = {_text(operator.get("yeastar_extension_id")): operator for operator in operators}
    number_candidates: dict[str, list[dict[str, Any]]] = {}
    for operator in operators:
        number_candidates.setdefault(_text(operator.get("extension_number")), []).append(operator)
    by_number = {
        number: candidates[0]
        for number, candidates in number_candidates.items()
        if number and len(candidates) == 1
    }
    timeline = detail.get("timeline") or detail.get("transactions") or []
    if not isinstance(timeline, list):
        timeline = []
    legs: list[dict[str, Any]] = []
    participants: list[InterpretedParticipant] = []
    seen: set[tuple[str, str, str]] = set()
    for sequence, raw in enumerate(timeline, start=1):
        if not isinstance(raw, dict):
            continue
        transaction_id = _text(raw.get("transaction_id"))
        cdr_id = _text(raw.get("cdr_id"))
        leg_id = ":".join(value for value in (transaction_id, cdr_id) if value)
        if not leg_id:
            leg_id = _text(raw.get("id") or f"timeline-{sequence}")
        from_id = _text(raw.get("call_from_ext_id") or raw.get("from_ext_id"))
        to_id = _text(raw.get("call_to_ext_id") or raw.get("to_ext_id"))
        from_number = _text(raw.get("call_from_number") or raw.get("from_number"))
        to_number = _text(raw.get("call_to_number") or raw.get("to_number"))
        answered_id = _text(
            raw.get("answered_by_ext_id")
            or raw.get("answer_ext_id")
            or (
                raw.get("call_to_ext_id")
                if str(raw.get("status", "")).upper() == "ANSWERED"
                else ""
            )
        )
        legs.append(
            {
                "yeastar_leg_id": leg_id,
                "transaction_id": transaction_id or None,
                "yeastar_cdr_id": cdr_id or None,
                "provider_leg": _text(raw.get("leg")) or None,
                "sequence_number": sequence,
                "caller_extension_id": from_id or None,
                "callee_extension_id": to_id or None,
                "call_from": _text(raw.get("call_from")) or None,
                "call_to": _text(raw.get("call_to")) or None,
                "caller_number": from_number or None,
                "callee_number": to_number or None,
                "answered_by_extension_id": answered_id or None,
                "duration_seconds": int(raw.get("call_duration") or raw.get("duration") or 0),
                "ring_duration_seconds": int(raw.get("ring_duration") or 0),
                "talk_duration_seconds": int(raw.get("talk_duration") or 0),
                "hold_duration_seconds": int(raw.get("hold_duration") or 0),
                "call_type": _text(raw.get("call_type")) or None,
                "status": _text(raw.get("status")) or None,
                "event_list": raw.get("event_list") if isinstance(raw.get("event_list"), list) else [],
                "provider_recording_id": _text(raw.get("recording_id")) or None,
                "provider_payload": raw,
                "provider_start_time": raw.get("start_time") or raw.get("time"),
                "provider_answer_time": raw.get("answer_time"),
                "provider_end_time": raw.get("end_time"),
            }
        )
        explicit = [
            (from_id, from_number, ParticipantRole.CALLER, True, False),
            (
                to_id,
                to_number,
                ParticipantRole.ANSWERING_OPERATOR if answered_id and answered_id == to_id else ParticipantRole.CALLEE,
                False,
                True,
            ),
        ]
        if answered_id and answered_id not in {from_id, to_id}:
            explicit.append((answered_id, "", ParticipantRole.ANSWERING_OPERATOR, False, True))
        for extension_id, number, role, was_caller, was_callee in explicit:
            operator = by_id.get(extension_id) if extension_id else None
            if operator is None and number:
                operator = by_number.get(number)
            if operator is None:
                continue
            key = (_text(operator["id"]), leg_id, role.value)
            if key in seen:
                continue
            seen.add(key)
            participants.append(
                InterpretedParticipant(
                    operator_id=_text(operator["id"]),
                    provider_extension_id=extension_id or _text(operator.get("yeastar_extension_id")),
                    extension_number=number or _text(operator.get("extension_number")),
                    leg_id=leg_id,
                    role=role,
                    was_caller=was_caller,
                    was_callee=was_callee,
                    answered=(answered_id == extension_id) or role == ParticipantRole.ANSWERING_OPERATOR,
                )
            )
    return legs, participants


def safe_operator_channel(
    participant: InterpretedParticipant,
    channel_count: int,
    stereo_separated: bool,
    *,
    one_to_one: bool = False,
    was_transferred: bool = False,
    operators_on_same_side: int = 1,
) -> int | None:
    """Return a zero-based channel only for unambiguous, explicitly separated stereo."""
    if (
        channel_count != 2
        or not stereo_separated
        or not one_to_one
        or was_transferred
        or operators_on_same_side != 1
    ):
        return None
    if participant.was_caller and not participant.was_callee:
        return 0
    if participant.was_callee and not participant.was_caller:
        return 1
    return None


def safe_caller_callee_channels(
    channel_count: int,
    stereo_separated: bool,
    *,
    one_to_one: bool = False,
    was_transferred: bool = False,
) -> tuple[int, int] | None:
    """Return Yeastar's caller/callee mapping only for a proven one-to-one recording."""

    if (
        channel_count != 2
        or not stereo_separated
        or not one_to_one
        or was_transferred
    ):
        return None
    return (0, 1)
