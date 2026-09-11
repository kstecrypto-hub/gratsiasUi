"""Place continuous ASR text against rough turns without rewriting its words.

Text recognition and speaker timing are independent evidence. Unmatched text
stays in the transcript; ambiguous attribution stays unknown. Times are estimates
within diarization turns, never forced-alignment or word-confidence scores.
"""

from __future__ import annotations

from dataclasses import dataclass
from difflib import SequenceMatcher
import unicodedata

from app.services.transcription.mono import AnonymousDiarizationTurn


CONVERSATION_ALIGNMENT_VERSION = "mono-continuous-alignment-v2"
WORDING_REVIEW_VERSION = "raw-normalized-wording-review-v1"
MIN_RELIABLE_TURN_SECONDS = 0.35
SHORT_TURN_SECONDS = 1.0
MIN_SHORT_TURN_MATCH_RATIO = 0.6
MAX_ALIGNMENT_TOKEN_PRODUCT = 4_000_000
MIN_ALIGNMENT_MATCH_RATIO = 0.4


@dataclass(frozen=True, slots=True)
class ConversationSegment:
    text: str
    start_seconds: float
    end_seconds: float
    speaker_label: str


@dataclass(frozen=True, slots=True)
class ConversationAlignment:
    segments: tuple[ConversationSegment, ...]
    match_ratio: float
    uncertain_word_count: int
    word_count: int
    limit_exceeded: bool = False


def review_conversation_wording(
    segments: tuple[ConversationSegment, ...], alternative_text: str,
) -> dict[str, object]:
    """Locate disagreements, without choosing a spelling or inventing confidence."""

    words: list[str] = []
    owners: list[int] = []
    for index, segment in enumerate(segments):
        tokens = segment.text.split()
        words.extend(tokens)
        owners.extend([index] * len(tokens))
    alternative = alternative_text.split()
    if not words or not alternative:
        return {"version": WORDING_REVIEW_VERSION, "status": "unavailable", "items": []}
    if len(words) * len(alternative) > MAX_ALIGNMENT_TOKEN_PRODUCT:
        return {"version": WORDING_REVIEW_VERSION, "status": "limit_exceeded", "items": []}
    opcodes = SequenceMatcher(
        None, [_key(word) for word in words], [_key(word) for word in alternative],
        autojunk=False,
    ).get_opcodes()
    items: list[dict[str, object]] = []
    for operation, a1, a2, b1, b2 in opcodes:
        if operation == "equal":
            continue
        first = owners[min(a1, len(owners) - 1)]
        last = owners[max(a1, a2 - 1)] if a1 < len(owners) else first
        items.append({
            "segment_indexes": list(range(first, last + 1)),
            "start_seconds": segments[first].start_seconds,
            "end_seconds": max(segment.end_seconds for segment in segments[first:last + 1]),
            "original_text": " ".join(words[a1:a2]),
            "alternative_text": " ".join(alternative[b1:b2]),
        })
    return {
        "version": WORDING_REVIEW_VERSION,
        "status": "complete",
        "difference_count": len(items),
        "truncated": len(items) > 32,
        "items": items[:32],
    }


def _key(word: str) -> str:
    normalized = "".join(
        char for char in unicodedata.normalize("NFD", word.casefold())
        if char.isalnum()
    ) or word
    # Greek contraction equivalence is for matching only. It must not change
    # the recognized spelling or let a repeated closing word match an earlier reply.
    return "και" if normalized == "κι" else normalized


def align_conversation(
    text: str,
    turns: tuple[AnonymousDiarizationTurn, ...],
    *,
    recording_duration_seconds: float,
) -> ConversationAlignment:
    """Preserve every ASR token once, including genuine repeated words.

    Only monotonic text matches and substitutions inside a single reliable turn
    inherit a speaker. Interjections shorter than 350 ms cannot establish identity.
    A bounded alignment fails closed to unknown attribution on poor evidence.
    """

    words = text.split()
    if not words:
        return ConversationAlignment((), 0.0, 0, 0)
    # Tiny diarization fragments often repeat neighboring words. They must not
    # win matching anchors over the longer turn that actually contains a phrase.
    ordered = sorted(
        (turn for turn in turns if turn.duration_seconds >= MIN_RELIABLE_TURN_SECONDS),
        key=lambda turn: (turn.start_seconds, turn.end_seconds),
    )
    rough: list[str] = []
    # (turn identity, estimated token start, estimated token end)
    source: list[tuple[int, float, float]] = []
    for index, turn in enumerate(ordered):
        tokens = turn.rough_text.split()
        for position, token in enumerate(tokens):
            rough.append(_key(token))
            source.append((
                index,
                turn.start_seconds + turn.duration_seconds * position / len(tokens),
                turn.start_seconds + turn.duration_seconds * (position + 1) / len(tokens),
            ))
    limited = len(rough) * len(words) > MAX_ALIGNMENT_TOKEN_PRODUCT

    def unknown(ratio: float) -> ConversationAlignment:
        return ConversationAlignment(
            (ConversationSegment(" ".join(words), 0.0, recording_duration_seconds, "Unknown"),),
            ratio, len(words), len(words), limited,
        )

    if not rough or limited:
        return unknown(0.0)
    recognized = [_key(word) for word in words]
    matcher = SequenceMatcher(None, rough, recognized, autojunk=False)
    opcodes = matcher.get_opcodes()
    matched_by_turn: dict[int, int] = {}
    for operation, a1, a2, _, _ in opcodes:
        if operation == "equal":
            for index, _, _ in source[a1:a2]:
                matched_by_turn[index] = matched_by_turn.get(index, 0) + 1
    weak_turns = {
        index for index, turn in enumerate(ordered)
        if turn.duration_seconds < SHORT_TURN_SECONDS
        and matched_by_turn.get(index, 0) / len(turn.rough_text.split()) < MIN_SHORT_TURN_MATCH_RATIO
    }
    if weak_turns:
        retained = [(token, entry) for token, entry in zip(rough, source) if entry[0] not in weak_turns]
        if not retained:
            return unknown(0.0)
        rough = [token for token, _ in retained]
        source = [entry for _, entry in retained]
        opcodes = SequenceMatcher(None, rough, recognized, autojunk=False).get_opcodes()
    ratio = sum(a2 - a1 for op, a1, a2, _, _ in opcodes if op == "equal") / max(len(rough), len(words))
    if ratio < MIN_ALIGNMENT_MATCH_RATIO:
        return unknown(ratio)

    # (group identity, speaker, start, end), one entry per recognized token.
    placements: list[tuple[int, str, float, float]] = []

    def placement(index: int, start: float, end: float) -> tuple[int, str, float, float]:
        turn = ordered[index]
        label = turn.speaker_label if turn.duration_seconds >= MIN_RELIABLE_TURN_SECONDS else "Unknown"
        return index, label, start, end

    for opcode_index, (operation, a1, a2, b1, b2) in enumerate(opcodes):
        if operation == "delete":
            continue
        if operation == "equal":
            placements.extend(placement(*source[index]) for index in range(a1, a2))
            continue
        turn_index: int | None = None
        if a1 < a2:
            start, end = source[a1][1], source[a2 - 1][2]
            if source[a1][0] == source[a2 - 1][0]:
                turn_index = source[a1][0]
        else:
            start = source[a1 - 1][2] if a1 else 0.0
            end = source[a1][1] if a1 < len(source) else recording_duration_seconds
            if 0 < a1 < len(source) and source[a1 - 1][0] == source[a1][0]:
                turn_index = source[a1][0]
        end = max(start, end)
        for offset in range(b2 - b1):
            token_start = start + (end - start) * offset / (b2 - b1)
            token_end = start + (end - start) * (offset + 1) / (b2 - b1)
            placements.append(
                placement(turn_index, token_start, token_end)
                if turn_index is not None else
                (-opcode_index - 1, "Unknown", token_start, token_end)
            )

    segments: list[ConversationSegment] = []
    group_start = 0
    for position in range(1, len(words) + 1):
        if position < len(words) and placements[position][:2] == placements[group_start][:2]:
            continue
        segments.append(ConversationSegment(
            text=" ".join(words[group_start:position]),
            start_seconds=min(item[2] for item in placements[group_start:position]),
            end_seconds=max(item[3] for item in placements[group_start:position]),
            speaker_label=placements[group_start][1],
        ))
        group_start = position
    return ConversationAlignment(
        tuple(segments), ratio,
        sum(item[1] == "Unknown" for item in placements), len(words),
    )
