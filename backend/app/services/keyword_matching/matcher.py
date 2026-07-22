from __future__ import annotations

import re
from dataclasses import dataclass

from rapidfuzz.fuzz import ratio

from app.models.enums import MatchMethod
from app.services.keyword_matching.normalization import normalize_greek, normalize_greek_with_map


@dataclass(frozen=True)
class KeywordDefinition:
    id: str
    phrase: str
    variants: tuple[str, ...] = ()
    accent_insensitive: bool = True
    whole_word: bool = True
    exact_phrase: bool = True
    fuzzy_match: bool = False
    fuzzy_threshold: int = 90


@dataclass(frozen=True)
class SegmentMatch:
    keyword_id: str
    original_matched_text: str
    normalized_match: str
    context_before: str
    context_after: str
    method: MatchMethod
    score: float
    normalized_start: int
    normalized_end: int


def _context(text: str, start: int, end: int, radius: int = 100) -> tuple[str, str]:
    return text[max(0, start - radius) : start].strip(), text[end : end + radius].strip()


def _exact_matches(
    normalized_text: str,
    original_text: str,
    phrase: str,
    definition: KeywordDefinition,
    source_map: list[int],
    *,
    variant: bool,
) -> list[SegmentMatch]:
    normalized_phrase = normalize_greek(phrase, remove_accents=definition.accent_insensitive)
    if not normalized_phrase:
        return []
    boundary_left = r"(?<!\w)" if definition.whole_word else ""
    boundary_right = r"(?!\w)" if definition.whole_word else ""
    if definition.exact_phrase or " " not in normalized_phrase:
        expression = re.escape(normalized_phrase)
        method = MatchMethod.EXACT_PHRASE if " " in normalized_phrase else MatchMethod.WHOLE_WORD
    else:
        tokens = normalized_phrase.split()
        # Optional mode: preserve term order while allowing up to three intervening words.
        expression = r"(?:\s+\w+){0,3}\s+".join(re.escape(token) for token in tokens)
        method = MatchMethod.ORDERED_TERMS
    pattern = re.compile(boundary_left + expression + boundary_right, re.UNICODE)
    results: list[SegmentMatch] = []
    for found in pattern.finditer(normalized_text):
        original_start = source_map[found.start()]
        original_end = source_map[found.end() - 1] + 1
        before, after = _context(original_text, original_start, original_end)
        results.append(
            SegmentMatch(
                definition.id,
                original_text[original_start:original_end],
                found.group(0),
                before,
                after,
                MatchMethod.VARIANT if variant else method,
                100.0,
                found.start(),
                found.end(),
            )
        )
    return results


def _fuzzy_matches(
    normalized_text: str,
    original_text: str,
    source_map: list[int],
    definition: KeywordDefinition,
    phrase: str,
) -> list[SegmentMatch]:
    phrase_tokens = normalize_greek(
        phrase, remove_accents=definition.accent_insensitive
    ).split()
    text_tokens = list(re.finditer(r"\w+", normalized_text, re.UNICODE))
    if not phrase_tokens or len(text_tokens) < len(phrase_tokens):
        return []
    results: list[SegmentMatch] = []
    target = " ".join(phrase_tokens)
    for index in range(len(text_tokens) - len(phrase_tokens) + 1):
        window_tokens = text_tokens[index : index + len(phrase_tokens)]
        start, end = window_tokens[0].start(), window_tokens[-1].end()
        candidate = normalized_text[start:end]
        score = float(ratio(target, candidate))
        if score < definition.fuzzy_threshold:
            continue
        original_start = source_map[start]
        original_end = source_map[end - 1] + 1
        before, after = _context(original_text, original_start, original_end)
        results.append(
            SegmentMatch(
                definition.id,
                original_text[original_start:original_end],
                candidate,
                before,
                after,
                MatchMethod.FUZZY,
                score,
                start,
                end,
            )
        )
    return results


def match_text(original_text: str, definitions: list[KeywordDefinition]) -> list[SegmentMatch]:
    all_matches: list[SegmentMatch] = []
    seen: set[tuple[str, int, int]] = set()
    for definition in definitions:
        normalized, source_map = normalize_greek_with_map(
            original_text, remove_accents=definition.accent_insensitive
        )
        phrases = ((definition.phrase, False),) + tuple(
            (variant, True) for variant in definition.variants
        )
        for phrase, is_variant in phrases:
            matches = _exact_matches(
                normalized,
                original_text,
                phrase,
                definition,
                source_map,
                variant=is_variant,
            )
            if definition.fuzzy_match:
                matches += _fuzzy_matches(normalized, original_text, source_map, definition, phrase)
            for match in matches:
                key = (match.keyword_id, match.normalized_start, match.normalized_end)
                if key not in seen:
                    seen.add(key)
                    all_matches.append(match)
    return sorted(all_matches, key=lambda item: (item.normalized_start, item.keyword_id))
