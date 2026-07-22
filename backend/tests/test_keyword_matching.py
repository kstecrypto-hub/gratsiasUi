from __future__ import annotations

from app.models.enums import MatchMethod
from app.services.keyword_matching.matcher import KeywordDefinition, match_text
from app.services.keyword_matching.normalization import normalize_greek


def _keyword(phrase: str, **overrides: object) -> KeywordDefinition:
    return KeywordDefinition(id="keyword-1", phrase=phrase, **overrides)


def test_greek_normalization_removes_accents_and_normalizes_final_sigma() -> None:
    assert normalize_greek("  ΤΈΛΟΣ,\tκόσμος!  ") == "τελοσ κοσμοσ"


def test_greek_normalization_can_preserve_accents() -> None:
    assert normalize_greek("Τόνος", remove_accents=False) == "τόνοσ"


def test_accent_insensitive_matching_preserves_original_text() -> None:
    matches = match_text("Η ΑΣΦΑΛΕΙΑ είναι ενεργή.", [_keyword("ασφάλεια")])

    assert len(matches) == 1
    assert matches[0].original_matched_text == "ΑΣΦΑΛΕΙΑ"
    assert matches[0].normalized_match == "ασφαλεια"


def test_accent_sensitive_matching_requires_the_same_accents() -> None:
    definition = _keyword("ασφάλεια", accent_insensitive=False)

    assert match_text("Η ασφαλεια είναι ενεργή.", [definition]) == []
    assert len(match_text("Η ασφάλεια είναι ενεργή.", [definition])) == 1


def test_exact_phrase_does_not_allow_intervening_words() -> None:
    definition = _keyword("επιστροφή χρημάτων", exact_phrase=True)

    assert len(match_text("Ζητώ επιστροφή χρημάτων τώρα.", [definition])) == 1
    assert match_text("Ζητώ επιστροφή των χρημάτων τώρα.", [definition]) == []


def test_non_exact_phrase_allows_ordered_terms_with_a_small_gap() -> None:
    definition = _keyword("επιστροφή χρημάτων", exact_phrase=False)

    matches = match_text("Ζητώ επιστροφή των χρημάτων τώρα.", [definition])

    assert len(matches) == 1
    assert matches[0].method == MatchMethod.ORDERED_TERMS


def test_whole_word_matching_prevents_substring_false_positives() -> None:
    definition = _keyword("λόγο", whole_word=True)

    assert match_text("Δεν υπάρχει λόγος ακύρωσης.", [definition]) == []
    assert len(match_text("Δεν βρήκα λόγο ακύρωσης.", [definition])) == 1


def test_substring_matching_is_only_enabled_explicitly() -> None:
    definition = _keyword("λόγο", whole_word=False)

    assert len(match_text("Δεν υπάρχει λόγος ακύρωσης.", [definition])) == 1


def test_variant_match_is_not_duplicated_by_fuzzy_matching() -> None:
    definition = _keyword(
        "ακύρωση",
        variants=("ακυρωτικό",),
        fuzzy_match=True,
        fuzzy_threshold=80,
    )

    matches = match_text("Χρειάζομαι ακυρωτικό.", [definition])

    assert len(matches) == 1
    assert matches[0].method == MatchMethod.VARIANT
