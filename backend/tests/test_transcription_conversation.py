from __future__ import annotations

import pytest

from app.services.transcription.conversation import align_conversation
from app.services.transcription.mono import AnonymousDiarizationTurn


def turn(start: float, end: float, text: str, speaker: str = "A") -> AnonymousDiarizationTurn:
    return AnonymousDiarizationTurn(speaker, start, end, text, 100.0)


def test_continuous_text_is_not_duplicated_by_spurious_interruption_turns() -> None:
    text = "Ναι, δεν ξέρω. Είναι μέσα στο service."
    result = align_conversation(text, (
        turn(1, 3, "Ναι δεν ξέρω"),
        turn(3, 3.1, "δεν ξέρω", "B"),
        turn(3.1, 3.3, "δεν ξέρω"),
        turn(3.3, 6, "Είναι μέσα στο service", "B"),
    ), recording_duration_seconds=10)
    assert " ".join(segment.text for segment in result.segments) == text
    assert [segment.speaker_label for segment in result.segments] == ["A", "B"]
    assert result.uncertain_word_count == 0


def test_real_repeated_words_across_speakers_are_preserved() -> None:
    result = align_conversation("Ναι, ναι. Ναι, ναι.", (
        turn(1, 2, "ναι ναι"), turn(2, 3, "ναι ναι", "B"),
    ), recording_duration_seconds=5)
    assert [segment.text for segment in result.segments] == ["Ναι, ναι.", "Ναι, ναι."]
    assert [segment.speaker_label for segment in result.segments] == ["A", "B"]


def test_greek_spelling_and_numbers_keep_the_recognizers_output() -> None:
    text = "Πείτε μου την πινακίδα. ΧΤ 4179."
    result = align_conversation(text, (
        turn(1, 3, "πείτε μου την πεινακίδα"),
        turn(3, 5, "ΧΤ σαράντα ένα εβδομήντα εννέα", "B"),
    ), recording_duration_seconds=7)
    assert " ".join(segment.text for segment in result.segments) == text
    assert result.segments[-1].speaker_label == "B"
    assert result.segments[-1].end_seconds == 5


def test_words_missing_from_diarization_are_retained_as_unknown() -> None:
    result = align_conversation("Θα σας καλέσει. Εντάξει, ευχαριστώ. Γεια σας.", (
        turn(1, 3, "Θα σας καλέσει", "B"), turn(6, 7, "Γεια σας", "B"),
    ), recording_duration_seconds=8)
    unknown = result.segments[1]
    assert (unknown.text, unknown.speaker_label) == ("Εντάξει, ευχαριστώ.", "Unknown")
    assert (unknown.start_seconds, unknown.end_seconds) == (3, 6)
    assert result.uncertain_word_count == 2


def test_subsecond_false_turn_does_not_establish_speaker_identity() -> None:
    result = align_conversation("Γεια σας. Ναι. Θα σας καλέσω.", (
        turn(1, 2, "Γεια σας"), turn(2, 2.1, "Ναι", "B"),
        turn(2.1, 4, "Θα σας καλέσω"),
    ), recording_duration_seconds=5)
    assert result.segments[1].speaker_label == "Unknown"
    assert result.segments[1].text == "Ναι."


def test_replacement_across_two_speakers_is_ambiguous() -> None:
    text = "Καλημέρα σας. Άγνωστο όνομα. Να σας καλέσω αύριο;"
    result = align_conversation(text, (
        turn(0, 2, "Καλημέρα σας λάθος"), turn(2, 3, "επώνυμο", "B"),
        turn(3, 6, "Να σας καλέσω αύριο"),
    ), recording_duration_seconds=8)
    assert result.segments[1].text == "Άγνωστο όνομα."
    assert result.segments[1].speaker_label == "Unknown"


def test_weak_half_second_turn_cannot_capture_the_next_speakers_words() -> None:
    result = align_conversation("Ναι, δεν ξέρω. Είναι μέσα στο service τώρα.", (
        turn(1, 3, "Ναι δεν ξέρω"),
        turn(3, 3.2, "Είναι", "B"),
        turn(3.2, 3.7, "εσύ μέσα επειδή"),
        turn(3.7, 6, "στο service τώρα", "B"),
    ), recording_duration_seconds=8)
    assert result.segments[1].text == "Είναι μέσα"
    assert result.segments[1].speaker_label == "Unknown"


def test_repeated_closing_word_does_not_absorb_a_missing_callers_reply() -> None:
    result = align_conversation("Θα σας καλέσει. Εντάξει, ευχαριστώ. Ευχαριστώ και εγώ, γεια σας.", (
        turn(1, 3, "Θα σας καλέσει", "B"),
        turn(6, 7, "Ευχαριστώ κι εγώ γεια σας", "B"),
    ), recording_duration_seconds=8)
    assert result.segments[1].text == "Εντάξει, ευχαριστώ."
    assert result.segments[1].speaker_label == "Unknown"
    assert result.segments[2].text == "Ευχαριστώ και εγώ, γεια σας."


@pytest.mark.parametrize("turns", [(), (turn(1, 3, "unrelated source"),)])
def test_missing_or_unrelated_turns_preserve_the_entire_text(turns: tuple) -> None:
    result = align_conversation("Καλημέρα σας, θέλω ραντεβού.", turns, recording_duration_seconds=8)
    assert len(result.segments) == 1
    assert result.segments[0].text == "Καλημέρα σας, θέλω ραντεβού."
    assert result.segments[0].speaker_label == "Unknown"
    assert result.uncertain_word_count == 4


def test_alignment_work_is_bounded_for_repetitive_long_recordings() -> None:
    text = "ναι " * 2100
    result = align_conversation(text, (turn(0, 100, text),), recording_duration_seconds=100)
    assert result.limit_exceeded
    assert result.word_count == 2100
    assert result.segments[0].text == text.strip()
    assert result.segments[0].speaker_label == "Unknown"


@pytest.mark.parametrize("text", ["", "\n\t "])
def test_empty_text_has_no_fabricated_segments(text: str) -> None:
    result = align_conversation(text, (turn(1, 3, "rough text"),), recording_duration_seconds=8)
    assert result.segments == ()


def test_insertion_within_one_turn_does_not_split_speaker_or_duplicate_text() -> None:
    text = "Το αυτοκίνητο έχει ήδη επισκευαστεί σήμερα."
    result = align_conversation(text, (
        turn(1, 6, "Το αυτοκίνητο έχει επισκευαστεί σήμερα"),
    ), recording_duration_seconds=8)
    assert len(result.segments) == 1
    assert result.segments[0].text == text
    assert result.segments[0].speaker_label == "A"
