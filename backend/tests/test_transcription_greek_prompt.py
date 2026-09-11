from __future__ import annotations

import hashlib
from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest

from app.services.transcription.prompt import (
    GREEK_CALLCENTER_PROMPT_VERSION,
    GLOBAL_CONVERSATION_CONTEXT_POLICY_VERSION,
    MAX_INDIVIDUAL_TERM_CHARACTERS,
    MAX_PREVIOUS_CONTEXT_CHARACTERS,
    MAX_PROMPT_CHARACTERS,
    MAX_VOCABULARY_CHARACTERS,
    PRIORITY_COMPANY,
    PRIORITY_CURRENT_CALL,
    PRIORITY_CURRENT_PARTY,
    PRIORITY_GENERAL,
    PRIORITY_QUEUE,
    PRIORITY_SELECTED_KEYWORD,
    PRIORITY_SELECTED_OPERATOR,
    TRACK_ROLE_CALLEE,
    TRACK_ROLE_CALLER,
    TRACK_ROLE_CHANNEL_A,
    TRACK_ROLE_CHANNEL_B,
    TRACK_ROLE_OPERATOR,
    TRACK_ROLE_ANONYMOUS,
    VOCABULARY_SOURCE_COMPANY,
    VOCABULARY_SOURCE_CURRENT_CALL,
    VOCABULARY_SOURCE_CURRENT_PARTY,
    VOCABULARY_SOURCE_GENERAL,
    VOCABULARY_SOURCE_QUEUE,
    VOCABULARY_SOURCE_SELECTED_KEYWORD,
    VOCABULARY_SOURCE_SELECTED_OPERATOR,
    LegacyVocabularyPromptBuilder,
    RankedVocabularyTerm,
    V2GreekPromptBuilder,
    build_vocabulary_prompt,
)
from app.services.transcription.types import AudioPlan, AudioTrack


def _operator_plan() -> AudioPlan:
    return AudioPlan(
        mode="operator_channel",
        tracks=(
            AudioTrack(
                track_id="operator-channel",
                source_path=Path("operator.wav"),
                channel_index=1,
                operator_id="operator-1",
                attribution_status="confirmed_by_pbx",
            ),
        ),
        operator_channel=1,
        stereo_separated=True,
        attribution_status="confirmed_by_pbx",
    )


def _dual_plan(*, caller_channel: int | None, callee_channel: int | None) -> AudioPlan:
    return AudioPlan(
        mode="dual_channel",
        tracks=tuple(
            AudioTrack(
                track_id=f"channel-{channel}",
                source_path=Path(f"channel-{channel}.wav"),
                channel_index=channel,
                attribution_status=(
                    "caller_callee_only"
                    if caller_channel is not None and callee_channel is not None
                    else "channel_unknown"
                ),
            )
            for channel in (0, 1)
        ),
        stereo_separated=True,
        caller_channel=caller_channel,
        callee_channel=callee_channel,
        attribution_status=(
            "caller_callee_only"
            if caller_channel is not None and callee_channel is not None
            else "channel_unknown"
        ),
    )


def _mono_plan() -> AudioPlan:
    return AudioPlan(
        mode="mono_diarization",
        tracks=(
            AudioTrack(
                track_id="mono-diarization",
                source_path=Path("mono.wav"),
                operator_id=None,
                attribution_status="anonymous_diarization",
                diarized=True,
                speaker_source="openai_diarization",
            ),
        ),
        attribution_status="anonymous_diarization",
    )


def _term(
    value: str,
    priority: int,
    source: str,
    *roles: str,
) -> RankedVocabularyTerm:
    return RankedVocabularyTerm(
        value=value,
        priority=priority,
        source=source,  # type: ignore[arg-type]
        roles=roles,  # type: ignore[arg-type]
    )


def test_legacy_prompt_contract_remains_byte_stable() -> None:
    values = ["  Alpha   Beta ", "alpha beta", "", " Γιώργος "]
    expected = "Greek business vocabulary and names: Alpha Beta, Γιώργος"
    expected_version = hashlib.sha256(expected.encode("utf-8")).hexdigest()[:16]

    plan = LegacyVocabularyPromptBuilder().build(values)

    assert (plan.text, plan.version) == (expected, expected_version)
    assert plan.prompt_hash is None
    assert plan.template_version is None
    assert plan.vocabulary_hash is None
    assert build_vocabulary_prompt(values) == (expected, expected_version)


def test_v2_prompt_contains_greek_safety_instructions_and_full_hashes() -> None:
    manifest = V2GreekPromptBuilder().build_manifest(
        _operator_plan(),
        (
            _term(
                "Μαρία Παπαδοπούλου",
                PRIORITY_SELECTED_OPERATOR,
                VOCABULARY_SOURCE_SELECTED_OPERATOR,
                TRACK_ROLE_OPERATOR,
            ),
            _term(
                "Road Assistance",
                PRIORITY_COMPANY,
                VOCABULARY_SOURCE_COMPANY,
            ),
        ),
    )

    prompt = manifest.build("operator-channel")

    assert manifest.template_version == GREEK_CALLCENTER_PROMPT_VERSION
    assert len(manifest.vocabulary_hash) == 64
    assert len(manifest.prompt_identity) == 64
    assert "Αυτή είναι ελληνική τηλεφωνική συνομιλία." in prompt.text
    assert "Μεταγράψε μόνο ό,τι ακούγεται πραγματικά." in prompt.text
    assert "Μην συμπληρώνεις και μην επινοείς" in prompt.text
    assert "ονόματα, επωνυμίες εταιρειών, μοντέλα οχημάτων" in prompt.text
    assert "αριθμούς τηλεφώνου, πινακίδες κυκλοφορίας και ημερομηνίες" in prompt.text
    assert "Μη μεταφράζεις αγγλικούς εμπορικούς ή τεχνικούς όρους." in prompt.text
    assert "μην το επαναλάβεις εκτός αν ακούγεται ξανά." in prompt.text
    assert "Γνωστός ρόλος καναλιού: Operator." in prompt.text
    assert "Μαρία Παπαδοπούλου" in prompt.text
    assert "Road Assistance" in prompt.text
    assert prompt.version == manifest.prompt_identity
    assert prompt.prompt_hash == hashlib.sha256(prompt.text.encode("utf-8")).hexdigest()
    assert prompt.template_version == GREEK_CALLCENTER_PROMPT_VERSION
    assert prompt.vocabulary_hash == manifest.track("operator-channel").vocabulary_hash
    assert prompt.previous_context_characters == 0


def test_ranked_vocabulary_is_normalized_deduplicated_and_order_independent() -> None:
    terms = (
        _term(
            "  Τρέχουσα   κλήση ",
            PRIORITY_CURRENT_CALL,
            VOCABULARY_SOURCE_CURRENT_CALL,
        ),
        _term("Alpha", PRIORITY_COMPANY, VOCABULARY_SOURCE_COMPANY),
        _term(
            " alpha ",
            PRIORITY_SELECTED_KEYWORD,
            VOCABULARY_SOURCE_SELECTED_KEYWORD,
        ),
        _term("Τμήμα Service", PRIORITY_QUEUE, VOCABULARY_SOURCE_QUEUE),
        _term("γενικός όρος", PRIORITY_GENERAL, VOCABULARY_SOURCE_GENERAL),
    )
    builder = V2GreekPromptBuilder()

    forward = builder.build_manifest(_operator_plan(), terms)
    reverse = builder.build_manifest(_operator_plan(), tuple(reversed(terms)))
    track = forward.track("operator-channel")

    assert forward.vocabulary_hash == reverse.vocabulary_hash
    assert forward.prompt_identity == reverse.prompt_identity
    assert track.vocabulary_hash == reverse.track("operator-channel").vocabulary_hash
    assert [term.value for term in track.terms] == [
        "Τρέχουσα κλήση",
        "alpha",
        "Τμήμα Service",
        "γενικός όρος",
    ]
    assert track.terms[1].priority == PRIORITY_SELECTED_KEYWORD
    assert track.terms[1].source == VOCABULARY_SOURCE_SELECTED_KEYWORD


def test_confirmed_track_roles_filter_operator_caller_and_callee_vocabulary() -> None:
    terms = (
        _term(
            "Selected Operator",
            PRIORITY_SELECTED_OPERATOR,
            VOCABULARY_SOURCE_SELECTED_OPERATOR,
            TRACK_ROLE_OPERATOR,
        ),
        _term(
            "Caller Name",
            PRIORITY_CURRENT_PARTY,
            VOCABULARY_SOURCE_CURRENT_PARTY,
            TRACK_ROLE_CALLER,
        ),
        _term(
            "Callee Name",
            PRIORITY_CURRENT_PARTY,
            VOCABULARY_SOURCE_CURRENT_PARTY,
            TRACK_ROLE_CALLEE,
        ),
        _term("Shared Company", PRIORITY_COMPANY, VOCABULARY_SOURCE_COMPANY),
    )
    operator = V2GreekPromptBuilder().build_manifest(_operator_plan(), terms)
    dual = V2GreekPromptBuilder().build_manifest(
        _dual_plan(caller_channel=0, callee_channel=1),
        terms,
    )

    assert [term.value for term in operator.track("operator-channel").terms] == [
        "Selected Operator",
        "Shared Company",
    ]
    assert [term.value for term in dual.track("channel-0").terms] == [
        "Caller Name",
        "Shared Company",
    ]
    assert [term.value for term in dual.track("channel-1").terms] == [
        "Callee Name",
        "Shared Company",
    ]
    assert "Selected Operator" not in dual.build("channel-0").text
    assert "Selected Operator" not in dual.build("channel-1").text


def test_unknown_dual_channels_stay_neutral_and_exclude_party_specific_terms() -> None:
    manifest = V2GreekPromptBuilder().build_manifest(
        _dual_plan(caller_channel=None, callee_channel=None),
        (
            _term(
                "Operator Name",
                PRIORITY_SELECTED_OPERATOR,
                VOCABULARY_SOURCE_SELECTED_OPERATOR,
                TRACK_ROLE_OPERATOR,
            ),
            _term(
                "Caller Name",
                PRIORITY_CURRENT_PARTY,
                VOCABULARY_SOURCE_CURRENT_PARTY,
                TRACK_ROLE_CALLER,
            ),
            _term("Shared Company", PRIORITY_COMPANY, VOCABULARY_SOURCE_COMPANY),
        ),
    )

    channel_a = manifest.build("channel-0")
    channel_b = manifest.build("channel-1")

    assert channel_a.track_role == TRACK_ROLE_CHANNEL_A
    assert channel_b.track_role == TRACK_ROLE_CHANNEL_B
    assert "Γνωστός ρόλος καναλιού: Channel A." in channel_a.text
    assert "Γνωστός ρόλος καναλιού: Channel B." in channel_b.text
    assert "Operator Name" not in channel_a.text
    assert "Caller Name" not in channel_a.text
    assert "Operator" not in channel_b.text
    assert "Caller" not in channel_b.text


def test_vocabulary_rejects_empty_and_secret_like_values() -> None:
    secret_values = (
        "",
        "   ",
        "sk-abcdefghijklmnopqrstuvwxyz123456",
        "password=hunter2",
        "-----BEGIN PRIVATE KEY-----",
        "Bearer abcdefghijklmnopqrstuvwxyz",
        "https://username:password@example.test/path",
    )
    terms = tuple(
        _term(value, PRIORITY_GENERAL, VOCABULARY_SOURCE_GENERAL) for value in secret_values
    ) + (
        _term(
            "  ασφαλής   εμπορικός όρος ",
            PRIORITY_COMPANY,
            VOCABULARY_SOURCE_COMPANY,
        ),
    )

    manifest = V2GreekPromptBuilder().build_manifest(_operator_plan(), terms)
    track = manifest.track("operator-channel")

    assert [term.value for term in track.terms] == ["ασφαλής εμπορικός όρος"]
    assert "hunter2" not in repr(manifest)
    assert "hunter2" not in repr(track)
    assert "ασφαλής εμπορικός όρος" not in repr(track)
    assert "ασφαλής εμπορικός όρος" not in repr(track.terms[0])


def test_vocabulary_and_rendered_prompt_obey_explicit_character_limits() -> None:
    terms = tuple(
        _term(
            f"{index:03d}-" + ("x" * 150),
            PRIORITY_GENERAL,
            VOCABULARY_SOURCE_GENERAL,
        )
        for index in range(50)
    )

    manifest = V2GreekPromptBuilder().build_manifest(_operator_plan(), terms)
    track = manifest.track("operator-channel")
    prompt = manifest.build("operator-channel")

    assert all(len(term.value) <= MAX_INDIVIDUAL_TERM_CHARACTERS for term in track.terms)
    assert len(track.text) <= MAX_VOCABULARY_CHARACTERS
    assert len(prompt.text) <= MAX_PROMPT_CHARACTERS


def test_previous_context_is_optional_tail_bounded_and_not_part_of_stable_identity() -> None:
    manifest = V2GreekPromptBuilder().build_manifest(
        _operator_plan(),
        (
            _term(
                "Service",
                PRIORITY_COMPANY,
                VOCABULARY_SOURCE_COMPANY,
            ),
        ),
    )
    without_context = manifest.build("operator-channel")
    previous = "discard-this-prefix " + ("λέξη " * 180) + "keep-this-tail"
    with_context = manifest.build("operator-channel", previous_context=previous)

    assert "Προηγούμενο αποδεκτό κείμενο" not in without_context.text
    assert without_context.previous_context_characters == 0
    assert "Προηγούμενο αποδεκτό κείμενο" in with_context.text
    assert "keep-this-tail" in with_context.text
    assert "discard-this-prefix" not in with_context.text
    assert with_context.previous_context_characters <= MAX_PREVIOUS_CONTEXT_CHARACTERS
    assert with_context.version == without_context.version == manifest.prompt_identity
    assert with_context.vocabulary_hash == without_context.vocabulary_hash
    assert with_context.prompt_hash != without_context.prompt_hash


def test_prompt_domain_objects_are_immutable_and_hide_personal_text_from_repr() -> None:
    term = _term(
        "Personal Name",
        PRIORITY_SELECTED_OPERATOR,
        VOCABULARY_SOURCE_SELECTED_OPERATOR,
        TRACK_ROLE_OPERATOR,
    )
    manifest = V2GreekPromptBuilder().build_manifest(_operator_plan(), (term,))
    prompt = manifest.build("operator-channel", previous_context="private transcript")

    assert "Personal Name" not in repr(term)
    assert "Personal Name" not in repr(manifest)
    assert "private transcript" not in repr(prompt)
    with pytest.raises(FrozenInstanceError):
        term.priority = 1  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        manifest.prompt_identity = "changed"  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        prompt.version = "changed"  # type: ignore[misc]


def test_vocabulary_or_proven_role_changes_prompt_identity() -> None:
    base_terms = (
        _term("Shared Company", PRIORITY_COMPANY, VOCABULARY_SOURCE_COMPANY),
        _term(
            "Caller Name",
            PRIORITY_CURRENT_PARTY,
            VOCABULARY_SOURCE_CURRENT_PARTY,
            TRACK_ROLE_CALLER,
        ),
    )
    builder = V2GreekPromptBuilder()
    base = builder.build_manifest(
        _dual_plan(caller_channel=0, callee_channel=1),
        base_terms,
    )
    changed_vocabulary = builder.build_manifest(
        _dual_plan(caller_channel=0, callee_channel=1),
        base_terms
        + (
            _term(
                "Call-specific",
                PRIORITY_CURRENT_CALL,
                VOCABULARY_SOURCE_CURRENT_CALL,
            ),
        ),
    )
    reversed_roles = builder.build_manifest(
        _dual_plan(caller_channel=1, callee_channel=0),
        base_terms,
    )

    assert changed_vocabulary.vocabulary_hash != base.vocabulary_hash
    assert changed_vocabulary.prompt_identity != base.prompt_identity
    assert reversed_roles.vocabulary_hash != base.vocabulary_hash
    assert reversed_roles.prompt_identity != base.prompt_identity


def test_mono_prompt_uses_anonymous_role_and_bounded_global_context() -> None:
    manifest = V2GreekPromptBuilder().build_manifest(
        _mono_plan(),
        (
            _term(
                "Selected Operator",
                PRIORITY_SELECTED_OPERATOR,
                VOCABULARY_SOURCE_SELECTED_OPERATOR,
                TRACK_ROLE_OPERATOR,
            ),
            _term("Shared Company", PRIORITY_COMPANY, VOCABULARY_SOURCE_COMPANY),
        ),
    )
    previous = "discard-prefix " + ("context " * 100) + "keep-tail"

    prompt = manifest.build_anonymous(
        "mono-diarization",
        "A",
        previous_context=previous,
    )

    assert manifest.context_policy == GLOBAL_CONVERSATION_CONTEXT_POLICY_VERSION
    assert prompt.track_role == TRACK_ROLE_ANONYMOUS
    assert "Τρέχων ανώνυμος ομιλητής: A" in prompt.text
    assert "μόνο ως συμφραζόμενο" in prompt.text
    assert "δεν επιτρέπεται να μετονομάσει" in prompt.text
    assert "keep-tail" in prompt.text
    assert "discard-prefix" not in prompt.text
    assert "Shared Company" in prompt.text
    assert "Selected Operator" not in prompt.text
    assert "Selected Operator" not in prompt.keywords
    assert prompt.previous_context_characters <= MAX_PREVIOUS_CONTEXT_CHARACTERS


@pytest.mark.parametrize("label", ["", "Speaker A", "../operator", "A\nOperator"])
def test_mono_prompt_rejects_malformed_anonymous_labels(label: str) -> None:
    manifest = V2GreekPromptBuilder().build_manifest(_mono_plan(), ())

    with pytest.raises(ValueError, match="anonymous speaker label"):
        manifest.build_anonymous("mono-diarization", label)


def test_v2_prompt_builder_rejects_legacy_mode() -> None:
    plan = AudioPlan(
        mode="legacy",
        tracks=(
            AudioTrack(
                track_id="unsupported",
                source_path=Path("unsupported.wav"),
            ),
        ),
    )

    with pytest.raises(ValueError, match="standard transcription"):
        V2GreekPromptBuilder().build_manifest(plan, ())


def test_conversation_prompt_uses_vocabulary_without_single_speaker_context() -> None:
    manifest = V2GreekPromptBuilder().build_manifest(
        _mono_plan(), (_term("Sample Company", PRIORITY_COMPANY, VOCABULARY_SOURCE_COMPANY),),
    )
    prompt = manifest.build_conversation("mono-diarization")
    assert "Sample Company" in prompt.text
    assert prompt.keywords == ("Sample Company",)
    assert "όλους τους ομιλητές" in prompt.text
    assert "Τρέχων ανώνυμος ομιλητής" not in prompt.text
    assert "Γνωστός ρόλος καναλιού" not in prompt.text
    assert prompt.previous_context_characters == 0
    assert prompt.prompt_hash != manifest.build_anonymous("mono-diarization", "A").prompt_hash


def test_keyword_hints_cannot_bypass_vocabulary_filtering_and_limits() -> None:
    terms = tuple(_term(f"Term {index}", PRIORITY_COMPANY, VOCABULARY_SOURCE_COMPANY) for index in range(80))
    terms += (
        _term("password=secret-value", 100, VOCABULARY_SOURCE_COMPANY),
        _term("123456789", 100, VOCABULARY_SOURCE_COMPANY),
    )
    prompt = V2GreekPromptBuilder().build_manifest(_mono_plan(), terms).build_conversation("mono-diarization")
    assert len(prompt.keywords) == 64
    assert all(term.startswith("Term ") for term in prompt.keywords)
    assert "secret-value" not in repr(prompt)
