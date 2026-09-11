from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from dataclasses import dataclass, field
from typing import Literal, Protocol

from app.services.transcription.types import AudioPlan, AudioTrack


GREEK_CALLCENTER_PROMPT_VERSION = "greek-callcenter-v2"
GREEK_CALLCENTER_RENDERER_VERSION = "greek-callcenter-renderer-v2"
RANKED_VOCABULARY_VERSION = "ranked-vocabulary-v1"
SAME_TRACK_CONTEXT_POLICY_VERSION = "previous-accepted-same-track-tail-v1"
GLOBAL_CONVERSATION_CONTEXT_POLICY_VERSION = (
    "previous-accepted-global-conversation-tail-v1"
)

MAX_VOCABULARY_CHARACTERS = 3000
MAX_PREVIOUS_CONTEXT_CHARACTERS = 500
MAX_INDIVIDUAL_TERM_CHARACTERS = 100
MAX_PROMPT_CHARACTERS = 5000
MAX_TRANSCRIPTION_KEYWORDS = 64

GREEK_CALLCENTER_INSTRUCTIONS = (
    "Αυτή είναι ελληνική τηλεφωνική συνομιλία.",
    "Μεταγράψε μόνο ό,τι ακούγεται πραγματικά.",
    "Μην συμπληρώνεις και μην επινοείς λέξεις ή γεγονότα που δεν ακούγονται.",
    (
        "Διατήρησε πιστά ονόματα, επωνυμίες εταιρειών, μοντέλα οχημάτων, "
        "αριθμούς τηλεφώνου, πινακίδες κυκλοφορίας και ημερομηνίες."
    ),
    "Μη μεταφράζεις αγγλικούς εμπορικούς ή τεχνικούς όρους.",
    (
        "Το προηγούμενο κείμενο παρέχεται μόνο ως συμφραζόμενο· μην το "
        "επαναλάβεις εκτός αν ακούγεται ξανά."
    ),
)
GREEK_CALLCENTER_TEMPLATE = "\n".join(GREEK_CALLCENTER_INSTRUCTIONS)

TRACK_ROLE_OPERATOR = "Operator"
TRACK_ROLE_CALLER = "Caller"
TRACK_ROLE_CALLEE = "Callee"
TRACK_ROLE_CHANNEL_A = "Channel A"
TRACK_ROLE_CHANNEL_B = "Channel B"
TRACK_ROLE_ANONYMOUS = "Anonymous speaker"
TrackRoleValue = Literal[
    "Operator",
    "Caller",
    "Callee",
    "Channel A",
    "Channel B",
    "Anonymous speaker",
]
TRACK_ROLES: tuple[TrackRoleValue, ...] = (
    TRACK_ROLE_OPERATOR,
    TRACK_ROLE_CALLER,
    TRACK_ROLE_CALLEE,
    TRACK_ROLE_CHANNEL_A,
    TRACK_ROLE_CHANNEL_B,
    TRACK_ROLE_ANONYMOUS,
)
_TRACK_ROLE_ORDER = {role: position for position, role in enumerate(TRACK_ROLES)}

VOCABULARY_SOURCE_SELECTED_OPERATOR = "selected_operator"
VOCABULARY_SOURCE_CURRENT_PARTY = "current_party"
VOCABULARY_SOURCE_CURRENT_CALL = "current_call"
VOCABULARY_SOURCE_SELECTED_KEYWORD = "selected_keyword"
VOCABULARY_SOURCE_COMPANY = "company"
VOCABULARY_SOURCE_QUEUE = "queue"
VOCABULARY_SOURCE_GENERAL = "general"
VocabularySourceValue = Literal[
    "selected_operator",
    "current_party",
    "current_call",
    "selected_keyword",
    "company",
    "queue",
    "general",
]
VOCABULARY_SOURCES: tuple[VocabularySourceValue, ...] = (
    VOCABULARY_SOURCE_SELECTED_OPERATOR,
    VOCABULARY_SOURCE_CURRENT_PARTY,
    VOCABULARY_SOURCE_CURRENT_CALL,
    VOCABULARY_SOURCE_SELECTED_KEYWORD,
    VOCABULARY_SOURCE_COMPANY,
    VOCABULARY_SOURCE_QUEUE,
    VOCABULARY_SOURCE_GENERAL,
)

PRIORITY_SELECTED_OPERATOR = 100
PRIORITY_CURRENT_PARTY = 95
PRIORITY_CURRENT_CALL = 90
PRIORITY_SELECTED_KEYWORD = 80
PRIORITY_COMPANY = 70
PRIORITY_QUEUE = 60
PRIORITY_GENERAL = 40

_SENSITIVE_VALUE_PATTERNS = (
    re.compile(r"-----BEGIN [A-Z0-9 ]*PRIVATE KEY-----", re.IGNORECASE),
    re.compile(r"\bsk-[A-Za-z0-9_-]{16,}\b"),
    re.compile(r"\bgh[pousr]_[A-Za-z0-9]{20,}\b", re.IGNORECASE),
    re.compile(r"\bxox[baprs]-[A-Za-z0-9-]{16,}\b", re.IGNORECASE),
    re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
    re.compile(r"\bBearer\s+[A-Za-z0-9._~+/-]{16,}\b", re.IGNORECASE),
    re.compile(
        r"\b(?:api[_ -]?key|client[_ -]?secret|password|passwd|private[_ -]?key|"
        r"secret|access[_ -]?token|refresh[_ -]?token|authorization)\b\s*[:=]",
        re.IGNORECASE,
    ),
    re.compile(r"://[^/\s:@]+:[^/\s@]+@"),
)


def _normalize_whitespace(value: str) -> str:
    normalized = unicodedata.normalize("NFC", str(value))
    return " ".join(normalized.strip().split())


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _canonical_sha256(value: object) -> str:
    canonical = json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )
    return _sha256_text(canonical)


def _canonical_role(value: str) -> TrackRoleValue:
    normalized = _normalize_whitespace(value).casefold()
    for role in TRACK_ROLES:
        if role.casefold() == normalized:
            return role
    raise ValueError("Track role is not supported by the V2 prompt policy.")


def _looks_sensitive(value: str) -> bool:
    return any(pattern.search(value) is not None for pattern in _SENSITIVE_VALUE_PATTERNS)


@dataclass(frozen=True, slots=True)
class GreekPromptLimits:
    maximum_vocabulary_characters: int = MAX_VOCABULARY_CHARACTERS
    maximum_previous_context_characters: int = MAX_PREVIOUS_CONTEXT_CHARACTERS
    maximum_individual_term_characters: int = MAX_INDIVIDUAL_TERM_CHARACTERS
    maximum_prompt_characters: int = MAX_PROMPT_CHARACTERS

    def __post_init__(self) -> None:
        if (
            self.maximum_vocabulary_characters <= 0
            or self.maximum_previous_context_characters <= 0
            or self.maximum_individual_term_characters <= 0
            or self.maximum_prompt_characters <= 0
        ):
            raise ValueError("Greek prompt limits must be positive.")
        if self.maximum_individual_term_characters > self.maximum_vocabulary_characters:
            raise ValueError(
                "The individual vocabulary limit cannot exceed the total vocabulary limit."
            )
        maximum_optional_characters = (
            self.maximum_vocabulary_characters + self.maximum_previous_context_characters + 256
        )
        if (
            len(GREEK_CALLCENTER_TEMPLATE) + maximum_optional_characters
            > self.maximum_prompt_characters
        ):
            raise ValueError("The total prompt limit cannot contain the configured sections.")

    def identity(self) -> dict[str, int]:
        return {
            "maximum_individual_term_characters": (self.maximum_individual_term_characters),
            "maximum_previous_context_characters": (self.maximum_previous_context_characters),
            "maximum_prompt_characters": self.maximum_prompt_characters,
            "maximum_vocabulary_characters": self.maximum_vocabulary_characters,
        }


DEFAULT_GREEK_PROMPT_LIMITS = GreekPromptLimits()


@dataclass(frozen=True, slots=True)
class RankedVocabularyTerm:
    value: str = field(repr=False)
    priority: int
    source: VocabularySourceValue
    roles: tuple[TrackRoleValue, ...] = ()

    def __post_init__(self) -> None:
        if isinstance(self.priority, bool) or not isinstance(self.priority, int):
            raise TypeError("Vocabulary priority must be an integer.")
        if not 0 <= self.priority <= 100:
            raise ValueError("Vocabulary priority must be between 0 and 100.")
        if self.source not in VOCABULARY_SOURCES:
            raise ValueError("Vocabulary source is not supported.")
        roles = tuple(
            sorted(
                {_canonical_role(role) for role in self.roles},
                key=_TRACK_ROLE_ORDER.__getitem__,
            )
        )
        object.__setattr__(self, "roles", roles)


@dataclass(frozen=True, slots=True)
class RankedVocabularyManifest:
    track_id: str
    track_role: TrackRoleValue
    terms: tuple[RankedVocabularyTerm, ...] = field(repr=False)
    text: str = field(repr=False)
    vocabulary_hash: str

    @property
    def keywords(self) -> tuple[str, ...]:
        # Use only the bounded, sanitized, role-filtered vocabulary. Bare PBX
        # numbers are context, not spelling hints for spoken registration numbers.
        return tuple(
            term.value for term in self.terms if any(char.isalpha() for char in term.value)
        )[:MAX_TRANSCRIPTION_KEYWORDS]

    def identity(self) -> dict[str, object]:
        return {
            "schema": RANKED_VOCABULARY_VERSION,
            "track_id": self.track_id,
            "track_role": self.track_role,
            "vocabulary_hash": self.vocabulary_hash,
        }


@dataclass(frozen=True, slots=True)
class PromptPlan:
    text: str = field(repr=False)
    version: str
    prompt_hash: str | None = None
    template_version: str | None = None
    vocabulary_hash: str | None = None
    track_id: str | None = None
    track_role: TrackRoleValue | None = None
    previous_context_characters: int = 0
    keywords: tuple[str, ...] = field(default=(), repr=False)


@dataclass(frozen=True, slots=True)
class V2PromptManifest:
    tracks: tuple[RankedVocabularyManifest, ...] = field(repr=False)
    limits: GreekPromptLimits
    template_version: str
    vocabulary_hash: str
    prompt_identity: str
    context_policy: str = SAME_TRACK_CONTEXT_POLICY_VERSION

    def identity(self) -> dict[str, object]:
        return {
            "context_policy": self.context_policy,
            "limits": self.limits.identity(),
            "prompt_identity": self.prompt_identity,
            "renderer_version": GREEK_CALLCENTER_RENDERER_VERSION,
            "maximum_transcription_keywords": MAX_TRANSCRIPTION_KEYWORDS,
            "template_hash": _sha256_text(GREEK_CALLCENTER_TEMPLATE),
            "template_version": self.template_version,
            "tracks": [
                {
                    "track_id": track.track_id,
                    "track_role": track.track_role,
                    "vocabulary_hash": track.vocabulary_hash,
                }
                for track in sorted(self.tracks, key=lambda item: item.track_id)
            ],
            "vocabulary_hash": self.vocabulary_hash,
        }

    def track(self, track_id: str) -> RankedVocabularyManifest:
        for track in self.tracks:
            if track.track_id == track_id:
                return track
        raise ValueError("The requested track has no V2 prompt manifest.")

    def build(
        self,
        track_id: str,
        previous_context: str | None = None,
    ) -> PromptPlan:
        track = self.track(track_id)
        context = _normalize_whitespace(previous_context or "")
        if len(context) > self.limits.maximum_previous_context_characters:
            context = context[-self.limits.maximum_previous_context_characters :]
        context = context.strip()

        sections = [GREEK_CALLCENTER_TEMPLATE]
        sections.append(f"Γνωστός ρόλος καναλιού: {track.track_role}.")
        if track.text:
            sections.append(f"Σχετικό λεξιλόγιο κατά σειρά προτεραιότητας:\n{track.text}")
        if context:
            sections.append(
                "Προηγούμενο αποδεκτό κείμενο του ίδιου καναλιού "
                f"(μόνο ως συμφραζόμενο):\n{context}"
            )
        text = "\n\n".join(sections)
        if len(text) > self.limits.maximum_prompt_characters:
            raise ValueError("The rendered V2 prompt exceeds its configured limit.")
        return PromptPlan(
            text=text,
            version=self.prompt_identity,
            prompt_hash=_sha256_text(text),
            template_version=self.template_version,
            vocabulary_hash=track.vocabulary_hash,
            track_id=track.track_id,
            track_role=track.track_role,
            previous_context_characters=len(context),
            keywords=track.keywords,
        )

    def build_conversation(self, track_id: str) -> PromptPlan:
        """Prompt the complete mixed recording, without a single-speaker role."""

        track = self.track(track_id)
        if (
            track.track_role != TRACK_ROLE_ANONYMOUS
            or self.context_policy != GLOBAL_CONVERSATION_CONTEXT_POLICY_VERSION
        ):
            raise ValueError("Conversation prompting requires a mono V2 prompt manifest.")
        sections = [
            GREEK_CALLCENTER_TEMPLATE,
            "Μετάγραψε όλους τους ομιλητές με τη σειρά που ακούγονται, "
            "χωρίς ετικέτες ομιλητών.",
        ]
        if track.text:
            sections.append(f"Σχετικό λεξιλόγιο κατά σειρά προτεραιότητας:\n{track.text}")
        text = "\n\n".join(sections)
        if len(text) > self.limits.maximum_prompt_characters:
            raise ValueError("The rendered V2 prompt exceeds its configured limit.")
        return PromptPlan(
            text=text,
            version=self.prompt_identity,
            prompt_hash=_sha256_text(text),
            template_version=self.template_version,
            vocabulary_hash=track.vocabulary_hash,
            track_id=track.track_id,
            track_role=track.track_role,
            keywords=track.keywords,
        )

    def build_anonymous(
        self,
        track_id: str,
        speaker_label: str,
        previous_context: str | None = None,
    ) -> PromptPlan:
        """Render one mono-refinement prompt without implying speaker identity."""

        track = self.track(track_id)
        if (
            track.track_role != TRACK_ROLE_ANONYMOUS
            or self.context_policy != GLOBAL_CONVERSATION_CONTEXT_POLICY_VERSION
        ):
            raise ValueError("Anonymous prompting requires a mono V2 prompt manifest.")
        label = _normalize_whitespace(speaker_label)
        if (
            len(label) > 32
            or re.fullmatch(r"[A-Za-z][A-Za-z0-9_-]{0,31}", label) is None
        ):
            raise ValueError("The anonymous speaker label is invalid.")

        context = _normalize_whitespace(previous_context or "")
        if len(context) > self.limits.maximum_previous_context_characters:
            context = context[-self.limits.maximum_previous_context_characters :]
        context = context.strip()

        sections = [GREEK_CALLCENTER_TEMPLATE]
        sections.append(
            f"Τρέχων ανώνυμος ομιλητής: {label}. "
            "Η ετικέτα είναι μόνο μεταδεδομένο και δεν δηλώνει ταυτότητα."
        )
        sections.append(
            "Μετέγραψε μόνο τον τρέχοντα ήχο. Μην αλλάξεις, μετονομάσεις ή "
            "αποδώσεις ταυτότητα στην ετικέτα ομιλητή."
        )
        if track.text:
            sections.append(
                "Σχετικό λεξιλόγιο κατά σειρά προτεραιότητας:\n"
                f"{track.text}"
            )
        if context:
            sections.append(
                "Προηγούμενο αποδεκτό κείμενο της συνομιλίας "
                "(μόνο ως συμφραζόμενο, όχι ως ήχος προς μεταγραφή):\n"
                f"{context}\n"
                "Το προηγούμενο κείμενο δεν επιτρέπεται να μετονομάσει ή "
                "να ταυτοποιήσει κανέναν ομιλητή."
            )
        text = "\n\n".join(sections)
        if len(text) > self.limits.maximum_prompt_characters:
            raise ValueError("The rendered V2 prompt exceeds its configured limit.")
        return PromptPlan(
            text=text,
            version=self.prompt_identity,
            prompt_hash=_sha256_text(text),
            template_version=self.template_version,
            vocabulary_hash=track.vocabulary_hash,
            track_id=track.track_id,
            track_role=track.track_role,
            previous_context_characters=len(context),
            keywords=track.keywords,
        )


class V2GreekPromptBuilder:
    def __init__(
        self,
        *,
        limits: GreekPromptLimits = DEFAULT_GREEK_PROMPT_LIMITS,
    ) -> None:
        self.limits = limits

    def build_manifest(
        self,
        plan: AudioPlan,
        terms: list[RankedVocabularyTerm] | tuple[RankedVocabularyTerm, ...],
    ) -> V2PromptManifest:
        if plan.mode not in {"operator_channel", "dual_channel", "mono_diarization"}:
            raise ValueError("The V2 Greek prompt applies only to standard transcription.")
        context_policy = (
            GLOBAL_CONVERSATION_CONTEXT_POLICY_VERSION
            if plan.mode == "mono_diarization"
            else SAME_TRACK_CONTEXT_POLICY_VERSION
        )
        candidates = tuple(terms)
        tracks = tuple(
            self._build_track_manifest(
                track,
                self._track_role(plan, track),
                candidates,
            )
            for track in plan.tracks
        )
        if not tracks or len({track.track_id for track in tracks}) != len(tracks):
            raise ValueError("The V2 prompt plan requires unique audio tracks.")

        vocabulary_payload = {
            "schema": RANKED_VOCABULARY_VERSION,
            "tracks": [
                {
                    "track_id": track.track_id,
                    "track_role": track.track_role,
                    "vocabulary_hash": track.vocabulary_hash,
                }
                for track in sorted(tracks, key=lambda item: item.track_id)
            ],
        }
        vocabulary_hash = _canonical_sha256(vocabulary_payload)
        prompt_identity = _canonical_sha256(
            {
                "context_policy": context_policy,
                "limits": self.limits.identity(),
                "renderer_version": GREEK_CALLCENTER_RENDERER_VERSION,
                "maximum_transcription_keywords": MAX_TRANSCRIPTION_KEYWORDS,
                "template_hash": _sha256_text(GREEK_CALLCENTER_TEMPLATE),
                "template_version": GREEK_CALLCENTER_PROMPT_VERSION,
                "vocabulary": vocabulary_payload,
                "vocabulary_hash": vocabulary_hash,
            }
        )
        return V2PromptManifest(
            tracks=tracks,
            limits=self.limits,
            template_version=GREEK_CALLCENTER_PROMPT_VERSION,
            vocabulary_hash=vocabulary_hash,
            prompt_identity=prompt_identity,
            context_policy=context_policy,
        )

    def _build_track_manifest(
        self,
        track: AudioTrack,
        track_role: TrackRoleValue,
        candidates: tuple[RankedVocabularyTerm, ...],
    ) -> RankedVocabularyManifest:
        if not track.track_id or len(track.track_id) > 128:
            raise ValueError("The V2 prompt track identifier is invalid.")

        deduplicated: dict[str, RankedVocabularyTerm] = {}
        for raw_term in candidates:
            if raw_term.roles and track_role not in raw_term.roles:
                continue
            value = _normalize_whitespace(raw_term.value)
            if not value or _looks_sensitive(value):
                continue
            value = value[: self.limits.maximum_individual_term_characters].strip()
            if not value or _looks_sensitive(value):
                continue
            term = RankedVocabularyTerm(
                value=value,
                priority=raw_term.priority,
                source=raw_term.source,
                roles=raw_term.roles,
            )
            key = value.casefold()
            existing = deduplicated.get(key)
            if existing is None or self._term_precedes(term, existing):
                deduplicated[key] = term

        ranked = sorted(deduplicated.values(), key=self._term_order)
        selected: list[RankedVocabularyTerm] = []
        selected_characters = 0
        for term in ranked:
            separator_characters = 2 if selected else 0
            candidate_characters = selected_characters + separator_characters + len(term.value)
            if candidate_characters > self.limits.maximum_vocabulary_characters:
                continue
            selected.append(term)
            selected_characters = candidate_characters

        selected_terms = tuple(selected)
        text = ", ".join(term.value for term in selected_terms)
        payload = {
            "schema": RANKED_VOCABULARY_VERSION,
            "track_id": track.track_id,
            "track_role": track_role,
            "terms": [
                {
                    "priority": term.priority,
                    "roles": list(term.roles),
                    "source": term.source,
                    "value": term.value,
                }
                for term in selected_terms
            ],
        }
        return RankedVocabularyManifest(
            track_id=track.track_id,
            track_role=track_role,
            terms=selected_terms,
            text=text,
            vocabulary_hash=_canonical_sha256(payload),
        )

    @staticmethod
    def _term_order(term: RankedVocabularyTerm) -> tuple[object, ...]:
        return (
            -term.priority,
            term.value.casefold(),
            term.value,
            term.source,
            tuple(_TRACK_ROLE_ORDER[role] for role in term.roles),
        )

    @classmethod
    def _term_precedes(
        cls,
        candidate: RankedVocabularyTerm,
        existing: RankedVocabularyTerm,
    ) -> bool:
        if candidate.priority != existing.priority:
            return candidate.priority > existing.priority
        return cls._term_order(candidate) < cls._term_order(existing)

    @staticmethod
    def _track_role(plan: AudioPlan, track: AudioTrack) -> TrackRoleValue:
        if plan.mode == "mono_diarization":
            if (
                plan.attribution_status != "anonymous_diarization"
                or track.attribution_status != "anonymous_diarization"
                or track.operator_id is not None
                or not track.diarized
            ):
                raise ValueError(
                    "A mono prompt role requires anonymous diarization without attribution."
                )
            return TRACK_ROLE_ANONYMOUS
        if plan.mode == "operator_channel":
            if (
                plan.attribution_status != "confirmed_by_pbx"
                or track.attribution_status != "confirmed_by_pbx"
                or track.operator_id is None
                or track.channel_index != plan.operator_channel
            ):
                raise ValueError(
                    "An operator prompt role requires PBX-confirmed channel attribution."
                )
            return TRACK_ROLE_OPERATOR
        mapping_proven = (
            plan.attribution_status == "caller_callee_only"
            and track.attribution_status == "caller_callee_only"
            and plan.caller_channel in {0, 1}
            and plan.callee_channel in {0, 1}
            and plan.caller_channel != plan.callee_channel
            and {plan.caller_channel, plan.callee_channel} == {0, 1}
        )
        if mapping_proven and track.channel_index == plan.caller_channel:
            return TRACK_ROLE_CALLER
        if mapping_proven and track.channel_index == plan.callee_channel:
            return TRACK_ROLE_CALLEE
        if track.channel_index == 0:
            return TRACK_ROLE_CHANNEL_A
        if track.channel_index == 1:
            return TRACK_ROLE_CHANNEL_B
        raise ValueError("A dual-channel prompt track requires channel 0 or 1.")


class PromptBuilder(Protocol):
    def build(
        self,
        values: list[str] | tuple[str, ...],
        *,
        max_characters: int = 4000,
    ) -> PromptPlan: ...


class LegacyVocabularyPromptBuilder:
    """Reproduce the existing Greek vocabulary prompt byte for byte."""

    def build(
        self,
        values: list[str] | tuple[str, ...],
        *,
        max_characters: int = 4000,
    ) -> PromptPlan:
        clean: list[str] = []
        seen: set[str] = set()
        for raw_value in values:
            value = " ".join(raw_value.strip().split())
            key = value.casefold()
            if value and key not in seen:
                clean.append(value)
                seen.add(key)
        prompt = "Greek business vocabulary and names: " + ", ".join(clean)
        prompt = prompt[:max_characters]
        version = hashlib.sha256(prompt.encode("utf-8")).hexdigest()[:16]
        return PromptPlan(text=prompt, version=version)


def build_vocabulary_prompt(
    values: list[str] | tuple[str, ...],
    *,
    max_characters: int = 4000,
) -> tuple[str, str]:
    """Backward-compatible tuple API used by existing callers and tests."""

    prompt = LegacyVocabularyPromptBuilder().build(
        values,
        max_characters=max_characters,
    )
    return prompt.text, prompt.version
