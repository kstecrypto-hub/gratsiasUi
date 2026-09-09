# Transcription Pipeline V2 architecture

## Scope of this foundation

This architecture introduces typed boundaries around the existing transcription
flow. It preserves the production legacy behavior and adds an explicitly
selectable `pipeline-v2` topology path with operator-channel, dual-channel, and
mono-diarization plans. V2 operator and dual tracks use deterministic local
speech segmentation and the versioned `greek-callcenter-v2` contextual prompt.
Those standard V2 tracks now use real model logprobs and at most one versioned
light-normalized retry. It does not add two-pass mono transcription or manual
speaker assignment.

## Ownership and data flow

1. `app/workers/pipeline.py` owns provider download, PBX topology evidence,
   operator-channel selection, legacy WAV preparation, Redis leases, deterministic
   recording/job row-lock protocols, job and item state, transcript identity,
   database transactions, persistence, keyword matching, cleanup, and retention.
2. `app/services/transcription/orchestrator.py` receives the already prepared
   legacy WAV or the original V2 recording and immutable context. It computes a
   plan before identity creation, materializes that exact plan, coordinates
   pass-through track handling, mode-aware segmentation, prompt preparation,
   the OpenAI adapter, logprob decisions, bounded retry audio, and result
   merging.
3. `app/services/transcription/client.py` remains the only OpenAI transcription
   API adapter. It owns request parameters, provider-error translation, usage
   extraction, and response-to-segment conversion.
4. The orchestrator returns `OrchestratedTranscriptionResult`. The worker maps
   that result to `Transcript` and `TranscriptSegment` rows and then runs keyword
   matching.

The low-level transcription and audio modules do not receive SQLAlchemy sessions
or ORM entities. They operate on paths, audio metadata, typed context, and typed
results.

## Module contracts

| Module | Current behavior | Future implementation prompt |
| --- | --- | --- |
| `transcription/types.py` | Typed tracks, chunks including hard-cut overlap metadata, hypotheses, logprobs, auditable attempts, track results, and merged results | Later prompts extend values without changing the worker boundary |
| `transcription/planning.py` | Legacy and PBX-evidence topology planners are available; V2 preserves uncertain separated stereo as two tracks | Later phases may extend quality and mono policies without changing the evidence boundary |
| `transcription/prompt.py` | `LegacyVocabularyPromptBuilder` reproduces the existing prompt and hash exactly; `V2GreekPromptBuilder` creates immutable, role-aware `greek-callcenter-v2` manifests and bounded per-chunk prompt plans | Confidence and later prompt revisions must use a new explicit template/renderer identity |
| `transcription/confidence.py` | Parses finite real logprobs and applies one centralized, explicitly uncalibrated policy; legacy/unsupported evidence remains unavailable | Prompt 10 must calibrate or replace provisional values before activation |
| `transcription/merge.py` | Legacy order is retained; real V2 tracks merge by start, end, channel, and chunk while preserving overlap | Later phases may add evidence-aware selection without cross-channel deduplication |
| `transcription/orchestrator.py` | Coordinates exact-plan V2 track materialization, mode-aware segmentation, raw/logprob work, one selective normalized retry, deterministic selection, and exact within-track overlap joining | Later prompts must not move worker persistence into this layer |
| `audio/segmentation.py` | `LocalSpeechAudioSegmenter` performs integer-sample WebRTC speech planning for V2 operator/dual tracks; `LegacyFixedAudioSegmenter` preserves one-upload, 15-second, and 480-second rules elsewhere | Later audio phases must keep both identities explicit |
| `audio/quality.py` | Legacy remains pass-through; V2 standard retry uses one centralized `ffmpeg-light-normalized-v1` profile with lossless PCM16 output | Later quality profiles require new identities and evaluation |

## Legacy adapters and invariants

The following legacy and V2 mono behavior is intentionally frozen until a later
phase replaces the corresponding adapter:

- The worker decides whether a safely attributed stereo channel can be extracted.
  Otherwise it creates the same mono fallback used for diarization.
- Prepared audio remains 16 kHz, mono, PCM s16le.
- Isolated operator audio is split into contiguous 15-second chunks.
- Diarized audio is uploaded once when it fits the configured upload limit;
  otherwise it is split into contiguous 480-second chunks.
- Legacy and V2 mono make exactly one application-level OpenAI request per
  chunk. V2 standard tracks alone may make a second request after a real
  logprob-based low-confidence decision.
- A legacy isolated request uses the configured transcription model and
  language, exact legacy vocabulary prompt, and JSON response format.
- The diarized request uses the configured diarization model and language,
  diarized JSON response format, and automatic provider chunking, without a
  prompt.
- Cancellation is checked immediately before each upload, as in the legacy
  client. Provider-error categories and messages are not wrapped or changed.
- Segment order, timestamps, labels, usage aggregation, and transcript text
  joining remain unchanged.
- The transcript row is committed before segmentation and provider calls.
  Transcript content and segments are committed atomically afterward.
- Keyword matching remains in the worker and runs only after transcript
  persistence.
- Chunk paths created by the orchestrator are registered with the worker so the
  worker retains cleanup ownership on both success and failure.

`audio_info` currently describes the original downloaded recording while
`source_path` is the worker-prepared legacy WAV. The legacy segmenter uses only
the original duration, which preserves current behavior. A future topology or
quality adapter must inspect and carry metadata for each prepared track before
making codec, channel, size, or checksum decisions. Any future quality adapter
that creates files must also register those artifacts with the worker cleanup
contract.

## Persistence boundary

The orchestrator does not select or mutate current transcripts, create attempts,
write segments, activate replacements, or decide retention. Pipeline version,
configuration hash, idempotency, retries, replacement ownership, and atomic
current-transcript swaps remain worker and database responsibilities.

Prompt 6 populates the safe V2 prompt metadata fields after deriving one
immutable manifest before transcript identity creation. Standard V2 transcript
rows store the prompt template version, aggregate vocabulary hash, and manifest
prompt identity. Each provider attempt stores only the SHA-256 hash of its
rendered chunk prompt alongside safe request evidence. Prompt 7 also stores each
attempt's provider response text, real mean/ratio metrics when available,
variant, usage, selection, and completion time. Full prompt bodies and raw
ranked-vocabulary values are not persisted.

If cancellation, normalization, or a later provider call interrupts a standard
V2 run after an upload completed, the orchestrator raises typed partial evidence.
The worker writes those one-or-two-attempt groups and the failed state in one
transaction without activating a replacement. A later retry archives the failed
transcript's execution key and creates a fresh row, preserving the paid attempt
history instead of deleting or mixing runs.

Prompt 4 now populates channel, track, chunk, speaker-source, label, and
per-segment operator metadata for `pipeline-v2`. Dual-channel segments retain
null operator IDs.

## Prompt 5 segmentation boundary

V2 operator and dual tracks use independent `LocalSpeechAudioSegmenter` calls
over prepared 16 kHz mono PCM16 WAVs. The pinned backend is
`webrtcvad-wheels==2.0.14`, mode 2, with fresh classifier state per track.
Decisions use 20 ms frames; region and chunk boundaries use integer samples.
Silence-only tracks create no chunks or provider requests. When all planned
tracks are silent, the provider client is not opened.

The frozen boundary configuration is a 300 ms start window at 60 percent voiced,
600 ms end silence, 250 ms prefix and suffix padding, 300 ms merge gap, 1.2
second minimum, 40 second target, 60 second maximum, and 800 ms hard-cut
overlap. Safe splits choose the lowest-energy valid frame in each qualifying
non-speech run, then rank those candidates by target distance, energy, and
earlier sample; planning scans bounded windows and yields for cancellation
between split iterations. Forced splits carry explicit overlap metadata, and
only an exact case-sensitive token
suffix/prefix on adjacent chunks in one track can be removed. Cross-channel and
fuzzy deduplication are prohibited.

V2 mono-diarization continues to use one upload below the configured limit or
fixed 480-second chunks. All `legacy-v1` one-upload, 15-second, and 480-second
behavior remains on `LegacyFixedAudioSegmenter`. The full local segmentation
configuration, backend, formats, overlap-join policy, and upload limit are part
of the V2 runtime identity.

## Prompt 6 contextual prompt boundary

V2 `operator_channel` and `dual_channel` requests use
`greek-callcenter-v2`, rendered in Greek. The fixed instructions say that the
audio is a Greek telephone conversation; only actually audible speech may be
transcribed; missing words or facts must not be completed or invented; names,
company names, vehicle models, phone numbers, registration plates, and dates
must be preserved; English commercial and technical terms must not be
translated; and previous text is context only and must not be repeated unless it
is audible again. The request language is always `el`, while the configured
standard transcription model remains unchanged.

The worker builds ranked vocabulary only from the current analysis context:

1. priority 100: the selected operator, scoped only to a confirmed Operator
   track;
2. priority 95: the current caller or callee name, scoped only to the proven
   matching role;
3. priority 90: current-call caller and callee numbers;
4. priority 80: keywords and variants from the job's selected categories;
5. priority 70: configured company vocabulary;
6. priority 60: the current call's queue or department name; and
7. priority 40: the fixed general dealership vocabulary.

No query for all operator names participates in the V2 manifest. Unknown dual
channels are labelled Channel A and Channel B and receive no Operator,
Caller, or Callee-scoped terms. Values are whitespace-normalized, empty and
secret-like values are discarded, duplicates are compared case-insensitively
with the highest-priority value retained, and final ordering is deterministic.
Each term is capped at 100 characters and the rendered vocabulary section at
3,000 characters.

Chunks are submitted sequentially within each track. The first chunk has no
previous-text section. After a chunk is accepted and exact hard-cut overlap is
resolved, its last accepted hypothesis may become context for the next chunk on
that same track. Context is whitespace-normalized and limited to its last 500
characters. Each track invocation has independent context state, so a dual
channel never receives text from its sibling channel.

Template, renderer, limits, role mapping, ranked terms, and track identities use
canonical JSON encoding and full SHA-256 identities. Equivalent vocabulary in a
different insertion order produces the same hashes. The manifest prompt identity
and aggregate vocabulary hash participate in the V2 runtime configuration and
transcript idempotency key. Each rendered prompt, including its optional
same-track context, receives a separate full SHA-256 prompt hash recorded as
safe attempt evidence. Prompt text, raw vocabulary, API credentials, and raw
audio paths are not written to prompt metadata or attempt rows.

`legacy-v1` continues to use the byte-identical
`LegacyVocabularyPromptBuilder`. V2 `mono_diarization` continues to use the
existing diarized request without a prompt and does not use contextual chaining.
Confidence remains explicitly unavailable for both paths.

## Prompt 7 logprob and selective-retry boundary

V2 `operator_channel` and `dual_channel` requests use the installed OpenAI SDK's
real `logprobs` response field with `response_format="json"`, language `el`, the
same rendered Greek prompt, and temperature `0.0`. SDK transport retries are
disabled for these uploads so the hard cap of two auditable provider requests
per chunk cannot be exceeded. Legacy and diarized requests retain their original
request parameters and retry behavior.

The provisional `v2-logprob-uncalibrated-v1` policy uses a mean-logprob cutoff
of `-0.75`, a per-token low cutoff of `-1.0`, a low-token-ratio cutoff of
`0.15`, and a raw-favoring mean tie tolerance of `0.05`. These are internal
signals, not accuracy percentages. Missing, malformed, NaN, or infinite values
produce unavailable evidence and never trigger a retry.

Variant A is the lossless PCM16 speech chunk produced by local segmentation.
Variant B is created only after an explicit low decision and uses high-pass
100 Hz, low-pass 3400 Hz, and `loudnorm=I=-23:LRA=7:TP=-2`, then writes 16 kHz
mono PCM16 WAV. The file is registered before creation, removed immediately
after the second request, and remains covered by worker-wide cleanup.

Selection prefers valid metrics over missing metrics, then higher mean logprob;
raw wins within the configured tie tolerance. Only selected text reaches the
final segment or next same-track context. Both provider costs remain in usage,
and one or two attempt rows are persisted atomically with exactly one selected.
The database additionally prevents two selected attempts for the same
transcript/track/chunk.

Cross-recording checksum cloning is disabled for standard V2 modes because a
segment-only clone would lose attempt provenance. Completed idempotent reuse of
the same transcript remains allowed and does not add attempt rows.

## Speaker identity prohibition

The pipeline must not collect or use operator voice samples, voiceprints,
speaker embeddings, or any other speaker biometric. Speaker attribution may use
PBX call metadata, channel topology, anonymous diarization labels, and explicit
manual assignment only.
