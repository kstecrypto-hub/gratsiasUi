# Transcription Pipeline V2 foundation handoff

This document is the implementation handoff for the persistence and architecture
foundation completed before Pipeline V2 audio behavior is activated. Prompt 4
adds an explicitly selectable `pipeline-v2` topology flow. Prompt 5 adds
deterministic local speech segmentation for its separated operator and dual
tracks. Prompt 6 adds the versioned `greek-callcenter-v2` prompt, ranked
current-call vocabulary, and isolated same-track context chaining for those
standard-transcription modes. Prompt 7 adds real logprob signals and one bounded
light-normalized retry for low-confidence standard V2 chunks. Ordinary work
still defaults to `legacy-v1` until the later evaluation and activation gate.

## 1. Current completed foundation

- Alembic revision `f3b9c7d1a620` adds the Pipeline V2 persistence contract,
  deterministic legacy backfill, current-transcript protection, job-result
  provenance, and the `transcription_attempts` table.
- Alembic revision `a91d4e7c2b30` rejects multi-hop supersession cycles and
  logical-target boundary violations at the PostgreSQL database boundary.
- Alembic revision `d62c8f3a1b40` adds nullable safe prompt-template and
  vocabulary-hash metadata to transcripts without storing prompt bodies.
- Alembic revision `e74f9b2c6d10` adds a partial unique index that prevents two
  selected attempts for one transcript, track, and chunk.
- Completed calls have a distinct reprocess API. Reprocessing creates separate
  work, retains the old current transcript until the replacement succeeds, and
  records an audit event.
- Versioned idempotency covers the recording, operator target, model, language,
  prompt and vocabulary identity, transcription mode, pipeline version, and
  pipeline configuration.
- The transcription architecture is split into typed planning, quality,
  segmentation, prompting, client, confidence, merge, and orchestration
  boundaries.
- `greek-callcenter-v2` is active for explicitly requested V2
  operator-channel and dual-channel work. Its immutable manifest is built before
  transcript identity, forces request language `el`, ranks only current-analysis
  vocabulary, and renders bounded previous context independently for each track.
- Standard V2 requests obtain real model logprobs, apply the centralized
  uncalibrated policy, and create at most one light-normalized retry only after
  an explicit low-confidence decision. SDK transport retries are disabled for
  that branch so every paid upload remains bounded and auditable.
- `backend/app/workers/pipeline.py` uses
  `TranscriptionOrchestrator` for new provider transcription work while retaining
  all job lifecycle and persistence responsibilities.
- Celery redelivery, stale discovery, and Redis lease loss have explicit
  retry-or-abort behavior. After the recording lease is acquired, its ownership
  fences stage commits and post-side-effect persistence for download, inspection,
  preprocessing, transcript reuse/create/retry/clone, provider results, keyword
  completion, ordinary failure or connection settlement, and post-transcription
  shared source-audio deletion. Provider calls additionally require the capacity
  lease. Lease loss exits through the retryable busy path without mutating
  successor work, and a crashed worker cannot be acknowledged as successful
  while leaving its job active.
- Ordinary discovery and retention take the same deterministic recording-row
  locks before creating work or deleting history. Retention then locks associated
  job rows and rechecks live ordinary and reprocess items before deleting
  transcript history or audio. Transcript-row locks serialize this deletion with
  explicit reprocess creation. Reprocess creation first rejects an existing
  active job and performs an unlocked candidate read; it then locks the candidate
  recording, re-reads and locks its transcript, and finally locks the call before
  creating the new job. After its completion commit, post-transcription source
  deletion revalidates the Redis lease, takes and retains the shared recording
  row lock followed by all associated job-row locks, refreshes the row, rechecks
  every other non-final item, performs the physical deletion, revalidates the
  lease, and commits the recording update.
  Recording-backed worker state transitions and failure settlement use the same
  `Recording` then `ProcessingJob` then `ProcessingJobItem` database lock order;
  transcript identity, provider-result, and keyword transactions take the
  recording fence first. Discovery, retry, reprocess, and cleanup transitions
  therefore cannot cross the protection check.
- Equivalence tests freeze the legacy model, language, prompt, provider request
  parameters, chunking, request count, cancellation, error translation,
  persistence, and keyword behavior.

Local WebRTC VAD is active only for `pipeline-v2` operator-channel and
dual-channel tracks. The V2 contextual prompt is active for the same two
standard-transcription modes. Confidence retry is active for those standard
modes; mono refinement and manual speaker assignment remain unimplemented.
Pipeline V2 behavior remains available only when `pipeline-v2` is requested
explicitly.

## 2. Database contract

The contract is defined in `backend/app/models/entities.py`,
`backend/app/models/enums.py`, and migrations
`backend/migrations/versions/f3b9c7d1a620_add_transcription_pipeline_v2_foundation.py`
and
`backend/migrations/versions/a91d4e7c2b30_enforce_transcript_supersession_integrity.py`,
followed by
`backend/migrations/versions/d62c8f3a1b40_add_safe_transcript_prompt_metadata.py`
and
`backend/migrations/versions/e74f9b2c6d10_enforce_one_selected_transcription_attempt.py`.

### Transcripts and job provenance

- `Transcript` carries `transcription_mode`,
  `speaker_attribution_status`, `pipeline_version`, `pipeline_config_hash`,
  `preprocessing_profile`, `quality_summary`, `prompt_template_version`,
  `vocabulary_hash`, `supersedes_transcript_id`, and `is_current`.
- Legacy rows are backfilled as `legacy-v1`. Diarized rows use anonymous
  attribution; safely attributed non-diarized rows use PBX-confirmed attribution;
  unresolved non-diarized rows retain channel-unknown attribution.
- PostgreSQL and SQLite partial unique indexes permit one current attributed
  transcript per `(recording_id, operator_id)` and one current unattributed
  transcript per `recording_id`.
- `supersedes_transcript_id` uses `SET NULL` deletion. The model rejects direct
  self-supersession; the worker validates and locks the complete ancestor chain;
  and PostgreSQL rejects multi-hop cycles, call/recording boundary changes, and
  invalid operator-target changes. Anonymous history may become attributed from
  PBX topology, but attributed history cannot move to another operator or back
  to anonymous.
- `ProcessingJobItem.requested_pipeline_version` carries explicit reprocess
  intent. `ProcessingJobItem.result_transcript_id` binds a completed item to the
  transcript that produced its result, with `SET NULL` retention behavior.
- A separate partial unique index on processing jobs enforces the global
  one-active-analysis invariant.

### Segments and attempts

- `TranscriptSegment` has nullable channel, track, chunk, confidence-evidence,
  quality-flag, and audio-variant fields. The legacy path does not invent values
  for unavailable evidence.
- Prompt 7 writes one raw attempt and, only after a low-confidence decision, one
  normalized attempt for each V2 standard speech chunk. Rows retain absolute
  track/chunk bounds, model, versioned audio variant, rendered-prompt SHA-256,
  response text, real mean/ratio metrics when available, selected state, safe
  usage, and provider-completion time.
- Code validates one or two attempts with exactly one selected before adding any
  rows. A PostgreSQL/SQLite partial unique index independently prevents two
  selected rows for one transcript/track/chunk. Only the selected response
  becomes final segment text.
- Cancellation, normalization failure, and second-call failure carry completed
  provider evidence into failure settlement. Attempt rows and the failed state
  commit together; the previous current transcript is never demoted. A later
  retry uses a fresh transcript row and retains the failed run as immutable
  attempt history.
- Standard V2 checksum cloning is disabled until a clone can retain durable
  attempt provenance. Exact completed-transcript idempotent reuse remains
  enabled and never duplicates attempts.
- Attempts cascade when their transcript is deleted. The schema stores no API
  keys, credentials, raw audio paths, raw vocabulary, or full prompt bodies.
- Transcript deletion safely cascades through segments, keyword matches, and
  attempts. Job-result bindings and surviving superseding children are retained
  with their deleted transcript references set to null.
- The legacy and V2 mono paths intentionally do not write
  `TranscriptionAttempt` rows yet.

Database enum values are persisted using the repository's enum-column convention,
which stores enum member names. Later migrations must use that convention rather
than assuming lowercase database values.

## 3. Orchestrator contract

`backend/app/services/transcription/orchestrator.py` is the production
transcription-flow boundary.

Its input is:

- the original recording for `pipeline-v2`, or the worker-prepared legacy audio
  `source_path` for `legacy-v1`;
- `AudioInfo` from the existing inspection stage;
- immutable `TranscriptionContext`, including language, vocabulary, attribution
  context, temporary-directory ownership, and file-registration callback; and
- an optional asynchronous cancellation check.

Its output is `OrchestratedTranscriptionResult`, containing typed track results,
ordered hypotheses, provider usage, model, language, prompt identity, processing
duration, and explicit confidence availability.

The orchestrator owns:

- invoking the planner, quality adapter, segmenter, prompt builder, OpenAI client,
  confidence analyzer, and merge contract;
- selecting the local speech segmenter for V2 operator/dual tracks and the
  frozen fixed segmenter for legacy and V2 mono tracks;
- rendering `greek-callcenter-v2` independently for each V2 standard track and
  submitting that track's chunks sequentially with bounded context from only its
  previously accepted hypothesis;
- selecting isolated or diarized provider work from each planned track; and
- returning typed results without database access.

`backend/app/services/transcription/client.py` remains the only OpenAI
transcription API adapter. It owns request parameters, provider-error translation,
response parsing, and usage extraction. The settings and health APIs may call its
`test_connection()` method for credential checks; they are not transcription-flow
bypasses.

`backend/app/workers/pipeline.py` continues to own Redis locks, job and item state,
Yeastar download, topology evidence, prepared legacy WAV creation, versioned
identity, transcript and segment persistence, keyword matching, cleanup,
retention, and atomic replacement activation.

Low-level audio and transcription modules must not import SQLAlchemy sessions,
ORM models, or application repositories.

## 4. Current legacy adapters

| Contract | Current adapter | Frozen behavior |
| --- | --- | --- |
| Planning | `LegacyAudioPlanner` / `TopologyAudioPlanner` | Legacy describes the worker-selected track; V2 selects operator-channel, dual-channel, or mono-diarization from sanitized file/PBX evidence |
| Quality | `LegacyPassThroughQualityProcessor` / `LightNormalizedRetryQualityProcessor` | Legacy is unchanged; V2 standard retry uses the versioned 100 Hz/3400 Hz/loudnorm lossless-output profile |
| Segmentation | `LocalSpeechAudioSegmenter` / `LegacyFixedAudioSegmenter` | V2 operator/dual tracks use deterministic local WebRTC speech regions; V2 mono and all legacy work retain a single diarized upload under the limit, 480-second diarized chunks, or 15-second isolated chunks |
| Prompting | `LegacyVocabularyPromptBuilder` / `V2GreekPromptBuilder` | Legacy reproduces the existing English-prefixed vocabulary prompt and truncated SHA-256 version exactly; V2 operator/dual tracks use immutable role-aware `greek-callcenter-v2` manifests, canonical full SHA-256 identities, and bounded same-track context |
| Provider API | `OpenAITranscriptionClient` | Legacy/diarized requests are frozen; V2 standard requests use supported real logprobs and disable hidden SDK retries |
| Confidence | `LogprobConfidenceAnalyzer` / `UnavailableConfidenceAnalyzer` | V2 standard aggregates selected real logprobs under `v2-logprob-uncalibrated-v1`; legacy and missing evidence remain explicitly unavailable |
| Merge | `merge_track_results` | Preserves legacy order and deterministically merges V2 tracks by start, end, channel, and chunk without clipping or deduplication |
| Orchestration | `TranscriptionOrchestrator` | Routes the exact planned mode to the matching segmenter and coordinates the adapters without changing business lifecycle or persistence |

## 5. Invariants later phases must preserve

1. A completed current transcript remains readable until a fully persisted
   replacement succeeds.
2. A failed, cancelled, duplicate, or stale replacement cannot demote the
   previous current transcript.
3. The database remains the final guard against duplicate active jobs and
   duplicate current transcripts, including nullable operator targets.
4. Persistent identity changes whenever pipeline version, configuration,
   transcription mode, model, language, prompt template, vocabulary, recording
   checksum, recording ID, or operator target changes.
5. Legacy execution identity includes the effective selected channel, prepared
   audio profile, upload limit, fixed chunk policy, request format, chunking
   strategy, and prompt-template version. Stable inputs retain stable hashes and
   request behavior.
6. Provider transcription calls go through
   `backend/app/services/transcription/client.py`; production transcription flow
   goes through `backend/app/services/transcription/orchestrator.py`.
7. The worker retains locks, transactions, status transitions, persistence,
   keyword matching, and cleanup. Low-level modules remain ORM-free.
8. Cancellation is checked during local speech analysis, around every speech
   chunk write, and before each provider upload. Safe provider error messages and
   categories remain unchanged.
9. After acquiring a recording lease and before releasing it, a worker must
   revalidate it around every ordinary processing commit and before persisting
   the result of an external or CPU side effect, including ordinary failure,
   connection settlement, and shared source-audio deletion. Provider uploads and
   provider-result persistence must own and revalidate both the recording lease
   and transcription-capacity lease. Lease loss is retryable and cannot mutate
   successor work. Replacement activation must revalidate after any blocking
   transcript locks and before its completion commit. Recording-backed
   transitions must acquire database locks in `Recording`, `ProcessingJob`,
   `ProcessingJobItem` order before any transcript or recording mutation.
   Database-authoritative cancellation settlement and final job aggregation
   remain separate post-processing responsibilities.
10. Discovery must retain its Redis lease, persist a `SyncRun` heartbeat, and
    lock recording rows in deterministic ID order before committing progress;
    lease loss is retryable and cannot mutate job state.
11. Temporary artifacts created below the worker boundary are registered through
    the cleanup callback on success and failure.
12. Full prompt bodies and plaintext API keys, credentials, or other secrets are
    never written to transcript, segment, attempt, or audit records, and are never
    logged. UI-managed credentials may exist only in the dedicated encrypted Redis
    configuration state.
13. Existing job-to-result transcript bindings remain authoritative for
    historical result searches until retention deliberately deletes that
    transcript.
14. Retention cannot delete transcript history or audio for a recording referenced
    by a non-final item in a non-final job. It must share discovery's deterministic
    recording-row lock protocol, lock associated job rows before the final
    protection check, lock transcript rows to serialize explicit reprocessing,
    and retain those locks through deletion commit. Post-transcription source
    cleanup must hold that same recording lock and all associated job locks while
    refreshing state, rechecking other non-final items, deleting the file, and
    committing.
15. No later phase may silently activate new audio or model behavior for
    `legacy-v1`; activation requires a distinct versioned identity and rollback
    path.
16. V2 standard requests use language `el`. Their prompt identity and aggregate
    vocabulary hash participate in pipeline configuration and transcript
    idempotency, while the actual rendered prompt hash is retained only as safe
    per-attempt evidence.
17. Previous accepted text may flow only to the next chunk in the same track.
    The first chunk has no context, context is capped at 500 characters, and no
    caller/callee or Channel A/Channel B text may cross into its sibling track.
18. A V2 standard chunk has one raw attempt and at most one normalized attempt.
    Retry requires valid low-confidence metrics and a cancellation check; SDK
    transport retries remain disabled. Exactly one attempt is selected and only
    its text may reach a final segment or the next prompt context.

## 6. Subsequent implementation sequence and module map

Steps 1 through 4 are complete. Remaining implementation proceeds from mono
refinement in step 5; the completed steps remain listed to preserve the module
ownership map.

1. **Topology**

   Modify `backend/app/services/transcription/types.py`,
   `backend/app/services/transcription/planning.py`, and
   `backend/app/services/transcription/orchestrator.py`. Modify
   `backend/app/workers/pipeline.py` only to pass additional PBX evidence and to
   version topology configuration in persistent identity. Do not move Yeastar
   access or ORM state into the planner.

2. **Segmentation**

   Modify `backend/app/services/audio/segmentation.py`,
   `backend/app/services/transcription/types.py`, and
   `backend/app/services/transcription/orchestrator.py`. Extend
   `backend/app/services/audio/processor.py` only for required audio primitives,
   and update `backend/app/workers/pipeline.py` only for configuration identity
   and temporary-file ownership. Keep the legacy fixed segmenter unchanged.

3. **Prompts**

   Completed by Prompt 6 in
   `backend/app/services/transcription/prompt.py`,
   `backend/app/services/transcription/orchestrator.py`,
   `backend/app/services/transcription/client.py`, and
   `backend/app/workers/pipeline.py`. Template, vocabulary, rendered-prompt, and
   pipeline identities are versioned together without persisting prompt bodies.

4. **Confidence**

   Completed by Prompt 7 in
   `backend/app/services/transcription/types.py`,
   `backend/app/services/transcription/confidence.py`,
   `backend/app/services/transcription/client.py`,
   `backend/app/services/audio/quality.py`,
   `backend/app/services/audio/processor.py`,
   `backend/app/services/transcription/orchestrator.py`, and
   `backend/app/workers/pipeline.py`. Attempt fields are persisted
   transactionally, and a narrow migration enforces at most one selected row.

5. **Mono refinement**

   Modify `backend/app/services/transcription/planning.py`,
   `backend/app/services/transcription/orchestrator.py`,
   `backend/app/services/transcription/merge.py`, and the relevant
   `backend/app/services/audio/quality.py` or segmentation adapter. Persist
   attempt selection through `backend/app/workers/pipeline.py` without weakening
   replacement atomicity.

6. **Manual assignment**

   Add an audited assignment API through `backend/app/api/results.py` and
   `backend/app/schemas/results.py`, using the existing manual attribution enum.
   Modify `backend/app/models/entities.py` and add a new migration only if an
   immutable assignment-history contract is required. Update
   `frontend/app/(app)/calls/[id]/page.tsx` and related result views without
   changing acoustic transcription.

7. **Evaluation and activation**

   Add evaluation fixtures and activation gates before changing defaults. Runtime
   configuration belongs in `backend/app/core/config.py`,
   `backend/app/services/application_settings.py`,
   `backend/app/schemas/settings.py`, and `backend/app/api/settings.py`.
   Version selection belongs at the orchestrator and job/reprocess boundary.
   Preserve an explicit `legacy-v1` rollback option and compare quality, cost,
   request count, latency, and failure rate before activation.

Every step must extend `backend/tests/test_transcription_v2_foundation.py` where
the foundation contract changes and add focused behavioral tests beside the
modified module.

## 7. Speaker identity and security prohibition

The prohibition is exact:

- No operator voice samples are permitted.
- No voiceprints are permitted.
- No speaker recognition is permitted.
- No speaker biometrics are permitted.

The application must not collect enrollment recordings, create voice embeddings,
compare voices to known operators, or infer a real identity from acoustic
similarity. Permitted attribution evidence is limited to PBX metadata, safe
channel topology, anonymous diarization labels, and explicit audited manual
assignment.

## 8. Known technical debt

- `AudioInfo` describes the downloaded recording while the orchestrator may
  receive or materialize a prepared WAV. Local speech segmentation validates and
  measures each prepared track directly. Frozen legacy segmentation still uses
  the original duration. A future per-track quality contract must carry prepared
  metadata before making broader codec, channel, size, or checksum decisions.
- The worker and orchestrator share the same immutable V2 prompt manifest. The
  worker uses its canonical identity before transcript creation, and the
  orchestrator renders each request from that exact manifest.
- Before this audit, the keyword-variant relationship had no explicit SQL order.
  The legacy prompt now uses deterministic `(created_at, id)` order, preserving
  entry order where that history exists; an older transient database plan cannot
  be reconstructed after the fact.
- V2 standard transcription now populates real per-chunk logprob evidence,
  selected quality flags, and one or two auditable `TranscriptionAttempt` rows.
  The provisional policy remains deliberately uncalibrated until Prompt 10.
- The multi-track merge contract is active for explicitly selected
  `pipeline-v2` work. It preserves overlaps and orders by start, end, channel,
  and chunk.
- Historical pre-migration job provenance was reconstructed deterministically and
  cannot recover information that never existed.
- Transcript retention can delete inactive historical transcript content and
  clear `result_transcript_id` through `SET NULL`; job metadata remains.
  Recording-row locks shared with ordinary discovery, followed by associated
  job-row locks and a live-item recheck, protect all history for active ordinary
  or reprocess work.
- SQLite development tests rely on the worker's complete lineage validation for
  multi-hop supersession. Production PostgreSQL additionally enforces the same
  rule with a trigger.
- Worker items and initial Redis leases use a two-hour stale window. Each
  successful ownership refresh replaces the Redis TTL with one hour, while
  OpenAI requests currently time out after 120 seconds. A future change that
  permits a single uninterruptible operation longer than the refreshed one-hour
  TTL must add an independent heartbeat before increasing that operation's
  timeout.
- Backward-compatible direct client methods remain public. New production
  transcription callers must use the orchestrator; settings and health
  connection tests are the only intended exceptions.

## 9. Rollback considerations

1. Keep `legacy-v1`, `LegacyAudioPlanner`,
   `LegacyFixedAudioSegmenter`, `LegacyVocabularyPromptBuilder`,
   `LegacyPassThroughQualityProcessor`, and
   `UnavailableConfidenceAnalyzer` available until a later pipeline has passed
   evaluation and a defined support window.
2. Roll back runtime activation by selecting `legacy-v1`; do not rewrite existing
   transcript history or reuse a V2 identity for legacy behavior.
3. Never demote or delete the current transcript as part of rollback. Queue a
   separately versioned replacement and use the same atomic activation path.
4. Do not change published legacy prompt or pipeline hash constants. A behavior
   change requires a new pipeline configuration identity.
5. Downgrade `e74f9b2c6d10` before `d62c8f3a1b40`, then
   `a91d4e7c2b30`, then `f3b9c7d1a620`. These remove the selected-attempt index,
   safe prompt metadata, PostgreSQL supersession trigger/function, and finally
   the remaining V2 fields, attempts, result bindings, and current-transcript
   indexes. Do not downgrade after relying on V2 data without a verified backup
   and an explicit data-loss plan.
6. If a worker release is rolled back while newer jobs are queued, reject
   unsupported requested pipeline versions safely rather than processing them as
   legacy.
7. Preserve old current transcripts and job audit records until the rolled-back
   deployment is verified healthy.

## 10. Prompt 4 audio-topology behavior

`pipeline-v2` computes one immutable `AudioPlan` before transcript identity is
created and executes that same plan:

1. A two-channel file plus PBX-confirmed separated recording and a safely proven
   operator side selects `operator_channel`. Only that channel is extracted,
   the operator ID/display name is attached, and attribution is
   `confirmed_by_pbx`.
2. A two-channel file plus PBX-confirmed separated recording with uncertain
   operator side selects `dual_channel`. Channels 0 and 1 are extracted and
   transcribed independently; this path never calls mono conversion. One-leg,
   non-queue, non-transferred PBX evidence may label the channels Caller/Callee.
   Otherwise they remain Channel A/Channel B. Segment operator IDs stay null and
   the item completes with an attribution warning.
3. Mono audio, multichannel audio, or stereo not confirmed as separated selects
   `mono_diarization` and retains the existing mono/diarization behavior.

Queue, transfer, multi-leg, ambiguous-side, and same-side-multiple-operator
evidence is never used to guess an operator channel. The stereo-capability cache
is scoped to a non-secret configuration fingerprint so one PBX cannot lend its
capability result to another.

Prompt 4 originally retained fixed-duration segmentation. Prompt 5 supersedes
that temporary rule for V2 operator and dual tracks only. V2 mono still uses one
diarized upload under the size limit or fixed 480-second chunks. Legacy behavior
is unchanged.

## 11. Prompt 5 deterministic speech segmentation

V2 `operator_channel` and `dual_channel` tracks use
`LocalSpeechAudioSegmenter`. The backend is the pinned
`webrtcvad-wheels==2.0.14` distribution, imported as `webrtcvad`, with
aggressiveness 2. There is no silent runtime fallback. Each track receives a
fresh classifier instance, so dual channels share neither VAD history nor
segmentation state.

The frozen analysis contract is 16 kHz, mono, uncompressed signed PCM16 WAV in
20 ms frames (320 samples). A start opens from a 300 ms rolling window when at
least 60 percent is voiced, and a region closes after 600 ms of continuous
silence. Regions receive 250 ms prefix and suffix padding, merge across gaps
strictly below 300 ms, and use 1.2 second minimum, 40 second target, and 60 second
maximum durations. All boundaries are calculated as integer sample indexes and
clamped to the real prepared-track duration.

For each qualifying non-speech run, long-region planning selects its
lowest-energy valid frame, with ties resolved by target distance and then the
earlier sample. Among those run candidates it prefers target distance, then
energy, then the earlier sample. Candidate scanning is bounded to the valid cut
window plus the frames needed to prove the silence run, so multi-hour planning
remains linear. If no safe cut exists, the segmenter hard-cuts at the target and
places an 800 ms prefix overlap on the following chunk.
`SpeechChunk.overlap_before_ms` carries that fact explicitly. Exact PCM sample
ranges are copied to atomic temporary WAV files; float-based FFmpeg seeking is
not used for chunk extraction.

Only adjacent chunks from the same track may remove a repeated hard-cut prefix.
The join requires exact, case-sensitive whitespace-token equality, at least two
tokens and eight characters, and considers at most 24 tokens. It performs no
fuzzy, normalized, non-adjacent, or cross-channel deduplication. The join-policy
version, backend version, complete segmentation configuration, analysis and
output formats, and upload limit are included in the Pipeline V2 runtime
identity.

Silence-only tracks produce no chunks and no provider request. Detected speech
that cannot be safely merged is retained. Cancellation is checked during long
analysis scans and around every chunk write; destinations and partial files are
registered before side effects and removed on cancellation or extraction
failure. V2 mono and all `legacy-v1` segmentation remain frozen on
`LegacyFixedAudioSegmenter`.

## 12. Prompt 6 Greek context and identity

V2 `operator_channel` and `dual_channel` tracks use the immutable
`greek-callcenter-v2` manifest. Its concise Greek instructions require
transcription of only audible speech, prohibit completing or inventing missing
words and facts, preserve names, company names, vehicle models, telephone
numbers, registration plates, and dates, retain English commercial and technical
terms without translation, and treat previous accepted text only as context that
must not be repeated unless audible. These requests use the configured standard
transcription model and force language `el`.

Vocabulary is limited to the current analysis and ranked as follows:

1. priority 100: the selected operator on a confirmed Operator track;
2. priority 95: the current caller/callee names on only their proven roles;
3. priority 90: the current call's caller and callee numbers;
4. priority 80: canonical keywords and ordered variants from selected
   categories;
5. priority 70: configured company vocabulary;
6. priority 60: the current call's queue or department; and
7. priority 40: fixed general dealership terminology.

Unlike the legacy vocabulary collector, this path does not load all operator
names. Unknown separated stereo remains Channel A/Channel B and does not claim
an Operator, Caller, or Callee role. The prompt policy normalizes whitespace,
discards empty and secret-like terms, deduplicates case-insensitively while
retaining the highest priority, applies a deterministic tie-break and order,
caps each term at 100 characters, and caps the combined vocabulary at 3,000
characters.

Each track is transcribed one speech chunk at a time. Chunk 1 has no previous
context. After exact hard-cut overlap handling, the last accepted hypothesis can
feed the next chunk on that same track; only its normalized final 500 characters
are rendered. Context state is created inside an individual track operation and
is reset before another track, which prevents caller/callee and Channel A/Channel
B context crossing.

Canonical JSON with sorted keys and compact separators feeds full SHA-256
identities. The template/renderer, explicit limits, role mapping, deterministic
ranked terms, and track manifests contribute to the aggregate vocabulary hash
and manifest prompt identity, so equivalent unordered inputs remain stable.
Template version, vocabulary hash, and manifest prompt identity participate in
the runtime configuration and transcript idempotency key. The full hash of each
actually rendered chunk prompt, including optional context, is carried as
per-attempt evidence.

Persistence is hash-only at the prompt boundary: transcript rows retain the
template version, vocabulary hash, and manifest prompt identity, while V2
standard attempt rows retain the rendered prompt hash and safe request metadata.
They do not retain prompt bodies, raw vocabulary, API credentials, raw audio
paths, or SDK response objects. Prompt 7 intentionally persists each attempt's
plain transcription response text for auditability.

The byte-identical legacy prompt and its hash remain unchanged. V2
`mono_diarization` remains on the existing diarized, prompt-free path and does
not use contextual chaining. Confidence is still reported as unavailable for
legacy and mono paths.

## 13. Prompt 7 logprob confidence and bounded retry

The pinned OpenAI Python SDK is `2.45.0`. Standard V2 calls using the configured
`gpt-4o-transcribe` model request `include=["logprobs"]` with JSON output,
language `el`, the existing rendered Greek prompt, and temperature `0.0`.
The V2-only request client uses `max_retries=0`; legacy and diarized request
shapes and SDK retry behavior remain unchanged.

`v2-logprob-uncalibrated-v1` calculates mean and minimum logprob, the ratio of
tokens below `-1.0`, token count, and geometric mean token probability as an
internal signal. A chunk is provisionally low when mean logprob is below
`-0.75` or the low-token ratio is above `0.15`. The geometric signal is not an
accuracy estimate. Missing, malformed, NaN, infinite, empty-token, and
mathematically invalid positive logprobs are ignored; no valid evidence means
unavailable and cannot trigger retry.

The first attempt uses the minimally processed lossless PCM16 speech chunk. A
low decision first checks cancellation, then creates
`ffmpeg-light-normalized-v1`: high-pass 100 Hz, low-pass 3400 Hz, loudness
normalization at `I=-23:LRA=7:TP=-2`, and lossless 16 kHz mono PCM16 output.
The retry file and its partial destination are registered before creation and
cleaned immediately after use, with worker cleanup retained as fallback.

Selection prefers an attempt with valid metrics, then higher mean logprob. Raw
wins an effective tie within `0.05`. A second attempt adds
`normalized_retry_used`; a selected low result adds `low_confidence`; and two
low attempts also add `both_attempts_low_confidence` and
`human_review_recommended`. Unavailable selected evidence uses
`logprobs_unavailable` without inventing a score. Only selected text advances
same-track context and becomes a final segment, while both attempt costs remain
in aggregate usage.

Completed raw evidence is also retained when cancellation, normalization, or
the second upload interrupts the chunk. The interrupted chunk selects the
strongest completed attempt for audit purposes, but no failed-run text becomes a
final segment. Failure state and partial attempt evidence are committed
atomically, and retries preserve that failed transcript as separate history.

The runtime configuration hash includes the logprob request/transport contract,
all provisional thresholds, tie tolerance, two-attempt cap, raw variant, and
exact normalized FFmpeg profile. Completed Prompt 6 transcripts therefore
cannot be silently reused as Prompt 7 results.
