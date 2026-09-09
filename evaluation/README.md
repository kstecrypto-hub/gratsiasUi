# Local human reference labeling

One application UI exists: the existing Next.js application has production call
detail at `/calls/[id]` and human labeling at `/evaluation/[id]`. There is no
standalone labeling server.

## Enable locally

Evaluation defaults to `EVALUATION_UI_ENABLED=false`. Its API returns 404 when
disabled, and the authenticated shell hides evaluation navigation and its editor.
The backend flag is the single source of truth; no frontend rebuild is needed.

For the existing Docker setup, explicitly opt in:

```sh
docker compose -f docker-compose.yml -f docker-compose.evaluation.yml up --build
```

This mounts `./evaluation:/app/evaluation` into the existing backend and enables
its evaluation API. Ensure the backend user can write `references/` and
`previews/`. Normal Compose startup does not mount or enable evaluation.
Use the normal application sign-in. Keep this environment local.

For a native backend, set these locally and restart the backend:

```text
EVALUATION_UI_ENABLED=true
EVALUATION_ROOT=<absolute path to this evaluation directory>
EVALUATION_MANIFEST=manifest.local.jsonl
```

The native root defaults to this repository's evaluation directory. Docker uses
`/app/evaluation`. The existing frontend proxies `/api` to the existing FastAPI
service. No additional UI server is needed.

## Freeze selected calls

1. Copy `manifest.example.jsonl` to `manifest.local.jsonl`.
2. Replace example rows with selected DEV/TEST calls. Use opaque evaluation IDs;
   do not put telephone numbers in IDs.
3. Copy original recordings to `audio/`. Freeze PBX-only context in
   `context/<evaluation-id>.json`; `context.example.json` documents allowed
   fields. Use actual production metadata, never ASR guesses.
4. Set each manifest row's audio, context and intended reference paths.
   References need not exist. Opening a call begins with an empty reference
   and writes nothing until Save.
5. Sign in to the existing app and open `/evaluation`.

Paths use forward slashes, are relative to the evaluation root, and must stay
within `audio/`, `references/`, `context/`, or `previews/` as applicable.
Absolute paths, traversal, symlinks and Windows junctions are rejected. Do not
change a frozen recording or context after beginning its reference.

```json
{"evaluation_id":"call-001","audio":"audio/call-001.wav","reference":"references/call-001.json","context":"context/call-001.json","split":"dev","mode":"stereo"}
```

The historical manifest `id` is accepted too. The UI uses `evaluation_id` and
does not depend on production call IDs or audio surviving retention.

Duration and channel count come from frozen audio; mismatched manifest modes
are rejected. PCM WAV needs no external tool. MP3, M4A, Ogg/Opus, FLAC, AAC and
non-PCM WAV require FFmpeg/ffprobe on PATH; the existing Docker image includes
them. Separate channel previews live in `previews/`, and never change the
source. Audio endpoints support byte ranges and native scrubbing.

## Label and verify

Write exactly what is audible. Keep repetitions, false starts, spoken mistakes
and English brand/model terms. Do not rewrite grammar, summarize, improve
sentences or infer missing words. These routes never request/display production
or evaluated ASR text.

Segments have speaker, optional channel, start/end seconds, exact text, WER
exclusion and repeatable entity inputs. Press Add or Enter to add entity chips.
Canonical entities may differ from spoken words; enter only fully audible
values. Overlapping segments are allowed.

For unintelligible regions, select `exclude_from_wer`. Text may be empty or
an optional annotator note; no special token is required. Mark only genuinely
spoken keywords from the active production catalog. Selections remain separate
from production KeywordMatch rows, including if a selected keyword is later
deactivated.

Choose one human quality label:

- `clean`: clear speech with little or no background noise.
- `normal`: ordinary call audio with minor noise or compression.
- `noisy`: noticeable noise/distortion, most speech understandable.
- `very_noisy`: heavy noise/distortion, substantial speech hard to understand.

Stereo requires Channel A (0), Channel B (1), or Cannot establish (null).
`operator_channel_answered` distinguishes an explicit null from no answer.
It never changes production attribution. Historical quality names are cleared
for a fresh controlled human classification.

Save before **Mark fully reviewed**. Verification requires quality, at least one
segment, ordered times within duration, audible text unless excluded, and an
explicit stereo channel answer. Unsaved changes block verification. Drafts may
contain incomplete segments.

Verification stores `verification_status` and UTC `verified_at`, without a
human name/account ID. Subsequent saves return to `in_progress`, clear the
timestamp, and require another explicit review. Revision checks reject stale
saves/verification, and writes use atomic replacement and OS locks. On conflict,
keep/copy your work before Reload saved reference. Navigation links and page
close warn about unsaved edits.

## Data separation

The evaluation file service has no production ORM or transcription imports.
Only the active keyword catalog is read from production. Authentication and
CSRF reuse the existing application. Frozen PBX context is allowlisted and
customer caller/callee numbers are masked. File paths, credentials, production
transcripts and production matches are not returned.

Actual audio, references, context, local manifests, previews and reports remain
Git-ignored. Never commit customer transcripts, names, numbers, plates, models,
audio, credentials or raw provider payloads. Checked-in examples are blank
templates, not completed ground truth or benchmark results.

## Deterministic evaluation

Validate the manifest:

```sh
python evaluation/evaluate.py validate evaluation/manifest.local.jsonl
```

Set `EVALUATION_TRANSCRIBE_COMMAND` to a local transcription executable, then:

```sh
python evaluation/evaluate.py run evaluation/manifest.local.jsonl --output evaluation/reports
```

The executable receives **two arguments**: frozen audio path and pipeline
version (`legacy-v1` or `pipeline-v2`). It prints hypothesis JSON on stdout.
It is not given the reference path or human answers. Update older wrappers
that expected a third reference argument. Audio/reference paths resolve
relative to the manifest directory. References must be explicitly verified.

Text scoring uses human channel truth instead of assuming Channel A is the
operator. Speaker attribution is scored separately. Mono role metrics work
without channels. Excluded reference notes/text and matching hypothesis regions
are omitted from WER/CER.

**Exclusion boundary limitation:** a hypothesis segment crossing only part of an
excluded region cannot be scored fairly without word timestamps. Split it at
the exclusion boundaries before scoring. The runner reports an error instead
of guessing which words to drop. Exclusions respect channel identity.

No LLM grading or invented scores are used. Unverified references and failed
calls result in a nonzero exit code. Reports remain local and ignored.
Acceptance thresholds are in `criteria.json`.

## Validation

Backend: `python -m pytest` from `backend/`.

Frontend: `npm test`, `npx tsc --noEmit`, `npm run lint`, `npm run build`
from `frontend/`.

`backend/tests/test_evaluation_ui.py` covers authentication/CSRF, feature gates,
unchanged production table snapshots, no ASR prefill, frozen WAV/range/channel
playback, annotation persistence, overlaps, verification/invalidation, stale
writes, traversal/junction rejection and Git ignores.
`frontend/tests/e2e/evaluation.spec.ts` covers the shared shell and full
labeling/audio workflow. Existing production call tests remain in
`administrator.spec.ts`.
