# Transcription pipeline V2 rollout

## Dataset preparation

- Use only local, held-out labeled calls. The evaluation directory (`evaluation/`) owns the manifest, references, and reports.
- Copy `evaluation/manifest.example.jsonl` to a local file (for example `evaluation/manifest.local.jsonl`) and replace paths with local audio/reference files.
- Reference files must be JSON and contain speaker, channel, start, end, text, entities, and expected keywords.
- Keep `audio/` and `references/` local. They are git-ignored and must never be committed.

## Privacy handling

- Do not commit real customer audio or transcripts.
- Do not add operator voice samples, voice prints, or biometric material.
- Manifests and references may contain personal data (names, telephone numbers, licence plates, vehicle models). Keep them local.
- Generated detailed reports are git-ignored. Redact API keys, secrets, and storage paths from aggregate reports.
- Voice biometrics are prohibited. There is no enrollment, speaker identification, or emotion/health inference.

## Evaluation commands

```bash
python evaluation/evaluate.py validate evaluation/manifest.local.jsonl

EVALUATION_TRANSCRIBE_COMMAND=/path/to/local-transcriber \
  python evaluation/evaluate.py run evaluation/manifest.local.jsonl --output evaluation/reports

python evaluation/evaluate.py report evaluation/reports/per-call --output evaluation/reports
```

The local transcriber command receives `(audio_path, version, reference_path)` and writes hypothesis JSON to stdout.

## Metric definitions

- **WER**: word error rate after `evaluate.normalize_greek`.
- **CER**: character error rate after `evaluate.normalize_greek`.
- **Operator/customer WER**: WER restricted to segments attributed to the operator or customer channel/label.
- **Keyword recall**: `|detected ∩ expected| / |expected|`.
- **Keyword precision**: `|detected ∩ expected| / |detected|`.
- **Keyword false-positive rate**: `|detected - expected| / |detected|`.
- **Entity accuracy**: matched entities of a type divided by reference entities of that type.
- **Attribution accuracy**: fraction of reference segments whose speaker/channel matches the best time-overlapping hypothesis segment.
- **Review rate**: fraction of hypothesis segments flagged `human_review_recommended` or `needs_review`.
- **Processing success rate**: successful calls divided by all calls.
- **Latency per audio minute**: processing seconds divided by audio minutes.
- **Usage/cost**: summed only from provider-supplied values; no assumptions or synthetic estimates.

## Acceptance criteria

Versioned in `evaluation/criteria.json`. The minimum gate is:

1. No transcript/data-corruption regression.
2. Confirmed stereo operator-channel attribution remains 100% correct in the labeled set.
3. Overall WER improves by at least 10% relative against legacy.
4. No major quality cohort becomes worse by more than 2 absolute WER points without documented approval.
5. Keyword recall is not worse.
6. Keyword false-positive rate does not increase by more than 1 absolute percentage point.
7. Timestamp seeking remains within the accepted test tolerance.
8. Processing success rate is not more than 1 percentage point below legacy.
9. No operator voice samples or biometrics exist.
10. Cost and latency are reported and accepted explicitly.

## Activation

V2 is enabled only after real held-out labeled data exists and the acceptance gate passes. Set:

```env
TRANSCRIPTION_PIPELINE_DEFAULT=pipeline-v2
TRANSCRIPTION_PIPELINE_V2_ENABLED=true
```

Then record the activation in the deployment log and keep this document unchanged for the rollback path.

## Rollback

```env
TRANSCRIPTION_PIPELINE_DEFAULT=legacy-v1
TRANSCRIPTION_PIPELINE_V2_ENABLED=false
```

Rollback is a configuration change only; no database rollback is required. Existing V2 rows remain readable. Disabling V2 prevents new V2 work and rejects explicit V2 reprocess requests.

## Monitoring

- Watch processing success rate, latency per audio minute, and provider usage/cost.
- Watch keyword false-positive alerts and attribution review rates.
- Keep legacy-v1 available so rollback is always a configuration change.

## Recalibrating confidence thresholds

Review thresholds are derived from evaluation data. After a new labeled cohort is added:

1. Run the harness for legacy-v1 and pipeline-v2.
2. Plot review-flag rate against false positives and missed entities.
3. Adjust thresholds in the transcription confidence policy and document the version.
4. Re-run the acceptance gate before enabling or keeping V2.

## Voice biometrics prohibition

This system performs transcription and keyword matching. It does not enroll or verify speakers, create voice prints, or otherwise process biometrics.
