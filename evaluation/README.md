# Transcription evaluation and rollout

This directory is for **local, held-out evaluation only**. It never grades with
an LLM and never invents production metrics.

## Layout

- `evaluate.py` — deterministic metric and report runner.
- `manifest.example.jsonl` — example manifest; copy to a local, git-ignored path before adding real data.
- `reference.example.json` — example reference shape.
- `criteria.json` — versioned minimum acceptance thresholds.
- `reports/` — generated reports; detailed reports are git-ignored.

## Privacy

Real `audio/`, `references/`, and manifests containing personal data must stay
local and are ignored by Git. Generated detailed reports are also ignored.
Never commit customer audio, transcripts, names, telephone numbers, licence
plates, or vehicle models.

## Reference shape

Each reference file contains:

```json
{
  "id": "call-001",
  "quality": "noisy_mobile",
  "mode": "stereo",
  "split": "test",
  "expected_keywords": ["προσφορά"],
  "segments": [
    {
      "speaker": "Operator",
      "channel": 0,
      "start": 0.0,
      "end": 3.2,
      "text": "Καλημέρα",
      "entities": {
        "names": [],
        "telephone_numbers": [],
        "licence_plates": [],
        "vehicle_models": []
      }
    }
  ]
}
```

## Manifest shape

Each line is one JSON object:

```json
{"id":"call-001","audio":"audio/call-001.wav","reference":"references/call-001.json","split":"test","quality":"noisy_mobile","mode":"stereo"}
```

## Commands

Validate a manifest:

```bash
python evaluation/evaluate.py validate evaluation/manifest.local.jsonl
```

Run the same labeled set through both pipeline versions. The runner invokes
`EVALUATION_TRANSCRIBE_COMMAND` for each audio/version and reads JSON from
stdout. It does nothing useful without real local data and that command:

```bash
EVALUATION_TRANSCRIBE_COMMAND=/path/to/local-transcriber \
  python evaluation/evaluate.py run evaluation/manifest.local.jsonl --output evaluation/reports
```

Aggregate existing per-call JSON results without re-running audio:

```bash
python evaluation/evaluate.py report evaluation/reports/per-call --output evaluation/reports
```

## Metrics

Metrics are computed deterministically from reference/hypothesis text:

- WER, CER, raw-text WER
- operator-only WER and customer-only WER
- keyword recall, precision, false-positive rate
- name, telephone-number, licence-plate, and vehicle-model accuracy
- speaker/channel attribution accuracy
- review-flagged chunk percentage
- processing success rate
- latency per audio minute and API usage/cost when supplied

Greek normalization is stricter than production normalization and is
implemented in `normalize_greek`. Raw-text metrics remain available for
inspection.
