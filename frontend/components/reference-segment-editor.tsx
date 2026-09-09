"use client";

import { useState } from "react";
import { channelLabel, speakerOptions, SpeakerLabel } from "@/components/speaker-label";
import type { HumanReferenceSegment, ReferenceEntities } from "@/lib/types";

const entities: { key: keyof ReferenceEntities; label: string }[] = [
  { key: "names", label: "Names" }, { key: "telephone_numbers", label: "Telephone numbers" },
  { key: "licence_plates", label: "Licence plates" }, { key: "vehicle_models", label: "Vehicle models" },
];

function EntityInput({ label, values, onChange }: {
  label: string; values: string[]; onChange: (values: string[]) => void;
}) {
  const [input, setInput] = useState("");
  function add() {
    const value = input.trim();
    if (!value) return;
    onChange(Array.from(new Set([...values, value])));
    setInput("");
  }
  return <div className="entity-input">
    <label className="field">{label}<input value={input} maxLength={500}
      onChange={(event) => setInput(event.target.value)} onKeyDown={(event) => {
        if (event.key === "Enter") { event.preventDefault(); add(); }
      }} /></label>
    <button type="button" className="button secondary compact" onClick={add} disabled={!input.trim()}
      aria-label={`Add ${label.toLowerCase()}`}>Add</button>
    <ul className="entity-chips" aria-label={`Saved ${label.toLowerCase()}`}>
      {values.map((value) => <li key={value}>{value}<button type="button" className="text-button"
        onClick={() => onChange(values.filter((entry) => entry !== value))}
        aria-label={`Remove ${value}`}>×</button></li>)}
    </ul>
  </div>;
}

export function ReferenceSegmentEditor({ segment, index, stereo, duration, onChange, onDelete, onSeek, position }: {
  segment: HumanReferenceSegment; index: number; stereo: boolean; duration: number;
  onChange: (segment: HumanReferenceSegment) => void; onDelete: () => void;
  onSeek: (seconds: number) => void; position: () => number;
}) {
  function patch(value: Partial<HumanReferenceSegment>) { onChange({ ...segment, ...value }); }
  return <section className={`reference-segment${segment.exclude_from_wer ? " excluded" : ""}`} aria-label={`Reference segment ${index + 1}`}>
    <div className="section-header"><h3>Segment {index + 1}</h3>
      <button type="button" className="button secondary compact" onClick={onDelete}>Delete segment</button></div>
    <SpeakerLabel label={segment.speaker} source={stereo ? channelLabel(segment.channel) : "Mono"}
      timestamp={segment.start} onSeek={onSeek} />
    <div className="form-grid">
      <label className="field">Speaker<select aria-label="Speaker" value={segment.speaker}
        onChange={(event) => patch({ speaker: event.target.value as HumanReferenceSegment["speaker"] })}>
        {speakerOptions.map((value) => <option key={value}>{value}</option>)}
      </select></label>
      <label className="field">Channel<select aria-label="Channel" value={segment.channel === null ? "none" : segment.channel}
        onChange={(event) => patch({ channel: event.target.value === "none" ? null : Number(event.target.value) as 0 | 1 })}>
        <option value="none">None</option>
        {stereo ? <><option value="0">{channelLabel(0)}</option><option value="1">{channelLabel(1)}</option></> : null}
      </select></label>
      <div><label className="field">Start time (seconds)<input type="number" min="0" max={duration} step="0.001"
        value={Number.isFinite(segment.start) ? segment.start : ""} onChange={(event) => patch({ start: event.target.valueAsNumber })} /></label>
        <div className="page-actions">
          <button type="button" className="button secondary compact" onClick={() => patch({ start: position() })}>Set start to player position</button>
          <button type="button" className="text-button" onClick={() => onSeek(segment.start)}>Seek to start</button>
        </div></div>
      <div><label className="field">End time (seconds)<input type="number" min="0" max={duration} step="0.001"
        value={Number.isFinite(segment.end) ? segment.end : ""} onChange={(event) => patch({ end: event.target.valueAsNumber })} /></label>
        <div className="page-actions">
          <button type="button" className="button secondary compact" onClick={() => patch({ end: position() })}>Set end to player position</button>
          <button type="button" className="text-button" onClick={() => onSeek(segment.end)}>Seek to end</button>
        </div></div>
    </div>
    <label className="field">{segment.exclude_from_wer ? "Optional annotator note (excluded from WER)" : "Exact reference transcript"}
      <textarea rows={3} value={segment.text} maxLength={20000} onChange={(event) => patch({ text: event.target.value })} />
    </label>
    <label className="checkbox-row"><input type="checkbox" checked={segment.exclude_from_wer}
      onChange={(event) => patch({ exclude_from_wer: event.target.checked })} />Exclude unintelligible region from WER</label>
    {segment.exclude_from_wer ? <p className="notice">Excluded from text scoring. Leave the text empty or add an optional note; no special token is needed.</p> : null}
    <details><summary>Entity annotations</summary><p className="muted">Enter only fully audible values. Canonical values may differ from the spoken words. Never infer missing values.</p>
      <div className="form-grid">{entities.map(({ key, label }) =>
        <EntityInput key={key} label={label} values={segment.entities[key]}
          onChange={(values) => patch({ entities: { ...segment.entities, [key]: values } })} />)}
      </div>
    </details>
  </section>;
}
