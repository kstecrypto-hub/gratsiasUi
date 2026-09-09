"use client";

import Link from "next/link";
import { useParams } from "next/navigation";
import { useCallback, useEffect, useState } from "react";
import { CallMetadata } from "@/components/call-metadata";
import { ErrorState, LoadingState } from "@/components/page-state";
import { RecordingPlayer, useRecordingPlayer } from "@/components/recording-player";
import { ReferenceSegmentEditor } from "@/components/reference-segment-editor";
import { StatusLabel } from "@/components/status-label";
import { api, messageFromError } from "@/lib/api";
import { editableReference, qualityDefinitions, verificationErrors } from "@/lib/evaluation";
import { formatDateTime, formatDuration, titleCase } from "@/lib/format";
import type { EvaluationDetail, EvaluationKeyword, HumanReference, QualityLabel, ReferenceDraft } from "@/lib/types";

export default function EvaluationDetailPage() {
  const { id } = useParams<{ id: string }>();
  // A route change discards component state only after the navigation guard.
  return <ReferenceEditor key={id} id={id} />;
}

function ReferenceEditor({ id }: { id: string }) {
  const [call, setCall] = useState<EvaluationDetail>();
  const [saved, setSaved] = useState<HumanReference>();
  const [draft, setDraft] = useState<ReferenceDraft>();
  const [catalog, setCatalog] = useState<EvaluationKeyword[]>();
  const [catalogError, setCatalogError] = useState("");
  const [error, setError] = useState("");
  const [actionError, setActionError] = useState("");
  const [notice, setNotice] = useState("");
  const [busy, setBusy] = useState(false);
  const [reload, setReload] = useState(0);
  const [rowKeys, setRowKeys] = useState<string[]>([]);
  const { audioRef, seek } = useRecordingPlayer();
  const dirty = Boolean(draft && saved && JSON.stringify(draft) !== JSON.stringify(editableReference(saved)));

  const loadCatalog = useCallback(async () => {
    setCatalogError("");
    try { setCatalog(await api.evaluation.keywords()); }
    catch (caught) { setCatalogError(messageFromError(caught)); }
  }, []);

  useEffect(() => { void loadCatalog(); }, [loadCatalog]);
  useEffect(() => {
    let active = true;
    setError("");
    setBusy(true);
    api.evaluation.get(id).then((value) => {
      if (!active) return;
      setRowKeys(value.reference.segments.map(() => crypto.randomUUID()));
      setCall(value);
      setSaved(value.reference);
      setDraft(editableReference(value.reference));
    }).catch((caught) => { if (active) setError(messageFromError(caught)); })
      .finally(() => { if (active) setBusy(false); });
    return () => { active = false; };
  }, [id, reload]);

  useEffect(() => {
    if (!dirty && !busy) return;
    const beforeUnload = (event: BeforeUnloadEvent) => { event.preventDefault(); event.returnValue = ""; };
    const navigate = (event: MouseEvent) => {
      const anchor = (event.target as Element)?.closest?.("a");
      if (!anchor || anchor.target === "_blank" || anchor.href === window.location.href) return;
      if (!window.confirm(busy ? "A reference action is still running. Leave this page?" : "You have unsaved reference edits. Discard them and leave?")) {
        event.preventDefault();
        event.stopPropagation();
      }
    };
    window.addEventListener("beforeunload", beforeUnload);
    document.addEventListener("click", navigate, true);
    return () => {
      window.removeEventListener("beforeunload", beforeUnload);
      document.removeEventListener("click", navigate, true);
    };
  }, [dirty, busy]);

  if (error && !call) return <ErrorState message={error} onRetry={() => setReload((value) => value + 1)} />;
  if (!call || !saved || !draft) return <LoadingState label="Loading human reference" />;

  const currentCall = call;
  const currentDraft = draft;
  const currentSaved = saved;
  const validation = verificationErrors(draft, call);
  const finiteTimes = draft.segments.every((segment) => Number.isFinite(segment.start) && Number.isFinite(segment.end) && segment.start >= 0 && segment.end >= 0);
  const status = dirty ? "in_progress" : saved.verification_status;
  const sources = [
    { label: call.mode === "stereo" ? "Stereo" : "Mono", url: api.evaluation.audioUrl(id) },
    ...(call.mode === "stereo" ? [
      { label: "Channel A", url: api.evaluation.audioUrl(id, 0) },
      { label: "Channel B", url: api.evaluation.audioUrl(id, 1) },
    ] : []),
  ];

  function change(value: Partial<ReferenceDraft>) {
    setDraft((previous) => previous ? { ...previous, ...value } : previous);
    setNotice("");
    setActionError("");
  }

  function position() { return Math.round((audioRef.current?.currentTime || 0) * 1000) / 1000; }

  async function save() {
    setBusy(true); setActionError(""); setNotice("");
    try {
      const result = await api.evaluation.save(id, currentDraft, currentSaved.revision);
      setSaved(result);
      setDraft(editableReference(result));
      setNotice("Reference saved.");
    } catch (caught) { setActionError(messageFromError(caught)); }
    finally { setBusy(false); }
  }

  async function verify() {
    if (dirty || validation.length) return;
    setBusy(true); setActionError(""); setNotice("");
    try {
      const result = await api.evaluation.verify(id, currentSaved.revision);
      setSaved(result);
      setDraft(editableReference(result));
      setNotice("Reference marked fully reviewed.");
    } catch (caught) { setActionError(messageFromError(caught)); }
    finally { setBusy(false); }
  }

  const inactiveKeywords = draft.expected_keywords.filter((phrase) =>
    catalog && !catalog.some((keyword) => keyword.canonical_phrase === phrase));

  return <>
    <div className="page-intro"><div><h2>{call.evaluation_id}</h2><p>{call.split.toUpperCase()} · Human ground truth</p></div>
      <Link href="/evaluation" className="button secondary">Back to dataset</Link></div>
    <section className="section" aria-labelledby="evaluation-info-title">
      <div className="section-header"><h2 id="evaluation-info-title">Call information</h2><StatusLabel status={status} /></div>
      <CallMetadata items={[
        { label: "Evaluation ID", value: call.evaluation_id },
        { label: "Split", value: call.split.toUpperCase() },
        { label: "Duration", value: formatDuration(call.duration_seconds) },
        { label: "Date and time", value: formatDateTime(call.metadata.occurred_at) },
        { label: "Direction", value: titleCase(call.metadata.direction) },
        { label: "Queue", value: call.metadata.is_queue === true ? call.metadata.queue || "Queue" : call.metadata.is_queue === false ? "Non-queue" : call.metadata.queue || "Unknown" },
        { label: "Transfer", value: titleCase(call.metadata.transfer_state) },
        { label: "Audio topology", value: call.metadata.audio_topology || titleCase(call.mode) },
        { label: "PBX-known operators", value: call.metadata.operators.map((operator) => [operator.name, operator.extension].filter(Boolean).join(" · ")).join(", ") || "Unknown" },
        { label: "Caller", value: call.metadata.caller || "—" },
        { label: "Callee", value: call.metadata.callee || "—" },
        { label: "Verified at", value: !dirty ? formatDateTime(saved.verified_at) : "Review required" },
      ]} />
    </section>
    <section className="section" aria-labelledby="ground-truth-instructions">
      <div className="section-header"><h2 id="ground-truth-instructions">Write exactly what is audible</h2></div>
      <p>Do not rewrite grammar, summarize, improve the sentence, or infer missing words.</p>
      <p>Keep repetitions, false starts, spoken mistakes, and English brand/model terms as spoken.</p>
      <p className="muted">Work from the audio. Only human reference text is shown here.</p>
    </section>
    <section className="section" aria-labelledby="evaluation-recording-title">
      <div className="section-header"><div><h2 id="evaluation-recording-title">Recording</h2>
        <p>Click a segment timestamp to listen. For stereo audio, compare both channels to identify the participants.</p></div></div>
      <RecordingPlayer audioRef={audioRef} src={sources[0].url} sources={sources} />
    </section>
    <fieldset className="reference-fields" disabled={busy}>
      <legend className="sr-only">Human reference editor</legend>
      <section className="section">
        <div className="section-header"><h2>Quality and operator channel truth</h2></div>
        <div className="form-grid">
          <div><label className="field">Quality label<select aria-label="Quality label" value={draft.quality || ""}
            onChange={(event) => change({ quality: event.target.value as QualityLabel || null })}>
            <option value="">Choose quality</option>
            {Object.keys(qualityDefinitions).map((value) => <option key={value} value={value}>{titleCase(value)}</option>)}
          </select></label>
            <p className="field-help">{draft.quality ? qualityDefinitions[draft.quality] : "Choose the quality you hear in the recording."}</p>
          </div>
          {call.mode === "stereo" ? <label className="field">Human operator channel
            <select value={!draft.operator_channel_answered ? "" : draft.operator_channel === null ? "unknown" : draft.operator_channel}
              onChange={(event) => change({
                operator_channel_answered: event.target.value !== "",
                operator_channel: event.target.value === "" || event.target.value === "unknown" ? null : Number(event.target.value) as 0 | 1,
              })}>
              <option value="">Choose an answer</option><option value="0">Operator is Channel A / 0</option>
              <option value="1">Operator is Channel B / 1</option><option value="unknown">Cannot establish</option>
            </select></label> : null}
        </div>
        <details className="quality-definitions"><summary>Quality definitions</summary>
          <ul>{Object.entries(qualityDefinitions).map(([value, definition]) => <li key={value}><strong>{titleCase(value)}:</strong> {definition}</li>)}</ul>
        </details>
        <p className="muted">Judge quality from the audio alone. This channel answer is stored as human evaluation truth.</p>
      </section>
      <section className="section" aria-labelledby="human-segments-title">
        <div className="section-header"><div><h2 id="human-segments-title">Human reference segments</h2><p>Overlapping segments are allowed.</p></div>
          <button type="button" className="button" onClick={() => {
            const start = Math.min(position(), Math.max(0, currentCall.duration_seconds - 0.001));
            setRowKeys((keys) => [...keys, crypto.randomUUID()]);
            change({ segments: [...currentDraft.segments, {
              speaker: "Unknown", channel: null, start, end: Math.min(start + 2, currentCall.duration_seconds),
              text: "", exclude_from_wer: false,
              entities: { names: [], telephone_numbers: [], licence_plates: [], vehicle_models: [] },
            }] });
          }}>Add segment</button></div>
        {!draft.segments.length ? <p className="notice">No reference text yet. Listen and add your first segment.</p> : null}
        {draft.segments.map((segment, index) => <ReferenceSegmentEditor key={rowKeys[index]}
          segment={segment} index={index} stereo={call.mode === "stereo"} duration={call.duration_seconds}
          position={position} onSeek={seek}
          onChange={(next) => change({ segments: currentDraft.segments.map((row, rowIndex) => rowIndex === index ? next : row) })}
          onDelete={() => {
            setRowKeys((keys) => keys.filter((_, rowIndex) => rowIndex !== index));
            change({ segments: currentDraft.segments.filter((_, rowIndex) => rowIndex !== index) });
          }} />)}
      </section>
      <section className="section" aria-labelledby="keyword-truth-title">
        <div className="section-header"><h2 id="keyword-truth-title">Keywords genuinely spoken</h2></div>
        <p>Mark every catalog phrase you actually heard.</p>
        {catalogError ? <ErrorState message={catalogError} onRetry={() => void loadCatalog()} /> : !catalog ? <LoadingState label="Loading active keyword catalog" /> :
          catalog.length ? <div className="keyword-truth">{catalog.map((keyword) => <label key={keyword.id} className="checkbox-row">
            <input type="checkbox" checked={draft.expected_keywords.includes(keyword.canonical_phrase)}
              onChange={(event) => change({ expected_keywords: event.target.checked
                ? Array.from(new Set([...currentDraft.expected_keywords, keyword.canonical_phrase]))
                : currentDraft.expected_keywords.filter((phrase) => phrase !== keyword.canonical_phrase) })} />
            {keyword.canonical_phrase} <span className="muted">({keyword.category_name})</span>
          </label>)}</div> : <p className="notice">There are no active production keywords.</p>}
        {inactiveKeywords.length ? <div className="notice"><p>Previously selected phrases no longer in the active catalog:</p>
          {inactiveKeywords.map((phrase) => <label key={phrase} className="checkbox-row"><input type="checkbox" checked
            onChange={() => change({ expected_keywords: currentDraft.expected_keywords.filter((value) => value !== phrase) })} />{phrase}</label>)}
        </div> : null}
      </section>
    </fieldset>
    <section className="section" aria-labelledby="verification-title">
      <div className="section-header"><h2 id="verification-title">Reference verification</h2><StatusLabel status={status} /></div>
      {dirty ? <p className="notice" role="status">Unsaved edits. Save before marking fully reviewed.{saved.verification_status === "verified" ? " Editing requires a new explicit review." : ""}</p> : null}
      {validation.length ? <ul className="muted">{validation.map((message) => <li key={message}>{message}</li>)}</ul> : null}
      {!finiteTimes ? <p className="form-error">Enter nonnegative numeric start and end times before saving.</p> : null}
      <div className="page-actions">
        <button type="button" className="button" disabled={busy || !dirty || !finiteTimes} onClick={() => void save()}>{busy ? "Working…" : "Save reference"}</button>
        <button type="button" className="button secondary" disabled={busy || dirty || validation.length > 0 || !catalog || status === "verified"}
          onClick={() => void verify()}>Mark fully reviewed</button>
        <button type="button" className="button secondary" disabled={busy} onClick={() => {
          if (dirty && !window.confirm("Discard unsaved reference edits and reload the saved version?")) return;
          setActionError(""); setNotice(""); setReload((value) => value + 1);
        }}>Reload saved reference</button>
      </div>
      {notice ? <p className="notice" role="status">{notice}</p> : null}
      {error || actionError ? <div className="form-error" role="alert">{actionError || error}</div> : null}
    </section>
  </>;
}
