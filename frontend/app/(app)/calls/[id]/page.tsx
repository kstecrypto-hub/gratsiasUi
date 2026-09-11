"use client";

import Link from "next/link";
import { useParams, useRouter } from "next/navigation";
import { Fragment, useCallback, useEffect, useMemo, useState } from "react";
import { CallMetadata } from "@/components/call-metadata";
import { RecordingPlayer, useRecordingPlayer } from "@/components/recording-player";
import { SpeakerLabel } from "@/components/speaker-label";
import { ErrorState, LoadingState, TableEmpty } from "@/components/page-state";
import { StatusLabel } from "@/components/status-label";
import { api, messageFromError } from "@/lib/api";
import { formatDateTime, formatDuration, formatTimestamp, titleCase } from "@/lib/format";
import type { CallDetail, KeywordMatch, TranscriptSegment } from "@/lib/types";

export default function CallDetailPage() {
  const params = useParams<{ id: string }>();
  const router = useRouter();
  const id = params.id;
  const { audioRef, seek } = useRecordingPlayer();
  const [call, setCall] = useState<CallDetail>();
  const [error, setError] = useState("");
  const [actionError, setActionError] = useState("");
  const [retrying, setRetrying] = useState(false);
  const [reprocessing, setReprocessing] = useState(false);
  const [transcriptionV2Enabled, setTranscriptionV2Enabled] = useState(false);
  const [assigning, setAssigning] = useState(false);
  const [assignmentError, setAssignmentError] = useState("");
  const [assignmentOperatorId, setAssignmentOperatorId] = useState("");
  const [assignmentDismissed, setAssignmentDismissed] = useState(false);
  const [resultsHref, setResultsHref] = useState("/results");

  const load = useCallback(async () => {
    setError("");
    try {
      const jobId = new URLSearchParams(window.location.search).get("job_id") || undefined;
      setCall(await api.calls.get(id, jobId));
    } catch (caught) {
      setError(messageFromError(caught));
    }
  }, [id]);

  useEffect(() => { void load(); }, [load]);
  useEffect(() => {
    let mounted = true;
    void api.features().then((features) => {
      if (mounted) setTranscriptionV2Enabled(features.transcription_v2_enabled === true);
    }).catch(() => {});
    return () => { mounted = false; };
  }, []);
  useEffect(() => { setResultsHref(`/results${window.location.search}`); }, []);

  const matches = useMemo(() => call?.matches || [], [call]);
  const segments = useMemo(() => call?.transcript_segments || [], [call]);
  const assignmentOperators = useMemo(() => {
    const rows = (call?.participants || [])
      .filter((participant) => participant.operator_id)
      .map((participant) => ({
        id: String(participant.operator_id),
        name: participant.operator_name || `Extension ${participant.extension || ""}`.trim(),
      }));
    return Array.from(new Map(rows.map((row) => [row.id, row])).values());
  }, [call]);
  useEffect(() => {
    if (assignmentOperators.length === 1 && assignmentOperatorId !== assignmentOperators[0].id) {
      setAssignmentOperatorId(assignmentOperators[0].id);
    }
    if (assignmentOperators.length === 0) setAssignmentOperatorId("");
  }, [assignmentOperators, assignmentOperatorId]);
  useEffect(() => { setAssignmentDismissed(false); }, [call?.id]);

  async function retry() {
    setRetrying(true);
    setActionError("");
    try {
      await api.calls.retry(id);
      await load();
    } catch (caught) {
      setActionError(messageFromError(caught, "This call could not be retried."));
    } finally {
      setRetrying(false);
    }
  }

  async function reprocess() {
    if (!call?.transcript_id) return;
    setReprocessing(true);
    setActionError("");
    try {
      const job = await api.calls.reprocess(id, call.transcript_id);
      router.push(`/processing/${encodeURIComponent(job.id)}`);
    } catch (caught) {
      setActionError(messageFromError(caught, "This call could not be retranscribed."));
    } finally {
      setReprocessing(false);
    }
  }

  const assignmentAvailable = call?.transcription_mode === "dual_channel" && (
    call?.speaker_assignment_required === true || call?.speaker_attribution_status === "manually_assigned"
  );
  const needsReview = (call?.confidence_status && call.confidence_status !== "unavailable") ||
    Boolean(call?.transcript_segments?.some((segment) => segment.quality_flags?.length));
  const approximateTimes = call?.transcript_segments?.some((segment) =>
    segment.quality_flags?.includes("approximate_timestamps"));

  async function assignChannel(channel: 0 | 1) {
    if (!call?.transcript_id || !assignmentOperatorId) {
      setAssignmentError("Choose the operator whose speech should be assigned.");
      return;
    }
    if (call.speaker_attribution_status === "manually_assigned") {
      const confirmed = window.confirm(
        "A manual operator-channel assignment already exists. Replace it with the channel you selected?",
      );
      if (!confirmed) return;
    }
    setAssigning(true);
    setAssignmentError("");
    try {
      await api.calls.assignSpeaker(id, {
        transcript_id: call.transcript_id,
        operator_id: assignmentOperatorId,
        operator_channel_index: channel,
      });
      await load();
    } catch (caught) {
      setAssignmentError(messageFromError(caught, "The operator channel could not be assigned."));
    } finally {
      setAssigning(false);
    }
  }

  if (!call) {
    if (error) return <ErrorState message={error} onRetry={() => void load()} />;
    return <LoadingState label="Loading call" />;
  }

  const hasAudio = call.audio_available === true || call.recording_available === true;
  const canRetry = ["failed", "completed_with_errors"].includes(call.processing_status || call.status || "");
  const operatorName = call.operator?.display_name || call.operator_name || "—";

  return (
    <>
      <div className="page-intro">
        <div><h2>{formatDateTime(call.occurred_at || call.started_at)}</h2><p>{operatorName}</p></div>
        <div className="page-actions">{canRetry ? <button className="button secondary" type="button" disabled={retrying || reprocessing} onClick={() => void retry()}>{retrying ? "Retrying…" : "Retry processing"}</button> : null}<Link className="button secondary" href={resultsHref}>Back to results</Link></div>
      </div>
      {actionError ? <div className="form-error" role="alert">{actionError}</div> : null}

      <section className="section" aria-labelledby="call-info-title">
        <div className="section-header"><div><h2 id="call-info-title">Call information</h2></div><StatusLabel status={call.processing_status || call.status} /></div>
        <CallMetadata items={[
          { label: "Date and time", value: formatDateTime(call.occurred_at || call.started_at) },
          { label: "Operator", value: operatorName },
          { label: "Caller", value: call.caller || "—" },
          { label: "Callee", value: call.callee || "—" },
          { label: "Duration", value: formatDuration(call.duration_seconds) },
          { label: "Direction", value: titleCase(call.direction) },
          { label: "Queue", value: call.queue || "—" },
          { label: "Detected phrases", value: matches.length },
        ]} />
      </section>

      {assignmentAvailable && !assignmentDismissed ? (
        <section className="section" aria-labelledby="speaker-assignment-title">
          <div className="section-header">
            <div>
              <h2 id="speaker-assignment-title">Operator channel</h2>
              {call.speaker_assignment_required ? (
                <p className="notice" role="status">
                  The operator channel could not be identified automatically.
                </p>
              ) : (
                <p>A manual operator-channel assignment is already in place.</p>
              )}
            </div>
          </div>
          {assignmentOperators.length > 1 ? (
            <label className="field">
              Operator
              <select
                value={assignmentOperatorId}
                onChange={(event) => setAssignmentOperatorId(event.target.value)}
              >
                <option value="">Choose operator</option>
                {assignmentOperators.map((operator) => (
                  <option key={operator.id} value={operator.id}>{operator.name}</option>
                ))}
              </select>
            </label>
          ) : null}
          <div className="page-actions">
            {call.available_channels?.includes(0) ? (
              <button
                type="button"
                className="button"
                disabled={assigning || !assignmentOperatorId}
                onClick={() => void assignChannel(0)}
              >
                {assigning ? "Assigning…" : "Operator is Channel A"}
              </button>
            ) : null}
            {call.available_channels?.includes(1) ? (
              <button
                type="button"
                className="button"
                disabled={assigning || !assignmentOperatorId}
                onClick={() => void assignChannel(1)}
              >
                {assigning ? "Assigning…" : "Operator is Channel B"}
              </button>
            ) : null}
            {call.speaker_assignment_required ? (
              <button
                type="button"
                className="button secondary"
                disabled={assigning}
                onClick={() => setAssignmentDismissed(true)}
              >
                Leave unassigned
              </button>
            ) : null}
          </div>
          {assignmentError ? <div className="form-error" role="alert">{assignmentError}</div> : null}
        </section>
      ) : null}

      <section className="section" aria-labelledby="recording-title">
        <div className="section-header"><div><h2 id="recording-title">Recording</h2><p>Use a detected phrase timestamp to jump to that moment.</p></div></div>
        {hasAudio ? <RecordingPlayer key={id} audioRef={audioRef} src={api.calls.audioUrl(id)} errorMessage="The recording could not be played. Your session may have expired or the audio may have been removed according to the retention settings." /> : <div className="notice" role="status">No recording is available for this call.</div>}
      </section>

      <section className="section" aria-labelledby="matches-title">
        <div className="section-header"><div><h2 id="matches-title">Detected phrases</h2><p>Phrase searches normally use attributed operator speech. If all speakers were explicitly included, unattributed speech remains marked as unknown below.</p></div></div>
        <div className="table-wrap">
          <table>
            <thead><tr><th scope="col">Time</th><th scope="col">Category</th><th scope="col">Phrase</th><th scope="col">Transcript context</th><th scope="col">Method</th></tr></thead>
            <tbody>{matches.length ? matches.map((match) => <tr key={String(match.id)}><td><button type="button" className="button secondary compact match-time" onClick={() => seek(match.start_timestamp)} disabled={!hasAudio} aria-label={`Play recording from ${formatTimestamp(match.start_timestamp)}`}>{formatTimestamp(match.start_timestamp)}</button></td><td>{match.category_name || match.category || "—"}</td><td>{match.keyword_phrase || match.keyword || match.original_matched_text || "—"}</td><td>{[match.context_before, match.original_matched_text, match.context_after].filter(Boolean).join(" ") || "—"}</td><td>{titleCase(match.match_method)}</td></tr>) : <TableEmpty colSpan={5}>No phrases were detected for this call.</TableEmpty>}</tbody>
          </table>
        </div>
      </section>

      <section className="section" aria-labelledby="transcript-title">
        {transcriptionV2Enabled && call.transcript_id ? (
          <div className="section-header">
            <div><p>Transcribe the audio again with Greek vocabulary and a second pass for mixed-speaker recordings. Transcription charges apply, and the previous version stays in history.</p></div>
            <button className="button secondary" type="button" disabled={reprocessing || retrying} onClick={() => void reprocess()}>{reprocessing ? "Queuing…" : "Retranscribe audio"}</button>
          </div>
        ) : null}
        <div className="section-header"><div><h2 id="transcript-title">Transcript</h2><p>Speaker labels reflect the available call and audio evidence. Unknown speakers remain marked as unknown.</p>{needsReview ? <span className="notice" role="status">Needs review</span> : null}</div></div>
        {approximateTimes ? <p className="muted">This transcript uses the complete conversation. Speaker timestamps are approximate; words with uncertain speaker attribution are marked Unknown.</p> : null}
        {segments.length ? <div className="transcript">{segments.map((segment) => <TranscriptRow key={String(segment.id)} segment={segment} matches={matches} onSeek={seek} />)}</div> : <div className="empty-state"><h2>No transcript is available</h2><p className="muted">The recording may not have been transcribed, or processing may still be underway.</p></div>}
      </section>

      <section className="section" aria-labelledby="history-title">
        <div className="section-header"><div><h2 id="history-title">Processing history</h2><p>A plain-language record of processing steps for this call.</p></div></div>
        <div className="table-wrap">
          <table><thead><tr><th scope="col">Time</th><th scope="col">Status</th><th scope="col">Details</th></tr></thead><tbody>{call.processing_history?.length ? call.processing_history.map((entry, index) => <tr key={String(entry.id ?? index)}><td>{formatDateTime(entry.occurred_at || entry.created_at)}</td><td><StatusLabel status={entry.status} /></td><td>{entry.message || "—"}</td></tr>) : <TableEmpty colSpan={3}>No processing history is available.</TableEmpty>}</tbody></table>
        </div>
      </section>
    </>
  );
}

function TranscriptRow({ segment, matches, onSeek }: { segment: TranscriptSegment; matches: KeywordMatch[]; onSeek: (seconds: number) => void }) {
  const segmentMatches = matches.filter((match) => String(match.transcript_segment_id) === String(segment.id));
  const terms = segmentMatches.map((match) => match.original_matched_text || match.keyword_phrase || match.keyword || "").filter(Boolean);
  return <div className="transcript-segment"><SpeakerLabel label={segment.speaker_label} source={segment.speaker_source} timestamp={segment.start_timestamp} onSeek={onSeek} /><p className="transcript-text">{highlight(segment.original_text, terms)}</p></div>;
}

function highlight(text: string, terms: string[]) {
  const unique = Array.from(new Set(terms.map((term) => term.trim()).filter(Boolean))).sort((a, b) => b.length - a.length);
  if (!unique.length) return text;
  const escaped = unique.map((term) => term.replace(/[.*+?^${}()|[\]\\]/g, "\\$&"));
  const expression = new RegExp(`(${escaped.join("|")})`, "giu");
  return text.split(expression).map((part, index) => unique.some((term) => term.toLocaleLowerCase("el-GR") === part.toLocaleLowerCase("el-GR")) ? <mark key={`${part}-${index}`}>{part}</mark> : <Fragment key={`${part}-${index}`}>{part}</Fragment>);
}
