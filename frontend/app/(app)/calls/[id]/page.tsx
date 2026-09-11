"use client";

import Link from "next/link";
import { useParams, useRouter } from "next/navigation";
import { useCallback, useEffect, useMemo, useState } from "react";
import { CallMetadata } from "@/components/call-metadata";
import { RecordingPlayer, useRecordingPlayer } from "@/components/recording-player";
import { CallTranscript, segmentNeedsReview } from "@/components/call-transcript";
import { ErrorState, LoadingState, TableEmpty } from "@/components/page-state";
import { StatusLabel } from "@/components/status-label";
import { api, messageFromError } from "@/lib/api";
import { formatDateTime, formatDuration, formatTimestamp, titleCase } from "@/lib/format";
import type { CallDetail } from "@/lib/types";

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
  const [position, setPosition] = useState(0);
  const [audioPlayable, setAudioPlayable] = useState(true);

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
  useEffect(() => { setAudioPlayable(true); setPosition(0); }, [id]);

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
  const reviewCount = segments.filter(segmentNeedsReview).length;
  const wordingReview = call?.transcript_quality_summaries?.length === 1
    ? call.transcript_quality_summaries[0].quality_summary?.wording_review : undefined;

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
  const canSeek = hasAudio && audioPlayable;
  const processingStatus = call.processing_status || call.status;
  const canRetry = ["failed", "completed_with_errors"].includes(call.processing_status || call.status || "");
  const operatorName = call.operator?.display_name || call.operator_name || "—";

  return (
    <div className="call-workspace">
      <Link className="call-back" href={resultsHref}>← Back to results</Link>
      <div className="call-intro">
        <div><span className="eyebrow">{titleCase(call.direction)} CALL · {formatDuration(call.duration_seconds)}</span>
          <h2>{operatorName === "—" ? "Call conversation" : operatorName}</h2><p>{formatDateTime(call.occurred_at || call.started_at)}</p></div>
        <StatusLabel status={processingStatus === "completed_speaker_attribution_unknown" ? "completed" : processingStatus} />
      </div>
      <div className="call-parties"><span><small>FROM</small> {call.caller || "Unknown caller"}</span><span aria-hidden="true">→</span><span><small>TO</small> {call.callee || "Unknown callee"}</span>
        {call.queue ? <span className="call-queue">{call.queue}</span> : null}</div>
      {actionError ? <div className="form-error" role="alert">{actionError}</div> : null}

      <section className="call-recording" aria-labelledby="recording-title">
        <div className="recording-heading"><h2 id="recording-title">Recording</h2><span>{hasAudio ? "Original audio · Listen and review" : "Audio unavailable"}</span></div>
        {hasAudio ? <RecordingPlayer key={id} audioRef={audioRef} src={api.calls.audioUrl(id)} onPositionChange={setPosition} onAvailabilityChange={setAudioPlayable}
          errorMessage="The recording could not be played. Try refreshing; it may have been removed from the phone system." /> : <p className="muted">No recording is available for this call.</p>}
      </section>

      <div className="call-review-summary">
        <div><strong>{reviewCount ? "A closer listen is needed" : "Ready to read"}</strong>
          <p>{reviewCount ? "Check highlighted wording and unclear speaker changes below." : "Read the conversation or select a timestamp to listen."}</p></div>
        <div className="page-actions">
          {canRetry ? <button className="button secondary compact" type="button" disabled={retrying || reprocessing} onClick={() => void retry()}>{retrying ? "Retrying…" : "Retry processing"}</button> : null}
          {transcriptionV2Enabled && call.transcript_id ? <button className="button secondary compact" type="button" disabled={reprocessing || retrying} onClick={() => void reprocess()}>{reprocessing ? "Queuing…" : "Retranscribe audio"}</button> : null}
        </div>
        {transcriptionV2Enabled && call.transcript_id ? <small>A new transcription checks the wording again. Transcription charges apply; previous versions are kept.</small> : null}
      </div>

      <CallTranscript key={String(call.transcript_id || id)} segments={segments} matches={matches} reviews={wordingReview?.items || []}
        verificationStatus={wordingReview?.status} truncated={wordingReview?.truncated} position={position} canSeek={canSeek} onSeek={seek} />

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

      <details className="call-disclosure" open={matches.length > 0}>
        <summary>Detected phrases <span>{matches.length}</span></summary>
        <div className="table-wrap">
          <table>
            <thead><tr><th scope="col">Time</th><th scope="col">Category</th><th scope="col">Phrase</th><th scope="col">Transcript context</th><th scope="col">Method</th></tr></thead>
            <tbody>{matches.length ? matches.map((match) => <tr key={String(match.id)}><td><button type="button" className="button secondary compact match-time" onClick={() => seek(match.start_timestamp)} disabled={!canSeek} aria-label={`Play recording from ${formatTimestamp(match.start_timestamp)}`}>{formatTimestamp(match.start_timestamp)}</button></td><td>{match.category_name || match.category || "—"}</td><td>{match.keyword_phrase || match.keyword || match.original_matched_text || "—"}</td><td>{[match.context_before, match.original_matched_text, match.context_after].filter(Boolean).join(" ") || "—"}</td><td>{titleCase(match.match_method)}</td></tr>) : <TableEmpty colSpan={5}>No phrases were detected for this call.</TableEmpty>}</tbody>
          </table>
        </div>
      </details>

      <details className="call-disclosure">
        <summary>Call details</summary>
        <CallMetadata items={[
          { label: "Date and time", value: formatDateTime(call.occurred_at || call.started_at) },
          { label: "Operator", value: operatorName },
          { label: "Duration", value: formatDuration(call.duration_seconds) },
          { label: "Direction", value: titleCase(call.direction) },
          { label: "Queue", value: call.queue || "—" },
          { label: "Detected phrases", value: matches.length },
        ]} />
      </details>
      <details className="call-disclosure">
        <summary>Processing history <span>{call.processing_history?.length || 0}</span></summary>
        <div className="table-wrap">
          <table><thead><tr><th scope="col">Time</th><th scope="col">Status</th><th scope="col">Details</th></tr></thead><tbody>{call.processing_history?.length ? call.processing_history.map((entry, index) => <tr key={String(entry.id ?? index)}><td>{formatDateTime(entry.occurred_at || entry.created_at)}</td><td><StatusLabel status={entry.status} /></td><td>{entry.message || "—"}</td></tr>) : <TableEmpty colSpan={3}>No processing history is available.</TableEmpty>}</tbody></table>
        </div>
      </details>
    </div>
  );
}
