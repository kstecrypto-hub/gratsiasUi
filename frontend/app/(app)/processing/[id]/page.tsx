"use client";

import Link from "next/link";
import { useParams } from "next/navigation";
import { useCallback, useEffect, useMemo, useState } from "react";
import { ErrorState, LoadingState } from "@/components/page-state";
import { StatusLabel } from "@/components/status-label";
import { api, messageFromError } from "@/lib/api";
import { formatDateTime } from "@/lib/format";
import type { ProcessingJob } from "@/lib/types";

const activeStates = new Set(["queued", "waiting_for_connection", "connecting", "fetching_calls", "fetching_call_details", "finding_recordings", "downloading_recordings", "inspecting_audio", "extracting_operator_audio", "transcribing", "searching_keywords"]);

export default function ProcessingPage() {
  const params = useParams<{ id: string }>();
  const id = params.id;
  const [job, setJob] = useState<ProcessingJob>();
  const [error, setError] = useState("");
  const [actionError, setActionError] = useState("");
  const [action, setAction] = useState<"retry" | "cancel" | "">("");

  const load = useCallback(async (quiet = false) => {
    if (!quiet) setError("");
    try {
      setJob(await api.jobs.get(id));
    } catch (caught) {
      if (!quiet) setError(messageFromError(caught));
    }
  }, [id]);

  useEffect(() => { void load(); }, [load]);
  useEffect(() => {
    if (!job || !activeStates.has(job.status)) return;
    const timer = window.setInterval(() => void load(true), 3000);
    return () => window.clearInterval(timer);
  }, [job, load]);
  useEffect(() => {
    if (!job) return;
    window.dispatchEvent(new CustomEvent("processing-status", { detail: humanStage(job.status, job.current_stage) }));
    return () => { window.dispatchEvent(new CustomEvent("processing-status", { detail: "" })); };
  }, [job]);

  const stages = useMemo(() => job ? stageRows(job) : [], [job]);

  async function runAction(nextAction: "retry" | "cancel") {
    if (nextAction === "cancel" && !window.confirm("Cancel this analysis? Completed calls will be kept.")) return;
    setAction(nextAction);
    setActionError("");
    try {
      const updated = nextAction === "retry" ? await api.jobs.retry(id) : await api.jobs.cancel(id);
      setJob(updated);
    } catch (caught) {
      setActionError(messageFromError(caught, `The analysis could not be ${nextAction === "retry" ? "retried" : "cancelled"}.`));
    } finally {
      setAction("");
    }
  }

  if (!job) {
    if (error) return <ErrorState message={error} onRetry={() => void load()} />;
    return <LoadingState label="Loading analysis progress" />;
  }

  const isActive = activeStates.has(job.status);
  const canRetry = (job.calls_failed || 0) > 0 || job.status === "failed" || job.status === "completed_with_errors";
  const progress = progressPercent(job);
  const waitingForRecordingAssignment = job.current_stage === "Waiting for recording assignment";
  const operatorText = job.operators?.length
    ? job.operators.map((operator) => operator.display_name).join(", ")
    : job.operator_ids?.length
      ? `${job.operator_ids.length} selected operator${job.operator_ids.length === 1 ? "" : "s"}`
      : "—";

  return (
    <>
      <div className="page-intro">
        <div>
          <h2>Analyzing calls from {formatDateTime(job.date_from)} to {formatDateTime(job.date_to)}</h2>
          <p>{operatorText}</p>
        </div>
        <StatusLabel status={job.status} />
      </div>
      {actionError ? <div className="form-error" role="alert">{actionError}</div> : null}
      {job.error_message ? <div className="form-error" role="alert">{job.error_message}</div> : null}

      <section className="flat-panel" aria-labelledby="progress-title">
        <div className="section-header"><div><h2 id="progress-title">Overall progress</h2><p>You can leave this page. The analysis continues in the background.</p></div></div>
        <div className="progress-block">
          <div className="progress-header"><span>{humanStage(job.status, job.current_stage)}</span><strong>{progress}%</strong></div>
          <progress max={100} value={progress}>{progress}%</progress>
        </div>
        {waitingForRecordingAssignment ? <div className="notice" role="status">We found more than one recording for a call. We are checking again automatically so the correct conversation is used.</div> : null}
        <dl className="job-summary">
          <div><dt>Calls found</dt><dd>{job.calls_found ?? 0}</dd></div>
          <div><dt>Recordings found</dt><dd>{job.recordings_found ?? 0}</dd></div>
          <div><dt>Calls completed</dt><dd>{job.calls_completed ?? 0}</dd></div>
          <div><dt>Calls failed</dt><dd>{job.calls_failed ?? 0}</dd></div>
        </dl>
        <div className="stage-list" aria-label="Analysis stages">
          {stages.map((stage) => <div className="stage-row" key={stage.label}><span>{stage.label}</span><span className="stage-value">{stage.value}</span></div>)}
        </div>
      </section>

      <div className="form-actions" style={{ marginTop: 20 }}>
        {canRetry ? <button className="button primary" type="button" disabled={Boolean(action)} onClick={() => void runAction("retry")}>{action === "retry" ? "Retrying…" : "Retry failed calls"}</button> : null}
        {isActive ? <button className="button danger" type="button" disabled={Boolean(action)} onClick={() => void runAction("cancel")}>{action === "cancel" ? "Cancelling…" : "Cancel analysis"}</button> : null}
        {job.status === "completed" || job.status === "completed_with_errors" ? <Link className="button secondary" href={`/results?job_id=${encodeURIComponent(String(job.id))}`}>View results</Link> : null}
        <Link className="button secondary" href="/dashboard">Return to dashboard</Link>
      </div>
    </>
  );
}

function progressPercent(job: ProcessingJob): number {
  if (typeof job.progress_percent === "number") return Math.min(100, Math.max(0, Math.round(job.progress_percent)));
  if (job.status === "completed" || job.status === "completed_with_errors") return 100;
  if (job.total_items && typeof job.calls_completed === "number") return Math.min(99, Math.round((job.calls_completed / job.total_items) * 100));
  const fallback: Record<string, number> = { queued: 2, connecting: 5, fetching_calls: 12, fetching_call_details: 22, finding_recordings: 32, downloading_recordings: 42, inspecting_audio: 50, extracting_operator_audio: 58, transcribing: 68, searching_keywords: 88, failed: 100, cancelled: 100 };
  return fallback[job.status] ?? 0;
}

function humanStage(status: string, currentStage?: string): string {
  if (currentStage === "Waiting for recording assignment") return "Waiting for recordings to be ready";
  const labels: Record<string, string> = {
    queued: "Waiting to start",
    waiting_for_connection: "Waiting for the phone-system connection",
    connecting: "Connecting to the phone system",
    fetching_calls: "Finding calls",
    fetching_call_details: "Reviewing call details",
    finding_recordings: "Finding recordings",
    downloading_recordings: "Preparing recordings",
    inspecting_audio: "Preparing recordings",
    extracting_operator_audio: "Preparing operator speech",
    transcribing: "Transcribing conversations",
    searching_keywords: "Preparing transcript search",
    completed: "Preparing results complete",
    completed_with_errors: "Preparing results complete with some errors",
    failed: "Analysis failed",
    cancelled: "Analysis cancelled",
  };
  return labels[status] || "Processing calls";
}

function stageRows(job: ProcessingJob): Array<{ label: string; value: string }> {
  const sequence = ["finding_calls", "finding_recordings", "transcribing", "searching_keywords", "preparing_results"];
  const currentMap: Record<string, number> = {
    queued: -1,
    waiting_for_connection: -1,
    connecting: -1,
    fetching_calls: 0,
    fetching_call_details: 0,
    finding_recordings: 1,
    downloading_recordings: 1,
    inspecting_audio: 1,
    extracting_operator_audio: 1,
    transcribing: 2,
    searching_keywords: 3,
    completed: 5,
    completed_with_errors: 5,
    failed: -2,
    cancelled: -2,
  };
  const current = currentMap[job.status] ?? -1;
  const stopped = job.status === "failed" || job.status === "cancelled";
  return sequence.map((stage, index) => {
    let value = index < current ? "Complete" : index === current ? "In progress" : stopped ? "Not completed" : "Waiting";
    if (index === current && stage === "transcribing" && job.recordings_found) value = `${job.transcribed_count ?? job.calls_completed ?? 0} of ${job.recordings_found}`;
    if (index === current && stage === "searching_keywords" && job.total_items) value = `${job.searched_count ?? job.calls_completed ?? 0} of ${job.total_items}`;
    return { label: ({ finding_calls: "Finding calls", finding_recordings: "Finding recordings", transcribing: "Transcribing conversations", searching_keywords: "Preparing transcript search", preparing_results: "Preparing results" } as Record<string, string>)[stage], value };
  });
}
