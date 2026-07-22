"use client";

import Link from "next/link";
import { FormEvent, useCallback, useEffect, useMemo, useState } from "react";
import { useRouter } from "next/navigation";
import { ConfigurationBanner } from "@/components/configuration-banner";
import { ErrorState, LoadingState } from "@/components/page-state";
import { StatusLabel } from "@/components/status-label";
import { api, asList, messageFromError } from "@/lib/api";
import { configurationReady, toIsoDateTime } from "@/lib/format";
import type { Configuration, Identifier, Operator, ProcessingJob } from "@/lib/types";

const terminalJobStates = new Set(["completed", "completed_with_errors", "failed", "cancelled"]);

export default function AnalyzePage() {
  const router = useRouter();
  const [configuration, setConfiguration] = useState<Configuration>();
  const [operators, setOperators] = useState<Operator[]>([]);
  const [activeJob, setActiveJob] = useState<ProcessingJob>();
  const [analysisDay, setAnalysisDay] = useState("");
  const [operatorIds, setOperatorIds] = useState<Identifier[]>([]);
  const [operatorSearch, setOperatorSearch] = useState("");
  const [loading, setLoading] = useState(true);
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState("");
  const [formError, setFormError] = useState("");

  const load = useCallback(async () => {
    setLoading(true);
    setError("");
    try {
      const [config, operatorPayload, runningJob] = await Promise.all([api.configuration(), api.operators.list(), api.jobs.active()]);
      setConfiguration(config);
      setOperators(asList(operatorPayload));
      setActiveJob(runningJob && !terminalJobStates.has(runningJob.status) ? runningJob : undefined);
    } catch (caught) {
      setError(messageFromError(caught));
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => { void load(); }, [load]);

  const enabledOperators = useMemo(
    () => operators.filter((operator) => operator.enabled),
    [operators],
  );
  const visibleOperators = useMemo(() => {
    const query = operatorSearch.trim().toLocaleLowerCase();
    if (!query) return enabledOperators;

    return enabledOperators.filter((operator) =>
      [operator.display_name, operator.extension_number, operator.email]
        .some((value) => value?.toLocaleLowerCase().includes(query)),
    );
  }, [enabledOperators, operatorSearch]);

  function toggle(values: Identifier[], id: Identifier, setValues: (next: Identifier[]) => void) {
    setValues(values.some((value) => String(value) === String(id)) ? values.filter((value) => String(value) !== String(id)) : [...values, id]);
  }

  async function submit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    setFormError("");
    if (!configuration || !configurationReady(configuration.yeastar.status) || !configurationReady(configuration.openai.status)) {
      setFormError("Configure the phone system and transcription service before starting an analysis.");
      return;
    }
    if (!analysisDay) {
      setFormError("Choose the day to analyze.");
      return;
    }
    if (operatorIds.length === 0) {
      setFormError("Select at least one enabled operator.");
      return;
    }
    try {
      const { from, to } = fullLocalDayRange(analysisDay);
      setSubmitting(true);
      const job = await api.jobs.create({
        date_from: from,
        date_to: to,
        operator_ids: operatorIds,
        keyword_category_ids: [],
        include_all_speakers: false,
      });
      const id = job.id ?? (job as ProcessingJob & { job_id?: Identifier }).job_id;
      if (id === undefined) throw new Error("The analysis started, but its progress page could not be opened.");
      router.push(`/processing/${id}`);
    } catch (caught) {
      if (caught && typeof caught === "object" && "status" in caught && (caught as { status?: number }).status === 409) {
        try {
          const runningJob = await api.jobs.active();
          if (runningJob && !terminalJobStates.has(runningJob.status)) {
            setActiveJob(runningJob);
            return;
          }
        } catch { /* The original safe API message is shown below. */ }
      }
      setFormError(messageFromError(caught, "The analysis could not be started."));
    } finally {
      setSubmitting(false);
    }
  }

  if (loading) return <LoadingState label="Preparing analysis options" />;
  if (error || !configuration) return <ErrorState message={error || "Configuration could not be loaded."} onRetry={() => void load()} />;

  const integrationsReady = configurationReady(configuration.yeastar.status) && configurationReady(configuration.openai.status);
  const canSubmit = integrationsReady && enabledOperators.length > 0 && !submitting && !activeJob;
  const operatorSearchActive = Boolean(operatorSearch.trim());

  if (activeJob) {
    return (
      <>
        <div className="page-intro"><div><h2>An analysis is already in progress</h2><p>Only one analysis runs at a time, so recordings are processed reliably.</p></div><StatusLabel status={activeJob.status} /></div>
        <ConfigurationBanner configuration={configuration} />
        <section className="flat-panel" aria-labelledby="active-analysis-title">
          <div className="section-header"><div><h2 id="active-analysis-title">Active analysis</h2><p>Wait for this analysis to finish before choosing someone else. You can leave the progress page while it works.</p></div></div>
          <div className="form-actions"><Link className="button primary" href={`/processing/${activeJob.id}`}>View analysis progress</Link><Link className="button secondary" href={`/results?job_id=${encodeURIComponent(String(activeJob.id))}`}>View available results</Link></div>
        </section>
      </>
    );
  }

  return (
    <>
      <div className="page-intro">
        <div><h2>Choose calls to analyze</h2><p>Choose a day and the operators, then analyze their recorded calls.</p></div>
      </div>
      <ConfigurationBanner configuration={configuration} />
      {!enabledOperators.length ? <div className="notice" role="status">No enabled operators are available. <Link href="/operators">Refresh or enable operators</Link> before starting.</div> : null}
      {formError ? <div className="form-error" role="alert">{formError}</div> : null}

      <form className="analysis-form stack-form" onSubmit={submit} noValidate>
        <section className="flat-panel" aria-labelledby="day-title">
          <div className="section-header"><div><h2 id="day-title">Day</h2><p>All recordings from the selected local calendar day are included.</p></div></div>
          <label><span>Day <span className="required">*</span></span><input type="date" value={analysisDay} onChange={(event) => setAnalysisDay(event.target.value)} required /></label>
        </section>

        <section className="flat-panel" aria-labelledby="operator-title">
          <div className="section-header"><div><h2 id="operator-title">Operators</h2><p>Only enabled operators can be selected.</p></div></div>
          <label htmlFor="operator-search"><span>Search operators</span><input id="operator-search" type="search" value={operatorSearch} onChange={(event) => setOperatorSearch(event.target.value)} placeholder="Name, extension, or email" autoComplete="off" aria-describedby="operator-search-help" /></label>
          <p className="field-help" id="operator-search-help">{operatorSearchActive ? `${visibleOperators.length} matching operator${visibleOperators.length === 1 ? "" : "s"}.` : "Search by name, extension, or email."}</p>
          <fieldset aria-describedby="operator-selection-help">
            <legend className="sr-only">Select operators</legend>
            {enabledOperators.length ? <div className="choice-toolbar"><button className="text-button" type="button" disabled={!visibleOperators.length} onClick={() => setOperatorIds(visibleOperators.map((operator) => operator.id))}>{operatorSearchActive ? "Select matching operators" : "Select all operators"}</button><button className="text-button" type="button" disabled={!operatorIds.length} onClick={() => setOperatorIds([])}>Clear selection</button></div> : null}
            <p className="field-help" id="operator-selection-help" aria-live="polite">{operatorIds.length ? `${operatorIds.length} operator${operatorIds.length === 1 ? "" : "s"} selected.` : "No operators selected."}</p>
            <div className="checkbox-list" id="operator-selection-list">
              {visibleOperators.length ? visibleOperators.map((operator) => <label key={String(operator.id)}><input type="checkbox" checked={operatorIds.some((id) => String(id) === String(operator.id))} onChange={() => toggle(operatorIds, operator.id, setOperatorIds)} /><span>{operator.display_name} <span className="subtle">Extension {operator.extension_number}</span></span></label>) : <p className="table-empty" role="status">No enabled operators match this search.</p>}
            </div>
          </fieldset>
        </section>

        <div className="form-actions">
          <button className="button primary" type="submit" disabled={!canSubmit}>{submitting ? "Starting analysis…" : "Analyze calls"}</button>
          <span className="field-help">Each matching recording is transcribed. Saved keywords are checked automatically against safely identified operator speech; you can always <Link href="/results">search any word in the transcripts</Link> afterward.</span>
          {!integrationsReady ? <span className="field-help">This action is disabled until both connections are configured and ready.</span> : null}
        </div>
      </form>
    </>
  );
}

function fullLocalDayRange(day: string): { from: string; to: string } {
  const match = /^(\d{4})-(\d{2})-(\d{2})$/.exec(day);
  if (!match) throw new Error("Choose a valid day.");

  const year = Number(match[1]);
  const month = Number(match[2]);
  const date = Number(match[3]);
  const calendarDay = new Date(Date.UTC(year, month - 1, date));
  if (
    calendarDay.getUTCFullYear() !== year
    || calendarDay.getUTCMonth() !== month - 1
    || calendarDay.getUTCDate() !== date
  ) {
    throw new Error("Choose a valid day.");
  }

  return {
    from: toIsoDateTime(`${day}T00:00:00`),
    to: toIsoDateTime(`${day}T23:59:59`),
  };
}
