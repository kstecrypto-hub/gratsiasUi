"use client";

import Link from "next/link";
import { FormEvent, useCallback, useEffect, useRef, useState } from "react";
import { EmptyState, ErrorState, LoadingState, TableEmpty } from "@/components/page-state";
import { SortButton } from "@/components/sort-button";
import { StatusLabel } from "@/components/status-label";
import { api, asList, asPage, messageFromError, resultQueryString } from "@/lib/api";
import { formatDate, formatDuration, formatTime, titleCase } from "@/lib/format";
import type { KeywordCategory, Operator, Paginated, ProcessingJob, ResultFilters, ResultRow } from "@/lib/types";

const HISTORY_PAGE_SIZE = 50;
const initialFilters: ResultFilters = { page: 1, page_size: 25, sort: "occurred_at", order: "desc" };
const terminalJobStates = new Set(["completed", "completed_with_errors", "failed", "cancelled"]);
const stringFilterKeys = ["date_from", "date_to", "operator_id", "transcript_query", "keyword", "category_id", "direction", "has_matches"] as const;

export default function ResultsPage() {
  const [operators, setOperators] = useState<Operator[]>([]);
  const [categories, setCategories] = useState<KeywordCategory[]>([]);
  const [jobs, setJobs] = useState<ProcessingJob[]>([]);
  const [currentJobId, setCurrentJobId] = useState("");
  const [historyPage, setHistoryPage] = useState(1);
  const [historyTotal, setHistoryTotal] = useState(0);
  const [historyLoading, setHistoryLoading] = useState(false);
  const [historyError, setHistoryError] = useState("");
  const [contextReady, setContextReady] = useState(false);
  const [contextLoading, setContextLoading] = useState(true);
  const [formFilters, setFormFilters] = useState<ResultFilters>(initialFilters);
  const [filters, setFilters] = useState<ResultFilters>(initialFilters);
  const [page, setPage] = useState<Paginated<ResultRow>>();
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");
  const [exporting, setExporting] = useState(false);
  const [exportError, setExportError] = useState("");
  const resultRequest = useRef(0);

  const loadContext = useCallback(async () => {
    setContextLoading(true);
    setContextReady(false);
    setError("");
    try {
      const [operatorPayload, categoryPayload, currentJob, jobPayload] = await Promise.all([
        api.operators.list(),
        api.categories.list(),
        api.jobs.current(),
        api.jobs.list({ page: 1, page_size: HISTORY_PAGE_SIZE }),
      ]);
      setOperators(asList(operatorPayload));
      setCategories(asList(categoryPayload));

      const history = asList(jobPayload);
      const requestedJobId = requestedJobIdFromLocation();
      let requestedJob = history.find((job) => String(job.id) === requestedJobId)
        || (currentJob && String(currentJob.id) === requestedJobId ? currentJob : undefined);
      if (requestedJobId && !requestedJob) {
        try {
          requestedJob = await api.jobs.get(requestedJobId);
        } catch {
          // An invalid or inaccessible history link safely falls back to current work.
        }
      }

      const nextJobs = uniqueJobs([...(currentJob ? [currentJob] : []), ...(requestedJob ? [requestedJob] : []), ...history]);
      const nextCurrentId = currentJob
        ? String(currentJob.id)
        : String(nextJobs.find((job) => job.is_current)?.id ?? nextJobs[0]?.id ?? "");
      const selectedJobId = requestedJob ? String(requestedJob.id) : nextCurrentId;
      const nextFilters = filtersFromLocation(selectedJobId);

      setJobs(nextJobs);
      setCurrentJobId(nextCurrentId);
      setHistoryPage(1);
      setHistoryTotal(Array.isArray(jobPayload) ? history.length : jobPayload.total);
      setFormFilters(nextFilters);
      setFilters(nextFilters);
      replaceResultsLocation(nextFilters);
      if (!selectedJobId) {
        setPage(emptyResultPage());
        setLoading(false);
      }
      setContextReady(true);
    } catch (caught) {
      setError(messageFromError(caught));
      setLoading(false);
    } finally {
      setContextLoading(false);
    }
  }, []);

  const loadResults = useCallback(async () => {
    if (!contextReady) return;
    const requestId = ++resultRequest.current;
    if (!filters.job_id) {
      setPage(emptyResultPage());
      setLoading(false);
      return;
    }
    setLoading(true);
    setError("");
    try {
      const nextPage = asPage(await api.results.list(filters));
      if (requestId === resultRequest.current) setPage(nextPage);
    } catch (caught) {
      if (requestId === resultRequest.current) setError(messageFromError(caught));
    } finally {
      if (requestId === resultRequest.current) setLoading(false);
    }
  }, [contextReady, filters]);

  useEffect(() => { void loadContext(); }, [loadContext]);
  useEffect(() => { void loadResults(); }, [loadResults]);

  function commitFilters(next: ResultFilters, resetPage = false) {
    const applied = sanitizeFilters({ ...next, page: resetPage ? 1 : next.page });
    setFilters(applied);
    replaceResultsLocation(applied);
  }

  function applyFilters(event: FormEvent) {
    event.preventDefault();
    const next = sanitizeFilters({
      ...formFilters,
      transcript_query: formFilters.transcript_query?.trim() || undefined,
      keyword: formFilters.keyword?.trim() || undefined,
      job_id: filters.job_id,
      page: 1,
      page_size: filters.page_size,
      sort: filters.sort,
      order: filters.order,
    });
    setFormFilters(next);
    commitFilters(next);
  }

  function clearFilters() {
    const next = { ...initialFilters, job_id: filters.job_id };
    setFormFilters(next);
    setPage(undefined);
    commitFilters(next);
  }

  function selectJob(jobId: string) {
    const next = { ...filters, job_id: jobId, page: 1 };
    setFormFilters((current) => ({ ...current, job_id: jobId }));
    setPage(undefined);
    commitFilters(next);
  }

  function sort(column: string) {
    const next = {
      ...filters,
      page: 1,
      sort: column,
      order: filters.sort === column && filters.order === "asc" ? "desc" as const : "asc" as const,
    };
    commitFilters(next);
  }

  function changePage(nextPage: number) {
    commitFilters({ ...filters, page: nextPage });
  }

  async function loadOlderJobs() {
    const nextHistoryPage = historyPage + 1;
    setHistoryLoading(true);
    setHistoryError("");
    try {
      const payload = await api.jobs.list({ page: nextHistoryPage, page_size: HISTORY_PAGE_SIZE });
      setJobs((current) => uniqueJobs([...current, ...asList(payload)]));
      setHistoryPage(nextHistoryPage);
      setHistoryTotal(Array.isArray(payload) ? Math.max(historyTotal, historyPage * HISTORY_PAGE_SIZE + payload.length) : payload.total);
    } catch (caught) {
      setHistoryError(messageFromError(caught, "Older analyses could not be loaded."));
    } finally {
      setHistoryLoading(false);
    }
  }

  async function exportCsv() {
    setExportError("");
    setExporting(true);
    try {
      const exportFilters = { ...filters };
      delete exportFilters.page;
      delete exportFilters.page_size;
      await api.results.export(exportFilters);
    } catch (caught) {
      setExportError(messageFromError(caught, "The export could not be prepared."));
    } finally {
      setExporting(false);
    }
  }

  const rows = page?.items || [];
  const totalPages = page ? Math.max(1, page.pages || Math.ceil(page.total / page.page_size)) : 1;
  const selectedJob = jobs.find((job) => String(job.id) === String(filters.job_id));
  const selectedIsCurrent = Boolean(selectedJob && String(selectedJob.id) === currentJobId);
  const currentJobIsActive = Boolean(selectedJob && selectedIsCurrent && !terminalJobStates.has(selectedJob.status));
  const canLoadOlder = historyPage * HISTORY_PAGE_SIZE < historyTotal;
  const callSearch = resultQueryString(filters);
  const savedMatchFilterActive = Boolean(formFilters.keyword?.trim() || formFilters.category_id);

  if (contextLoading) return <LoadingState label="Loading analysis history" />;
  if (error && !contextReady) return <ErrorState message={error} onRetry={() => void loadContext()} />;

  return (
    <>
      <div className="page-intro">
        <div><h2>Analyzed calls</h2><p>Search words spoken in one analysis, then open a call to read its transcript.</p></div>
        <div className="page-actions">
          <button className="button secondary" type="button" disabled={!rows.length || exporting} onClick={() => void exportCsv()}>{exporting ? "Preparing CSV…" : "Export CSV"}</button>
          <Link className="button primary" href="/analyze">Analyze someone else</Link>
        </div>
      </div>
      {exportError ? <div className="form-error" role="alert">{exportError}</div> : null}

      {!jobs.length ? (
        <EmptyState title="No analyses are available" action={<Link className="button primary" href="/analyze">Analyze calls</Link>}>
          <p>Choose a day and one or more operators to create the first set of searchable transcripts.</p>
        </EmptyState>
      ) : (
        <>
          <section className="flat-panel" aria-labelledby="analysis-history-title">
            <div className="section-header"><div><h2 id="analysis-history-title">Analysis</h2><p>The current analysis opens automatically. Older analyses remain saved and searchable.</p></div></div>
            <div className="filter-grid">
              <label><span>Choose analysis</span><select value={String(filters.job_id || "")} onChange={(event) => selectJob(event.target.value)}>{jobs.map((job) => <option key={String(job.id)} value={String(job.id)}>{jobLabel(job, String(job.id) === currentJobId, operators)}</option>)}</select></label>
            </div>
            {selectedJob ? <div className="form-actions"><StatusLabel status={selectedJob.status} /><Link href={`/processing/${selectedJob.id}`}>View analysis details</Link>{!selectedIsCurrent && currentJobId ? <button className="button secondary compact" type="button" onClick={() => selectJob(currentJobId)}>Return to current analysis</button> : null}{canLoadOlder ? <button className="button secondary compact" type="button" disabled={historyLoading} onClick={() => void loadOlderJobs()}>{historyLoading ? "Loading older analyses…" : "Load older analyses"}</button> : null}</div> : null}
            {historyError ? <div className="form-error" role="alert">{historyError}</div> : null}
            {currentJobIsActive ? <div className="notice" role="status">This analysis is still running. Completed calls will appear here; open the analysis details to follow its progress.</div> : null}
          </section>

          <form className="filters" onSubmit={applyFilters} aria-label="Result filters">
            <div className="filter-grid">
              <label className="span-full"><span>Find words in transcripts</span><input value={formFilters.transcript_query || ""} onChange={(event) => setFormFilters((current) => ({ ...current, transcript_query: event.target.value }))} placeholder="For example: cancellation, refund, delivery" aria-describedby="transcript-search-help" /><span className="field-help" id="transcript-search-help">Searches completed call transcripts in the selected analysis. Leave blank to show every analyzed call.</span></label>
              <label><span>From date</span><input type="date" value={formFilters.date_from || ""} onChange={(event) => setFormFilters((current) => ({ ...current, date_from: event.target.value }))} /></label>
              <label><span>To date</span><input type="date" value={formFilters.date_to || ""} onChange={(event) => setFormFilters((current) => ({ ...current, date_to: event.target.value }))} /></label>
              <label><span>Operator</span><select value={formFilters.operator_id || ""} onChange={(event) => setFormFilters((current) => ({ ...current, operator_id: event.target.value }))}><option value="">All operators</option>{operators.map((operator) => <option key={String(operator.id)} value={String(operator.id)}>{operator.display_name}</option>)}</select></label>
            </div>
            <div className="form-actions"><button className="button primary" type="submit">Search calls</button><button className="button secondary" type="button" onClick={clearFilters}>Clear search</button></div>
            <details className="optional-settings">
              <summary>More filters</summary>
              <div className="filter-grid">
                <label><span>Saved keyword</span><input value={formFilters.keyword || ""} onChange={(event) => setFormFilters((current) => ({ ...current, keyword: event.target.value, has_matches: event.target.value.trim() && current.has_matches === "false" ? undefined : current.has_matches }))} /><span className="field-help">Filters phrases saved in the Keywords area.</span></label>
                <label><span>Keyword category</span><select value={formFilters.category_id || ""} onChange={(event) => setFormFilters((current) => ({ ...current, category_id: event.target.value, has_matches: event.target.value && current.has_matches === "false" ? undefined : current.has_matches }))}><option value="">All categories</option>{categories.map((category) => <option key={String(category.id)} value={String(category.id)}>{category.name}</option>)}</select></label>
                <label><span>Direction</span><select value={formFilters.direction || ""} onChange={(event) => setFormFilters((current) => ({ ...current, direction: event.target.value }))}><option value="">All directions</option><option value="inbound">Incoming</option><option value="outbound">Outgoing</option></select></label>
                <label><span>Phrase matches</span><select value={formFilters.has_matches || ""} onChange={(event) => setFormFilters((current) => ({ ...current, has_matches: event.target.value }))}><option value="">All calls</option><option value="true">Has saved phrase matches</option><option value="false" disabled={savedMatchFilterActive}>No saved phrase matches</option></select>{savedMatchFilterActive ? <span className="field-help">A saved keyword or category only applies to calls with phrase matches.</span> : null}</label>
              </div>
            </details>
          </form>

          {loading && !page ? <LoadingState label="Loading results" /> : error ? <ErrorState message={error} onRetry={() => void loadResults()} /> : (
            <section aria-labelledby="results-table-title">
              <h2 className="sr-only" id="results-table-title">Call results</h2>
              {loading ? <div className="notice" role="status">Updating results…</div> : null}
              <div className="table-wrap">
                <table>
                  <thead><tr>
                    <th scope="col"><SortButton label="Date" column="occurred_at" activeColumn={filters.sort || ""} order={filters.order || "desc"} onSort={sort} /></th>
                    <th scope="col">Time</th>
                    <th scope="col"><SortButton label="Operator" column="operator" activeColumn={filters.sort || ""} order={filters.order || "desc"} onSort={sort} /></th>
                    <th scope="col">Phone number</th>
                    <th scope="col"><SortButton label="Duration" column="duration_seconds" activeColumn={filters.sort || ""} order={filters.order || "desc"} onSort={sort} /></th>
                    <th scope="col">Keywords found</th>
                    <th scope="col"><SortButton label="Matches" column="match_count" activeColumn={filters.sort || ""} order={filters.order || "desc"} onSort={sort} /></th>
                    <th scope="col">Status</th>
                    <th scope="col"><span className="sr-only">Action</span></th>
                  </tr></thead>
                  <tbody>
                    {rows.length ? rows.map((row, index) => {
                      const occurredAt = row.occurred_at || row.started_at;
                      const keywords = row.keywords_found || row.keywords || [];
                      const rowKey = `${String(row.call_id)}:${String(row.operator_id ?? index)}`;
                      const callHref = `/calls/${row.call_id}${callSearch ? `?${callSearch}` : ""}`;
                      return <tr key={rowKey}><td>{row.date || formatDate(occurredAt)}</td><td>{row.time || formatTime(occurredAt)}</td><td>{row.operator_name || "—"}</td><td>{row.masked_phone_number || "—"}</td><td>{formatDuration(row.duration_seconds)}</td><td>{keywords.length ? <div className="keyword-list">{keywords.slice(0, 3).map((keyword) => <span className="keyword-token" key={keyword}>{keyword}</span>)}{keywords.length > 3 ? <span className="subtle">+{keywords.length - 3}</span> : null}</div> : "—"}</td><td>{row.match_count ?? 0}</td><td><StatusLabel status={row.processing_status || row.status} /></td><td><Link href={callHref}>View call</Link></td></tr>;
                    }) : <TableEmpty colSpan={9}>No calls found. Try another word or clear the search.</TableEmpty>}
                  </tbody>
                </table>
              </div>
              {page && page.total > 0 ? <div className="pagination"><span>Showing {(page.page - 1) * page.page_size + 1}–{Math.min(page.page * page.page_size, page.total)} of {page.total}</span><div className="page-actions"><button className="button secondary compact" type="button" disabled={page.page <= 1 || loading} onClick={() => changePage(Math.max(1, (filters.page || 1) - 1))}>Previous</button><span>Page {page.page} of {totalPages}</span><button className="button secondary compact" type="button" disabled={page.page >= totalPages || loading} onClick={() => changePage((filters.page || 1) + 1)}>Next</button></div></div> : null}
            </section>
          )}
        </>
      )}
    </>
  );
}

function emptyResultPage(): Paginated<ResultRow> {
  return { items: [], total: 0, page: 1, page_size: initialFilters.page_size || 25, pages: 1 };
}

function requestedJobIdFromLocation(): string {
  if (typeof window === "undefined") return "";
  return new URLSearchParams(window.location.search).get("job_id") || "";
}

function filtersFromLocation(jobId: string): ResultFilters {
  const params = typeof window === "undefined" ? new URLSearchParams() : new URLSearchParams(window.location.search);
  const filters: ResultFilters = { ...initialFilters, job_id: jobId || undefined };
  for (const key of stringFilterKeys) {
    const value = params.get(key);
    if (value) filters[key] = value;
  }
  const page = Number(params.get("page"));
  if (Number.isInteger(page) && page > 0) filters.page = page;
  const sort = params.get("sort");
  if (sort && ["occurred_at", "operator", "duration_seconds", "match_count"].includes(sort)) filters.sort = sort;
  const order = params.get("order");
  if (order === "asc" || order === "desc") filters.order = order;
  return sanitizeFilters(filters);
}

function sanitizeFilters(filters: ResultFilters): ResultFilters {
  if ((filters.keyword?.trim() || filters.category_id) && filters.has_matches === "false") {
    return { ...filters, has_matches: undefined };
  }
  return filters;
}

function replaceResultsLocation(filters: ResultFilters) {
  if (typeof window === "undefined") return;
  const query = resultQueryString(filters);
  window.history.replaceState(window.history.state, "", `/results${query ? `?${query}` : ""}`);
}

function uniqueJobs(jobs: ProcessingJob[]): ProcessingJob[] {
  const byId = new Map<string, ProcessingJob>();
  for (const job of jobs) byId.set(String(job.id), job);
  return [...byId.values()].sort((left, right) => String(right.created_at || right.date_from).localeCompare(String(left.created_at || left.date_from)));
}

function jobLabel(job: ProcessingJob, isCurrent: boolean, operators: Operator[]): string {
  const operatorCount = job.operator_ids?.length || job.operators?.length || 0;
  const selectedNames = (job.operator_ids || [])
    .map((id) => operators.find((operator) => String(operator.id) === String(id))?.display_name)
    .filter((name): name is string => Boolean(name));
  const operatorLabel = selectedNames.length
    ? `${selectedNames.slice(0, 2).join(", ")}${selectedNames.length > 2 ? ` +${selectedNames.length - 2}` : ""}`
    : `${operatorCount} operator${operatorCount === 1 ? "" : "s"}`;
  return `${isCurrent ? "Current — " : ""}${formatDate(job.date_from)} — ${operatorLabel} — ${titleCase(job.status)}`;
}
