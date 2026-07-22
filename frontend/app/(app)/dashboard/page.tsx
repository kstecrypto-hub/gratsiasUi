"use client";

import Link from "next/link";
import { useCallback, useEffect, useState } from "react";
import { ConfigurationBanner } from "@/components/configuration-banner";
import { EmptyState, ErrorState, LoadingState, TableEmpty } from "@/components/page-state";
import { StatusLabel } from "@/components/status-label";
import { api, messageFromError } from "@/lib/api";
import { formatDateTime } from "@/lib/format";
import type { Configuration, DashboardData } from "@/lib/types";

export default function DashboardPage() {
  const [dashboard, setDashboard] = useState<DashboardData>();
  const [configuration, setConfiguration] = useState<Configuration>();
  const [error, setError] = useState("");

  const load = useCallback(async () => {
    setError("");
    try {
      const [dashboardData, configurationData] = await Promise.all([api.dashboard(), api.configuration()]);
      setDashboard(dashboardData);
      setConfiguration(configurationData);
    } catch (caught) {
      setError(messageFromError(caught));
    }
  }, []);

  useEffect(() => { void load(); }, [load]);

  if (!dashboard || !configuration) {
    if (error) return <ErrorState message={error} onRetry={() => void load()} />;
    return <LoadingState label="Loading dashboard" />;
  }

  const operatorResults = dashboard.results_by_operator || [];
  const categoryResults = dashboard.results_by_keyword_category || [];
  const recentJobs = dashboard.recent_jobs || [];
  const counts = [
    dashboard.calls_analyzed,
    dashboard.calls_with_recordings,
    dashboard.calls_transcribed,
    dashboard.calls_with_matches,
    dashboard.failed_calls,
    dashboard.processing_jobs,
  ];
  const hasActivity = counts.some((count) => typeof count === "number" && count > 0) || operatorResults.length > 0 || categoryResults.length > 0 || recentJobs.length > 0;

  return (
    <>
      <div className="page-intro">
        <div>
          <h2>Call analysis overview</h2>
          <p>Review completed work and start a new analysis when you are ready.</p>
        </div>
        <Link className="button primary" href="/analyze">Analyze calls</Link>
      </div>
      <ConfigurationBanner configuration={configuration} />

      {!hasActivity ? (
        <EmptyState title="No calls have been analyzed yet" action={<Link className="button primary" href="/analyze">Analyze your first period</Link>}>
          <p>After an analysis finishes, call totals and phrase matches will appear here.</p>
        </EmptyState>
      ) : (
        <>
          <dl className="metrics" aria-label="Analysis totals">
            <Metric label="Calls analyzed" value={dashboard.calls_analyzed} />
            <Metric label="Calls with recordings" value={dashboard.calls_with_recordings} />
            <Metric label="Calls transcribed" value={dashboard.calls_transcribed} />
            <Metric label="Calls with matches" value={dashboard.calls_with_matches} />
            <Metric label="Failed calls" value={dashboard.failed_calls} />
            <Metric label="Processing jobs" value={dashboard.processing_jobs} />
          </dl>

          <div className="dashboard-columns">
            <section className="section" aria-labelledby="operator-results-title">
              <div className="section-header"><div><h2 id="operator-results-title">Results by operator</h2><p>Calls with detected phrases.</p></div></div>
              <div className="table-wrap">
                <table>
                  <thead><tr><th scope="col">Operator</th><th scope="col">Results</th></tr></thead>
                  <tbody>
                    {operatorResults.length ? operatorResults.map((row, index) => <tr key={String(row.id ?? row.operator_name ?? index)}><td>{row.operator_name || row.name || "Unknown operator"}</td><td>{row.count}</td></tr>) : <TableEmpty colSpan={2}>No operator results for this period.</TableEmpty>}
                  </tbody>
                </table>
              </div>
            </section>
            <section className="section" aria-labelledby="category-results-title">
              <div className="section-header"><div><h2 id="category-results-title">Results by keyword category</h2><p>Detected phrases grouped by category.</p></div></div>
              <div className="table-wrap">
                <table>
                  <thead><tr><th scope="col">Category</th><th scope="col">Results</th></tr></thead>
                  <tbody>
                    {categoryResults.length ? categoryResults.map((row, index) => <tr key={String(row.id ?? row.category_name ?? index)}><td>{row.category_name || row.name || "Uncategorized"}</td><td>{row.count}</td></tr>) : <TableEmpty colSpan={2}>No category results for this period.</TableEmpty>}
                  </tbody>
                </table>
              </div>
            </section>
          </div>

          {recentJobs.length ? (
            <section className="section" aria-labelledby="recent-jobs-title">
              <div className="section-header"><div><h2 id="recent-jobs-title">Recent analyses</h2><p>Open an analysis to review its progress or retry failures.</p></div></div>
              <div className="table-wrap">
                <table>
                  <thead><tr><th scope="col">Started</th><th scope="col">Date range</th><th scope="col">Status</th><th scope="col">Completed calls</th><th scope="col"><span className="sr-only">Action</span></th></tr></thead>
                  <tbody>{recentJobs.map((job) => <tr key={String(job.id)}><td>{formatDateTime(job.created_at)}</td><td>{formatDateTime(job.date_from)} – {formatDateTime(job.date_to)}</td><td><StatusLabel status={job.status} /></td><td>{job.calls_completed ?? 0}</td><td><Link href={`/processing/${job.id}`}>View progress</Link></td></tr>)}</tbody>
                </table>
              </div>
            </section>
          ) : null}
        </>
      )}
    </>
  );
}

function Metric({ label, value }: { label: string; value?: number }) {
  return <div className="metric"><dt>{label}</dt><dd>{typeof value === "number" ? value.toLocaleString() : "—"}</dd></div>;
}
