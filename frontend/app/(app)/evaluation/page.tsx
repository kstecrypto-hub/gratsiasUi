"use client";

import Link from "next/link";
import { useEffect, useState } from "react";
import { ErrorState, LoadingState, TableEmpty } from "@/components/page-state";
import { StatusLabel } from "@/components/status-label";
import { api, messageFromError } from "@/lib/api";
import { formatDuration, titleCase } from "@/lib/format";
import type { EvaluationFilter, EvaluationList } from "@/lib/types";

export default function EvaluationPage() {
  const [filter, setFilter] = useState<EvaluationFilter>("all");
  const [data, setData] = useState<EvaluationList>();
  const [error, setError] = useState("");
  const [retry, setRetry] = useState(0);
  useEffect(() => {
    let active = true;
    setError("");
    setData(undefined);
    api.evaluation.list(filter).then((result) => { if (active) setData(result); })
      .catch((caught) => { if (active) setError(messageFromError(caught)); });
    return () => { active = false; };
  }, [filter, retry]);

  return <>
    <div className="page-intro"><div><h2>Human reference dataset</h2>
      <p>Listen to the selected recordings and write exactly what you hear.</p></div></div>
    <div className="page-actions" role="group" aria-label="Dataset filters">
      {(["all", "dev", "test", "unverified", "verified"] as const).map((value) =>
        <button key={value} type="button" className={filter === value ? "button" : "button secondary"}
          aria-pressed={filter === value} onClick={() => setFilter(value)}>
          {value === "dev" || value === "test" ? value.toUpperCase() : titleCase(value)}
        </button>)}
    </div>
    {error ? <ErrorState message={error} onRetry={() => setRetry((value) => value + 1)} /> : !data ? <LoadingState label="Loading evaluation dataset" /> : <>
      <section className="section" aria-label="Verification progress">
        <div className="section-header"><h2>{data.progress.all.verified} / {data.progress.all.total} verified</h2></div>
        <div className="page-actions">
          <span>DEV: {data.progress.dev.verified} / {data.progress.dev.total}</span>
          <span>TEST: {data.progress.test.verified} / {data.progress.test.total}</span>
        </div>
      </section>
      <section className="section">
        <div className="table-wrap"><table>
          <thead><tr><th>Evaluation ID</th><th>Split</th><th>Duration</th><th>Mode</th><th>Direction</th><th>Quality</th><th>Reference status</th></tr></thead>
          <tbody>{data.items.length ? data.items.map((row) => <tr key={row.evaluation_id}>
            <td><Link href={`/evaluation/${encodeURIComponent(row.evaluation_id)}`}>{row.evaluation_id}</Link></td>
            <td>{row.split.toUpperCase()}</td><td>{formatDuration(row.duration_seconds)}</td>
            <td>{titleCase(row.mode)}</td><td>{titleCase(row.direction)}</td><td>{titleCase(row.quality)}</td>
            <td><StatusLabel status={row.verification_status} /></td>
          </tr>) : <TableEmpty colSpan={7}>No evaluation recordings match this filter. Local recordings must be selected in the evaluation manifest.</TableEmpty>}</tbody>
        </table></div>
      </section>
    </>}
  </>;
}
