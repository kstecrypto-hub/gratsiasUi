import { statusTone, titleCase } from "@/lib/format";

export function StatusLabel({ status, label }: { status?: string | null; label?: string }) {
  const value = status || "unknown";
  return <span className={`status-label ${statusTone(value)}`}>{label || titleCase(value)}</span>;
}
