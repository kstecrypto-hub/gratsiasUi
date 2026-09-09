import type { ReactNode } from "react";

export function CallMetadata({ items }: { items: { label: string; value: ReactNode }[] }) {
  return <dl className="detail-grid">
    {items.map(({ label, value }) => <div key={label}><dt>{label}</dt><dd>{value}</dd></div>)}
  </dl>;
}
