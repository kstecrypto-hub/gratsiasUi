export function SortButton({
  label,
  column,
  activeColumn,
  order,
  onSort,
}: {
  label: string;
  column: string;
  activeColumn: string;
  order: "asc" | "desc";
  onSort: (column: string) => void;
}) {
  const active = activeColumn === column;
  return (
    <button className="sort-button" type="button" onClick={() => onSort(column)} aria-label={`Sort by ${label}${active ? `, currently ${order === "asc" ? "ascending" : "descending"}` : ""}`}>
      {label}<span aria-hidden="true">{active ? (order === "asc" ? " ↑" : " ↓") : ""}</span>
    </button>
  );
}
