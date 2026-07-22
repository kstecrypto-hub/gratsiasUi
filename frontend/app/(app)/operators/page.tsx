"use client";

import { useCallback, useEffect, useMemo, useState } from "react";
import { ConfigurationBanner } from "@/components/configuration-banner";
import { ErrorState, LoadingState, TableEmpty } from "@/components/page-state";
import { SortButton } from "@/components/sort-button";
import { StatusLabel } from "@/components/status-label";
import { mayRefreshOperators, phoneSystemStatusMessage } from "@/components/yeastar-connection-status";
import { api, asList, messageFromError, phoneSystemMessageFromError } from "@/lib/api";
import { formatDateTime } from "@/lib/format";
import type { Configuration, Operator, YeastarConnectionStatus } from "@/lib/types";

export default function OperatorsPage() {
  const [operators, setOperators] = useState<Operator[]>([]);
  const [configuration, setConfiguration] = useState<Configuration>();
  const [yeastarStatus, setYeastarStatus] = useState<YeastarConnectionStatus>();
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");
  const [actionError, setActionError] = useState("");
  const [success, setSuccess] = useState("");
  const [syncing, setSyncing] = useState(false);
  const [updatingId, setUpdatingId] = useState<string>();
  const [sortColumn, setSortColumn] = useState("display_name");
  const [sortOrder, setSortOrder] = useState<"asc" | "desc">("asc");

  const load = useCallback(async () => {
    setLoading(true);
    setError("");
    try {
      const [payload, config, phoneStatus] = await Promise.all([
        api.operators.list(),
        api.configuration(),
        api.yeastar.status(),
      ]);
      setOperators(asList(payload));
      setConfiguration(config);
      setYeastarStatus(phoneStatus);
    } catch (caught) { setError(messageFromError(caught)); }
    finally { setLoading(false); }
  }, []);

  useEffect(() => { void load(); }, [load]);

  const sorted = useMemo(() => [...operators].sort((a, b) => {
    const value = (operator: Operator) => sortColumn === "extension_number" ? operator.extension_number : sortColumn === "enabled" ? String(operator.enabled) : sortColumn === "last_synced_at" ? operator.last_synced_at || "" : operator.display_name;
    return value(a).localeCompare(value(b), undefined, { numeric: true, sensitivity: "base" }) * (sortOrder === "asc" ? 1 : -1);
  }), [operators, sortColumn, sortOrder]);

  function sort(column: string) {
    if (sortColumn === column) setSortOrder((current) => current === "asc" ? "desc" : "asc");
    else { setSortColumn(column); setSortOrder("asc"); }
  }

  async function synchronize() {
    setSyncing(true);
    setActionError("");
    setSuccess("");
    try {
      const result = await api.operators.sync();
      setSuccess(typeof result.synchronized === "number" ? `${result.synchronized} operator${result.synchronized === 1 ? "" : "s"} synchronized.` : "Operators refreshed from the phone system.");
      await load();
    } catch (caught) {
      setActionError(phoneSystemMessageFromError(caught, "Operators could not be refreshed from the phone system. Check the connection in Settings."));
    } finally { setSyncing(false); }
  }

  async function toggle(operator: Operator) {
    setUpdatingId(String(operator.id));
    setActionError("");
    try {
      const updated = await api.operators.update(operator.id, !operator.enabled);
      setOperators((current) => current.map((item) => String(item.id) === String(operator.id) ? updated : item));
    } catch (caught) { setActionError(messageFromError(caught, "The operator status could not be changed.")); }
    finally { setUpdatingId(undefined); }
  }

  if (loading && (!configuration || !yeastarStatus)) return <LoadingState label="Loading operators" />;
  if (error || !configuration || !yeastarStatus) return <ErrorState message={error || "Connection status could not be loaded."} onRetry={() => void load()} />;

  const canSync = mayRefreshOperators(yeastarStatus);
  const safeConfiguration: Configuration = {
    ...configuration,
    yeastar: { status: canSync ? "connected" : yeastarStatus.status, message: phoneSystemStatusMessage(yeastarStatus.status) },
  };

  return (
    <>
      <div className="page-intro">
        <div><h2>Phone system operators</h2><p>Refresh the operator list, then enable the people whose calls should be available for analysis.</p></div>
        <button className="button primary" type="button" disabled={!canSync || syncing} onClick={() => void synchronize()}>{syncing ? "Refreshing..." : "Refresh operators"}</button>
      </div>
      <ConfigurationBanner configuration={safeConfiguration} />
      <div className="operator-connection-state" role="status">
        <StatusLabel
          status={yeastarStatus.status === "connected" ? "connected" : canSync ? "warning" : "failed"}
          label={phoneSystemStatusMessage(yeastarStatus.status)}
        />
        <p>{operatorSyncGuidance(yeastarStatus.status)}</p>
      </div>
      {actionError ? <div className="form-error" role="alert">{actionError}</div> : null}
      {success ? <div className="success-message" role="status">{success}</div> : null}
      <div className="table-wrap">
        <table>
          <thead><tr><th scope="col"><SortButton label="Operator name" column="display_name" activeColumn={sortColumn} order={sortOrder} onSort={sort} /></th><th scope="col"><SortButton label="Extension" column="extension_number" activeColumn={sortColumn} order={sortOrder} onSort={sort} /></th><th scope="col"><SortButton label="Enabled" column="enabled" activeColumn={sortColumn} order={sortOrder} onSort={sort} /></th><th scope="col"><SortButton label="Last synchronized" column="last_synced_at" activeColumn={sortColumn} order={sortOrder} onSort={sort} /></th><th scope="col"><span className="sr-only">Action</span></th></tr></thead>
          <tbody>{sorted.length ? sorted.map((operator) => <tr key={String(operator.id)}><td>{operator.display_name}{operator.email ? <div className="subtle">{operator.email}</div> : null}</td><td>{operator.extension_number}</td><td><StatusLabel status={operator.enabled ? "enabled" : "disabled"} /></td><td>{formatDateTime(operator.last_synced_at)}</td><td><button className="button secondary compact" type="button" disabled={updatingId === String(operator.id)} onClick={() => void toggle(operator)}>{updatingId === String(operator.id) ? "Saving..." : operator.enabled ? "Disable operator" : "Enable operator"}</button></td></tr>) : <TableEmpty colSpan={5}>{operatorEmptyMessage(yeastarStatus)}</TableEmpty>}</tbody>
        </table>
      </div>
    </>
  );
}

function operatorSyncGuidance(status: YeastarConnectionStatus["status"]): string {
  if (status === "connected") return "Operators can be refreshed from the connected phone system.";
  if (status === "not_tested") return "Use Test connection in Settings before refreshing operators.";
  if (status === "not_configured") return "Add the connection details in Settings, then use Test connection.";
  if (status === "ip_blocked") return "Ask your IT administrator to remove the server block before using Test connection in Settings.";
  if (status === "ip_not_allowed") return "Ask your IT administrator to check the API IP allowlist before using Test connection in Settings.";
  return "Operator refresh is paused. Resolve the phone-system issue, then use Test connection in Settings.";
}

function operatorEmptyMessage(status: YeastarConnectionStatus): string {
  if (mayRefreshOperators(status)) return "No operators are available. Refresh operators to synchronize them from the phone system.";
  if (status.status === "not_configured") return "No operators are available until the phone system is configured.";
  if (status.status === "not_tested") return "No operators are available until the phone-system connection has been tested.";
  return "No operators are available while the phone-system connection is paused.";
}
