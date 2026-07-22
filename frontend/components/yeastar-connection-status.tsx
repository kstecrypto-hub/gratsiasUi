import { StatusLabel } from "@/components/status-label";
import { YeastarTestConnectionButton } from "@/components/yeastar-test-connection-button";
import { formatDateTime } from "@/lib/format";
import type {
  YeastarConnectionConfiguration,
  YeastarConnectionState,
  YeastarConnectionStatus as YeastarStatus,
} from "@/lib/types";

const STATUS_MESSAGES: Record<YeastarConnectionState, string> = {
  not_configured: "Phone system not configured",
  not_tested: "Connection has not been tested",
  connected: "Phone system connected",
  auth_rejected: "The connection details were rejected",
  token_refresh_failed: "Connection attempts have been paused for safety",
  ip_blocked: "The phone system has blocked this server",
  ip_not_allowed: "This server is not permitted to access the phone system",
  api_disabled: "The phone-system API is not enabled",
  permission_denied: "The phone-system connection does not have the required permissions",
  unsupported_api_version: "The installed phone-system API version is not supported",
  unsupported_firmware: "The installed phone-system version is not supported",
  network_unavailable: "Could not reach the phone system",
  temporarily_unavailable: "The phone system is temporarily unavailable",
};

export function phoneSystemStatusMessage(status: YeastarConnectionState): string {
  return STATUS_MESSAGES[status];
}

export function mayRefreshOperators(status: YeastarStatus): boolean {
  return status.configured && status.status === "connected";
}

function statusTone(status: YeastarConnectionState): string {
  if (status === "connected") return "connected";
  if (status === "not_configured" || status === "not_tested") return "warning";
  return "failed";
}

type Props = {
  status: YeastarStatus;
  configuration: YeastarConnectionConfiguration;
  testing: boolean;
  resetting: boolean;
  actionsDisabled: boolean;
  actionError: string;
  actionSuccess: string;
  onTest: () => void;
  onReset: () => void;
};

export function YeastarConnectionStatus({
  status,
  configuration,
  testing,
  resetting,
  actionsDisabled,
  actionError,
  actionSuccess,
  onTest,
  onReset,
}: Props) {
  const message = status.message || phoneSystemStatusMessage(status.status);

  return (
    <section className="section phone-system-section" aria-labelledby="phone-system-title">
      <div className="section-header">
        <div>
          <h2 id="phone-system-title">Phone system</h2>
          <p>Check the Yeastar connection without displaying connection credentials.</p>
        </div>
      </div>

      {configuration.Settings.IgnoreSslErrors ? (
        <div className="ssl-warning" role="note">
          <strong>Certificate verification is disabled for the phone-system connection.</strong>
          <span>Use this only when approved by your IT administrator.</span>
        </div>
      ) : null}

      {actionError ? <div className="form-error" role="alert">{actionError}</div> : null}
      {actionSuccess ? <div className="success-message" role="status">{actionSuccess}</div> : null}

      <div className="phone-system-panel flat-panel">
        <dl className="phone-system-summary">
          <div>
            <dt>Status</dt>
            <dd><StatusLabel status={statusTone(status.status)} label={message} /></dd>
          </div>
          <div>
            <dt>System</dt>
            <dd>{status.model_name || "Not detected"}</dd>
          </div>
          <div>
            <dt>Version</dt>
            <dd>{status.firmware_version || "Not detected"}</dd>
          </div>
          <div>
            <dt>Last checked</dt>
            <dd>{formatDateTime(status.last_tested_at)}</dd>
          </div>
        </dl>

        <div className="form-actions phone-system-actions">
          <YeastarTestConnectionButton
            busy={testing}
            disabled={resetting || actionsDisabled}
            onTest={onTest}
          />
          <button
            className="button danger"
            type="button"
            disabled={testing || resetting || actionsDisabled}
            aria-busy={resetting}
            onClick={onReset}
          >
            {resetting ? "Resetting connection..." : "Reset connection"}
          </button>
        </div>

        <details className="technical-details">
          <summary>Technical details</summary>
          <dl>
            <div><dt>Detected system type</dt><dd>{status.model_name || "Not available"}</dd></div>
            <div><dt>Firmware version</dt><dd>{status.firmware_version || "Not available"}</dd></div>
            <div><dt>Last successful check</dt><dd>{formatDateTime(status.last_successful_connection_at)}</dd></div>
            <div><dt>Sanitized error reference</dt><dd>{status.last_error_reference || "None"}</dd></div>
          </dl>
        </details>
      </div>
    </section>
  );
}
