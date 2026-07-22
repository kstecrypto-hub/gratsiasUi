"use client";

import { FormEvent, useCallback, useEffect, useState } from "react";
import { ErrorState, LoadingState } from "@/components/page-state";
import {
  OpenAIConfigurationForm,
  type OpenAIConfigurationField,
  type OpenAIConfigurationFieldErrors,
} from "@/components/openai-configuration-form";
import { StatusLabel } from "@/components/status-label";
import {
  YeastarConfigurationForm,
  type YeastarConfigurationField,
  type YeastarConfigurationFieldErrors,
} from "@/components/yeastar-configuration-form";
import { YeastarConnectionStatus } from "@/components/yeastar-connection-status";
import { ApiError, api, messageFromError, openAIMessageFromError, phoneSystemMessageFromError } from "@/lib/api";
import type {
  Configuration,
  OpenAIConnectionConfiguration,
  OpenAIConnectionConfigurationInput,
  Settings,
  YeastarConnectionConfiguration,
  YeastarConnectionConfigurationInput,
  YeastarConnectionStatus as YeastarStatus,
} from "@/lib/types";
import { setActiveTimezone } from "@/lib/format";

export default function SettingsPage() {
  const [configuration, setConfiguration] = useState<Configuration>();
  const [settings, setSettings] = useState<Settings>();
  const [yeastarStatus, setYeastarStatus] = useState<YeastarStatus>();
  const [yeastarConfiguration, setYeastarConfiguration] = useState<YeastarConnectionConfiguration>();
  const [yeastarConfigurationDraft, setYeastarConfigurationDraft] = useState<YeastarConnectionConfigurationInput>();
  const [openAIConfiguration, setOpenAIConfiguration] = useState<OpenAIConnectionConfiguration>();
  const [openAIConfigurationDraft, setOpenAIConfigurationDraft] = useState<OpenAIConnectionConfigurationInput>();
  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);
  const [savingYeastarConfiguration, setSavingYeastarConfiguration] = useState(false);
  const [savingOpenAIConfiguration, setSavingOpenAIConfiguration] = useState(false);
  const [testingConnection, setTestingConnection] = useState(false);
  const [testingOpenAIConnection, setTestingOpenAIConnection] = useState(false);
  const [resettingConnection, setResettingConnection] = useState(false);
  const [error, setError] = useState("");
  const [saveError, setSaveError] = useState("");
  const [success, setSuccess] = useState("");
  const [connectionError, setConnectionError] = useState("");
  const [connectionSuccess, setConnectionSuccess] = useState("");
  const [configurationFieldErrors, setConfigurationFieldErrors] = useState<YeastarConfigurationFieldErrors>({});
  const [configurationSaveError, setConfigurationSaveError] = useState("");
  const [configurationSaveSuccess, setConfigurationSaveSuccess] = useState("");
  const [openAIConfigurationFieldErrors, setOpenAIConfigurationFieldErrors] = useState<OpenAIConfigurationFieldErrors>({});
  const [openAIConfigurationSaveError, setOpenAIConfigurationSaveError] = useState("");
  const [openAIConfigurationSaveSuccess, setOpenAIConfigurationSaveSuccess] = useState("");

  const load = useCallback(async () => {
    setLoading(true);
    setError("");
    try {
      const [config, currentSettings, phoneStatus, phoneConfiguration, openAIConfiguration] = await Promise.all([
        api.configuration(),
        api.settings.get(),
        api.yeastar.status(),
        api.yeastar.configuration(),
        api.openai.configuration(),
      ]);
      setConfiguration(config);
      setSettings(currentSettings);
      setYeastarStatus(phoneStatus);
      setYeastarConfiguration(phoneConfiguration);
      setYeastarConfigurationDraft(configurationDraftFromSafe(phoneConfiguration));
      setOpenAIConfiguration(openAIConfiguration);
      setOpenAIConfigurationDraft(openAIConfigurationDraftFromSafe());
    } catch (caught) { setError(messageFromError(caught)); }
    finally { setLoading(false); }
  }, []);

  useEffect(() => { void load(); }, [load]);

  async function save(event: FormEvent) {
    event.preventDefault();
    if (!settings) return;
    setSaving(true);
    setSaveError("");
    setSuccess("");
    try {
      const updated = await api.settings.update(settings);
      setActiveTimezone(updated.default_timezone);
      setSettings(updated);
      setSuccess("Settings saved.");
    } catch (caught) { setSaveError(messageFromError(caught, "Settings could not be saved.")); }
    finally { setSaving(false); }
  }

  async function testConnection() {
    if (testingConnection || resettingConnection || savingYeastarConfiguration) return;
    setTestingConnection(true);
    setConnectionError("");
    setConnectionSuccess("");
    try {
      const result = await api.yeastar.testConnection();
      const [phoneStatus, phoneConfiguration] = await Promise.all([
        api.yeastar.status(),
        api.yeastar.configuration(),
      ]);
      setYeastarStatus(phoneStatus);
      setYeastarConfiguration(phoneConfiguration);
      if (result.configurationAccepted && phoneStatus.status === "connected") {
        setConnectionSuccess("Phone system connected.");
      } else {
        setConnectionError(phoneStatus.message || connectionFailureMessage(phoneStatus.status));
      }
    } catch (caught) {
      setConnectionError(phoneSystemMessageFromError(caught, "The phone-system connection could not be tested. Ask your IT administrator to check the connection details."));
      try {
        const latestStatus = await api.yeastar.status();
        setYeastarStatus(latestStatus);
        setConnectionError(latestStatus.message || connectionFailureMessage(latestStatus.status));
      } catch { /* Keep the last safe local state. */ }
    } finally { setTestingConnection(false); }
  }

  async function resetConnection() {
    if (testingConnection || resettingConnection || savingYeastarConfiguration) return;
    const confirmed = window.confirm("Reset the phone-system connection? Call analysis and operator refresh will remain unavailable until Test connection succeeds again.");
    if (!confirmed) return;
    setResettingConnection(true);
    setConnectionError("");
    setConnectionSuccess("");
    try {
      await api.yeastar.resetConnection();
      const [phoneStatus, phoneConfiguration] = await Promise.all([
        api.yeastar.status(),
        api.yeastar.configuration(),
      ]);
      setYeastarStatus(phoneStatus);
      setYeastarConfiguration(phoneConfiguration);
      setConnectionSuccess("Phone-system connection reset. Use Test connection when you are ready to reconnect.");
    } catch (caught) {
      setConnectionError(phoneSystemMessageFromError(caught, "The phone-system connection could not be reset. Try again, or ask your IT administrator for help."));
    } finally { setResettingConnection(false); }
  }

  function updateYeastarConfigurationDraft(value: YeastarConnectionConfigurationInput) {
    setYeastarConfigurationDraft(value);
    setConfigurationFieldErrors({});
    setConfigurationSaveError("");
    setConfigurationSaveSuccess("");
  }

  async function saveYeastarConfiguration() {
    if (!yeastarConfigurationDraft || savingYeastarConfiguration || testingConnection || resettingConnection) return;
    setSavingYeastarConfiguration(true);
    setConfigurationFieldErrors({});
    setConfigurationSaveError("");
    setConfigurationSaveSuccess("");
    try {
      const saved = await api.yeastar.updateConfiguration(yeastarConfigurationDraft);
      setYeastarConfiguration(saved.configuration);
      setYeastarConfigurationDraft(configurationDraftFromSafe(saved.configuration));
      setConnectionError("");
      setConnectionSuccess("");
      setConfigurationSaveSuccess("Connection details saved.");
      // A successful write may invalidate the prior tested connection. Move
      // the badge to a safe local state before refreshing so a failed refresh
      // can never leave stale "Connected" UI behind.
      setYeastarStatus((current) => current ? {
        ...current,
        status: "not_tested",
        configured: true,
        message: "Connection status has not been refreshed. Reload this page before continuing.",
        last_error_reference: null,
      } : current);
      try {
        const [phoneStatus, phoneConfiguration] = await Promise.all([
          api.yeastar.status(),
          api.yeastar.configuration(),
        ]);
        setYeastarStatus(phoneStatus);
        setYeastarConfiguration(phoneConfiguration);
        setYeastarConfigurationDraft(configurationDraftFromSafe(phoneConfiguration));
        setConfigurationSaveSuccess(
          phoneStatus.status === "connected"
            ? "Connection details saved. The tested connection remains active."
            : "Connection details saved. Test connection is required before operator refresh or call analysis.",
        );
      } catch {
        setConfigurationSaveError("Connection details were saved, but the latest status could not be refreshed. Reload this page before continuing.");
      }
    } catch (caught) {
      const fieldErrors = safeConfigurationFieldErrors(caught);
      setConfigurationFieldErrors(fieldErrors);
      setConfigurationSaveError(
        Object.keys(fieldErrors).length
          ? "Review the highlighted connection details."
          : phoneSystemMessageFromError(caught, "Connection details could not be saved. Try again, or ask your IT administrator for help."),
      );
    } finally {
      setSavingYeastarConfiguration(false);
      setYeastarConfigurationDraft((current) => current ? {
        ...current,
        Settings: { ...current.Settings, ClientId: "", ClientSecret: "" },
      } : current);
    }
  }

  function updateOpenAIConfigurationDraft(value: OpenAIConnectionConfigurationInput) {
    setOpenAIConfigurationDraft(value);
    setOpenAIConfigurationFieldErrors({});
    setOpenAIConfigurationSaveError("");
    setOpenAIConfigurationSaveSuccess("");
  }

  async function saveOpenAIConfiguration() {
    if (!openAIConfigurationDraft || savingOpenAIConfiguration || testingOpenAIConnection) return;
    setSavingOpenAIConfiguration(true);
    setOpenAIConfigurationFieldErrors({});
    setOpenAIConfigurationSaveError("");
    setOpenAIConfigurationSaveSuccess("");
    try {
      const saved = await api.openai.updateConfiguration(openAIConfigurationDraft);
      setOpenAIConfiguration(saved.configuration);
      setOpenAIConfigurationDraft(openAIConfigurationDraftFromSafe());
      // Never retain a successful or failed submitted key in client state.
      setConfiguration((current) => current ? {
        ...current,
        openai: {
          status: "not_tested",
          message: "OpenAI configuration was saved. Test the connection before starting an analysis.",
        },
      } : current);
      try {
        const [updatedConfiguration, safeConfiguration] = await Promise.all([
          api.configuration(),
          api.openai.configuration(),
        ]);
        setConfiguration(updatedConfiguration);
        setOpenAIConfiguration(safeConfiguration);
        setOpenAIConfigurationDraft(openAIConfigurationDraftFromSafe());
        setOpenAIConfigurationSaveSuccess(
          openAIConnectionIsReady(updatedConfiguration.openai.status)
            ? "OpenAI API key saved. The transcription service is ready."
            : "OpenAI API key saved. Test the connection before starting an analysis.",
        );
      } catch {
        setOpenAIConfigurationSaveSuccess("OpenAI API key saved. Test the connection before starting an analysis.");
        setOpenAIConfigurationSaveError("The API key was saved, but the latest connection status could not be refreshed. Reload this page before continuing.");
      }
    } catch (caught) {
      const fieldErrors = safeOpenAIConfigurationFieldErrors(caught);
      setOpenAIConfigurationFieldErrors(fieldErrors);
      setOpenAIConfigurationSaveError(
        Object.keys(fieldErrors).length
          ? "Review the highlighted API key."
          : openAIMessageFromError(caught, "The OpenAI API key could not be saved. Try again, or check the key with your administrator."),
      );
    } finally {
      setSavingOpenAIConfiguration(false);
      setOpenAIConfigurationDraft(openAIConfigurationDraftFromSafe());
    }
  }

  async function testOpenAIConnection() {
    if (testingOpenAIConnection || savingOpenAIConfiguration) return;
    setTestingOpenAIConnection(true);
    setOpenAIConfigurationSaveError("");
    setOpenAIConfigurationSaveSuccess("");
    try {
      const result = await api.openai.testConnection();
      setOpenAIConfiguration(result.configuration);
      setConfiguration((current) => current ? { ...current, openai: result.connection } : current);
      try {
        const [updatedConfiguration, safeConfiguration] = await Promise.all([
          api.configuration(),
          api.openai.configuration(),
        ]);
        setConfiguration(updatedConfiguration);
        setOpenAIConfiguration(safeConfiguration);
      } catch { /* The successful test result remains a safe local status. */ }
      if (result.configurationAccepted && openAIConnectionIsReady(result.connection.status)) {
        setOpenAIConfigurationSaveSuccess("OpenAI connection is ready for transcription.");
      } else {
        setOpenAIConfigurationSaveError(result.connection.message || "The OpenAI connection could not be confirmed. Check the API key and try again.");
      }
    } catch (caught) {
      setOpenAIConfigurationSaveError(
        openAIMessageFromError(caught, "The OpenAI connection could not be tested. Check the API key and try again."),
      );
      try {
        const updatedConfiguration = await api.configuration();
        setConfiguration(updatedConfiguration);
      } catch { /* Keep the last safe local status. */ }
    } finally {
      setTestingOpenAIConnection(false);
    }
  }

  if (loading || !settings || !configuration || !yeastarStatus || !yeastarConfiguration || !yeastarConfigurationDraft || !openAIConfiguration || !openAIConfigurationDraft) {
    if (error) return <ErrorState message={error} onRetry={() => void load()} />;
    return <LoadingState label="Loading settings" />;
  }

  const connections = [
    { label: "OpenAI connection", state: configuration.openai },
    { label: "Database", state: configuration.database },
    { label: "Processing service", state: configuration.processing },
  ];

  return (
    <>
      <div className="page-intro"><div><h2>Connections and processing settings</h2><p>Add phone-system and OpenAI connection details here. Existing credentials are never displayed.</p></div></div>

      <YeastarConnectionStatus
        status={yeastarStatus}
        configuration={yeastarConfiguration}
        testing={testingConnection}
        resetting={resettingConnection}
        actionsDisabled={savingYeastarConfiguration}
        actionError={connectionError}
        actionSuccess={connectionSuccess}
        onTest={() => void testConnection()}
        onReset={() => void resetConnection()}
      />

      <YeastarConfigurationForm
        safeConfiguration={yeastarConfiguration}
        value={yeastarConfigurationDraft}
        errors={configurationFieldErrors}
        generalError={configurationSaveError}
        success={configurationSaveSuccess}
        saving={savingYeastarConfiguration}
        disabled={testingConnection || resettingConnection}
        onChange={updateYeastarConfigurationDraft}
        onSubmit={() => void saveYeastarConfiguration()}
      />

      <OpenAIConfigurationForm
        configuration={openAIConfiguration}
        value={openAIConfigurationDraft}
        connection={configuration.openai}
        errors={openAIConfigurationFieldErrors}
        generalError={openAIConfigurationSaveError}
        success={openAIConfigurationSaveSuccess}
        saving={savingOpenAIConfiguration}
        testing={testingOpenAIConnection}
        onChange={updateOpenAIConfigurationDraft}
        onSubmit={() => void saveOpenAIConfiguration()}
        onTest={() => void testOpenAIConnection()}
      />

      <section className="section" aria-labelledby="connections-title">
        <div className="section-header"><div><h2 id="connections-title">Other service status</h2><p>Availability checks for the remaining services used by call analysis.</p></div></div>
        <div className="table-wrap connection-table">
          <table><thead><tr><th scope="col">Service</th><th scope="col">Status</th><th scope="col">Details</th></tr></thead><tbody>{connections.map((connection) => <tr key={connection.label}><td>{connection.label}</td><td><StatusLabel status={connection.state.status} /></td><td>{connection.state.message || connectionDetail(connection.label, connection.state.status)}</td></tr>)}</tbody></table>
        </div>
      </section>

      <section className="section settings-form" aria-labelledby="processing-settings-title">
        <div className="section-header"><div><h2 id="processing-settings-title">Processing settings</h2><p>These settings apply to future processing and retention work.</p></div></div>
        {saveError ? <div className="form-error" role="alert">{saveError}</div> : null}
        {success ? <div className="success-message" role="status">{success}</div> : null}
        <form className="stack-form flat-panel" onSubmit={save}>
          <div className="form-grid">
            <label><span>Default language</span><select value={settings.default_language} onChange={(event) => setSettings({ ...settings, default_language: event.target.value })}>{!["el", "en"].includes(settings.default_language) ? <option value={settings.default_language}>{settings.default_language}</option> : null}<option value="el">Greek</option><option value="en">English</option></select></label>
            <label><span>Default timezone</span><input value={settings.default_timezone} onChange={(event) => setSettings({ ...settings, default_timezone: event.target.value })} aria-describedby="timezone-help" required /><span className="field-help" id="timezone-help">Use an IANA timezone such as Europe/Athens.</span></label>
            <label><span>Transcript retention period</span><input type="number" min={1} max={3650} value={settings.transcript_retention_days} onChange={(event) => setSettings({ ...settings, transcript_retention_days: Number(event.target.value) })} required /><span className="field-help">Number of days to keep transcripts.</span></label>
            <label><span>Maximum simultaneous transcriptions</span><input type="number" min={1} max={16} value={settings.max_parallel_transcriptions} onChange={(event) => setSettings({ ...settings, max_parallel_transcriptions: Number(event.target.value) })} required /><span className="field-help">Keep this low unless the service capacity has been increased.</span></label>
            <label className="span-full"><span>Company vocabulary</span><textarea rows={6} value={settings.company_vocabulary} onChange={(event) => setSettings({ ...settings, company_vocabulary: event.target.value })} /><span className="field-help">Add company, product, and business terms that may help Greek transcription. Do not enter secrets.</span></label>
            <label className="checkbox-row span-full"><input type="checkbox" checked={settings.delete_audio_after_transcription} onChange={(event) => setSettings({ ...settings, delete_audio_after_transcription: event.target.checked })} /><span>Delete temporary audio after successful transcription</span></label>
          </div>
          <div className="form-actions"><button className="button primary" type="submit" disabled={saving}>{saving ? "Saving…" : "Save settings"}</button></div>
        </form>
      </section>
    </>
  );
}

const YEASTAR_CONFIGURATION_FIELDS = new Set<YeastarConfigurationField>([
  "Name",
  "Settings.BaseUrl",
  "Settings.ClientId",
  "Settings.ClientSecret",
  "Settings.DateFormat",
  "Settings.PageSize",
  "Settings.IgnoreSslErrors",
]);

const OPENAI_CONFIGURATION_FIELDS = new Set<OpenAIConfigurationField>(["api_key"]);

function configurationDraftFromSafe(configuration: YeastarConnectionConfiguration): YeastarConnectionConfigurationInput {
  const safeBaseUrl = configuration.Settings.BaseUrl;
  return {
    Name: configuration.Name,
    Settings: {
      BaseUrl: safeBaseUrl.startsWith("https://") || safeBaseUrl.startsWith("http://") ? safeBaseUrl : "",
      ClientId: "",
      ClientSecret: "",
      DateFormat: configuration.Settings.DateFormat || "MM/dd/yyyy HH:mm:ss",
      PageSize: configuration.Settings.PageSize || 500,
      IgnoreSslErrors: configuration.Settings.IgnoreSslErrors,
    },
  };
}

function openAIConfigurationDraftFromSafe(): OpenAIConnectionConfigurationInput {
  // The server returns only a marker, but keeping the draft explicit makes it
  // impossible for a safe marker (or an accidental raw response) to populate
  // the password field.
  return { api_key: "" };
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function safeConfigurationFieldErrors(error: unknown): YeastarConfigurationFieldErrors {
  if (!(error instanceof ApiError) || !isRecord(error.details)) return {};
  const payload = isRecord(error.details.detail) ? error.details.detail : error.details;
  if (!Array.isArray(payload.errors)) return {};

  const result: YeastarConfigurationFieldErrors = {};
  for (const item of payload.errors) {
    if (!isRecord(item) || typeof item.field !== "string" || typeof item.message !== "string") continue;
    if (!YEASTAR_CONFIGURATION_FIELDS.has(item.field as YeastarConfigurationField)) continue;
    result[item.field as YeastarConfigurationField] = item.message;
  }
  return result;
}

function safeOpenAIConfigurationFieldErrors(error: unknown): OpenAIConfigurationFieldErrors {
  if (!(error instanceof ApiError) || !isRecord(error.details)) return {};
  const payload = isRecord(error.details.detail) ? error.details.detail : error.details;
  if (!Array.isArray(payload.errors)) return {};

  const result: OpenAIConfigurationFieldErrors = {};
  for (const item of payload.errors) {
    if (!isRecord(item) || typeof item.field !== "string" || typeof item.message !== "string") continue;
    if (!OPENAI_CONFIGURATION_FIELDS.has(item.field as OpenAIConfigurationField)) continue;
    result[item.field as OpenAIConfigurationField] = item.message;
  }
  return result;
}

function openAIConnectionIsReady(status?: string | null): boolean {
  const normalized = (status || "").toLowerCase().replaceAll("_", " ");
  return normalized === "ready" || normalized === "connected";
}

function connectionFailureMessage(status: YeastarStatus["status"]): string {
  if (status === "auth_rejected") return "The phone-system connection details were rejected. Connection attempts have been paused for safety.";
  if (status === "token_refresh_failed") return "Connection attempts have been paused for safety. Use Test connection only after your IT administrator checks the connection.";
  if (status === "ip_not_allowed") return "This server is not allowed to access the phone system. Ask your IT administrator to check the API IP allowlist.";
  if (status === "ip_blocked") return "The phone system has blocked this server. Do not try again until your IT administrator removes the block.";
  if (status === "api_disabled") return "The phone-system API is not enabled. Ask your IT administrator to enable it before testing again.";
  if (status === "permission_denied") return "The phone-system connection does not have the required permissions. Ask your IT administrator to review extension, CDR, and recording access.";
  if (status === "unsupported_api_version" || status === "unsupported_firmware") return "The installed phone-system version does not support the required call API.";
  if (status === "network_unavailable") return "Could not reach the phone system. Ask your IT administrator to check the network connection.";
  if (status === "temporarily_unavailable") return "The phone system is temporarily unavailable. Try again after your IT administrator confirms it is available.";
  if (status === "not_configured") return "Phone system not configured. Add the connection details in Settings before testing.";
  return "The phone-system connection could not be tested. Ask your IT administrator to check the connection details.";
}

function connectionDetail(label: string, status: string): string {
  const value = status.toLowerCase().replaceAll("_", " ");
  if (value === "not configured") {
    if (label.startsWith("Yeastar")) return "Add the connection details in Settings, then use Test connection.";
    if (label.startsWith("OpenAI")) return "Add an API key in Settings, then test the connection.";
    return "Add the required environment variable and restart the application.";
  }
  if (value === "connected" || value === "ready") return "Available for use.";
  if (value === "unavailable") return "The service did not respond to the latest check.";
  return "Status reported by the application.";
}
