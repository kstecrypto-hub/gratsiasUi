import { StatusLabel } from "@/components/status-label";
import type {
  ConnectionState,
  OpenAIConnectionConfiguration,
  OpenAIConnectionConfigurationInput,
} from "@/lib/types";

export type OpenAIConfigurationField = "api_key";

export type OpenAIConfigurationFieldErrors = Partial<Record<OpenAIConfigurationField, string>>;

type Props = {
  configuration: OpenAIConnectionConfiguration;
  value: OpenAIConnectionConfigurationInput;
  connection: ConnectionState;
  errors: OpenAIConfigurationFieldErrors;
  generalError: string;
  success: string;
  saving: boolean;
  testing: boolean;
  onChange: (value: OpenAIConnectionConfigurationInput) => void;
  onSubmit: () => void;
  onTest: () => void;
};

function isConfigured(value: string): boolean {
  return value === "configured" || value === "[CONFIGURED]" || value === "[REDACTED]";
}

export function OpenAIConfigurationForm({
  configuration,
  value,
  connection,
  errors,
  generalError,
  success,
  saving,
  testing,
  onChange,
  onSubmit,
  onTest,
}: Props) {
  const apiKeyConfigured = isConfigured(configuration.api_key);
  const disabled = saving || testing;

  return (
    <section className="section openai-configuration-section" aria-labelledby="openai-configuration-title">
      <div className="section-header">
        <div>
          <h2 id="openai-configuration-title">OpenAI transcription</h2>
          <p>Add or replace the API key used for transcription. Saved keys are never displayed.</p>
        </div>
      </div>

      <div className="openai-connection-state" role="status">
        <StatusLabel status={connection.status} />
        <p>{connection.message || openAIConnectionDetail(connection.status)}</p>
      </div>

      {generalError ? <div className="form-error" role="alert">{generalError}</div> : null}
      {success ? <div className="success-message" role="status">{success}</div> : null}

      <form
        className="stack-form flat-panel openai-configuration-form"
        autoComplete="off"
        aria-busy={disabled}
        onSubmit={(event) => {
          event.preventDefault();
          onSubmit();
        }}
      >
        <fieldset className="form-grid" disabled={disabled}>
          <label className="span-full">
            <span>OpenAI API key</span>
            <input
              id="openai-api-key"
              type="password"
              autoComplete="new-password"
              spellCheck={false}
              value={value.api_key}
              placeholder={apiKeyConfigured ? "Configured. Leave blank to keep the existing API key." : "Enter an OpenAI API key"}
              required={!apiKeyConfigured}
              aria-invalid={Boolean(errors.api_key)}
              aria-describedby={errors.api_key ? "openai-api-key-error" : "openai-api-key-help"}
              onChange={(event) => onChange({ api_key: event.target.value })}
            />
            {errors.api_key ? (
              <span className="field-error" id="openai-api-key-error">{errors.api_key}</span>
            ) : (
              <span className="field-help" id="openai-api-key-help">
                {apiKeyConfigured
                  ? "Leave this blank to retain the configured API key. Its saved value cannot be viewed here."
                  : "The key is used only by this application and its saved value cannot be viewed here."}
              </span>
            )}
          </label>
        </fieldset>

        <div className="form-actions">
          <button className="button primary" type="submit" disabled={disabled}>
            {saving ? "Saving OpenAI API key..." : "Save OpenAI API key"}
          </button>
          <button className="button secondary" type="button" disabled={disabled || !apiKeyConfigured} onClick={onTest}>
            {testing ? "Testing OpenAI connection..." : "Test OpenAI connection"}
          </button>
        </div>
      </form>
    </section>
  );
}

function openAIConnectionDetail(status?: string | null): string {
  const normalized = (status || "").toLowerCase().replaceAll("_", " ");
  if (normalized.includes("not configured") || normalized.includes("missing")) {
    return "Add an API key above, then test the connection.";
  }
  if (normalized === "ready" || normalized === "connected") return "Available for transcription.";
  if (normalized.includes("unavailable")) return "The transcription service did not respond to the latest check.";
  return "Status reported by the application.";
}
