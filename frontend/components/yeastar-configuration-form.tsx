import type {
  YeastarConnectionConfiguration,
  YeastarConnectionConfigurationInput,
} from "@/lib/types";

export type YeastarConfigurationField =
  | "Name"
  | "Settings.BaseUrl"
  | "Settings.ClientId"
  | "Settings.ClientSecret"
  | "Settings.DateFormat"
  | "Settings.PageSize"
  | "Settings.IgnoreSslErrors";

export type YeastarConfigurationFieldErrors = Partial<Record<YeastarConfigurationField, string>>;

type Props = {
  safeConfiguration: YeastarConnectionConfiguration;
  value: YeastarConnectionConfigurationInput;
  errors: YeastarConfigurationFieldErrors;
  generalError: string;
  success: string;
  saving: boolean;
  disabled: boolean;
  onChange: (value: YeastarConnectionConfigurationInput) => void;
  onSubmit: () => void;
};

function isConfigured(value: string): boolean {
  return value === "configured" || value === "[CONFIGURED]" || value === "[REDACTED]";
}

export function YeastarConfigurationForm({
  safeConfiguration,
  value,
  errors,
  generalError,
  success,
  saving,
  disabled,
  onChange,
  onSubmit,
}: Props) {
  const clientIdConfigured = isConfigured(safeConfiguration.Settings.ClientId);
  const clientSecretConfigured = isConfigured(safeConfiguration.Settings.ClientSecret);
  const updateSettings = (changes: Partial<YeastarConnectionConfigurationInput["Settings"]>) => {
    onChange({ ...value, Settings: { ...value.Settings, ...changes } });
  };

  return (
    <section className="section yeastar-configuration-section" aria-labelledby="yeastar-configuration-title">
      <div className="section-header">
        <div>
          <h2 id="yeastar-configuration-title">Phone-system connection details</h2>
          <p>Saved credentials are never displayed. Leave a configured credential blank to keep its existing value.</p>
        </div>
      </div>

      {generalError ? <div className="form-error" role="alert">{generalError}</div> : null}
      {success ? <div className="success-message" role="status">{success}</div> : null}

      <form
        className="stack-form flat-panel yeastar-configuration-form"
        autoComplete="off"
        aria-busy={saving}
        onSubmit={(event) => {
          event.preventDefault();
          onSubmit();
        }}
      >
        <fieldset className="form-grid" disabled={disabled || saving}>
          <label>
            <span>Connection name</span>
            <input
              id="yeastar-name"
              value={value.Name}
              aria-invalid={Boolean(errors.Name)}
              aria-describedby={errors.Name ? "yeastar-name-error" : "yeastar-name-help"}
              onChange={(event) => onChange({ ...value, Name: event.target.value })}
            />
            {errors.Name ? <span className="field-error" id="yeastar-name-error">{errors.Name}</span> : <span className="field-help" id="yeastar-name-help">A name that helps you identify this phone system.</span>}
          </label>

          <label>
            <span>Base URL</span>
            <input
              id="yeastar-base-url"
              type="url"
              inputMode="url"
              spellCheck={false}
              value={value.Settings.BaseUrl}
              placeholder="https://pbx.example.com:8088"
              required
              aria-invalid={Boolean(errors["Settings.BaseUrl"])}
              aria-describedby={errors["Settings.BaseUrl"] ? "yeastar-base-url-error" : "yeastar-base-url-help"}
              onChange={(event) => updateSettings({ BaseUrl: event.target.value })}
            />
            {errors["Settings.BaseUrl"] ? <span className="field-error" id="yeastar-base-url-error">{errors["Settings.BaseUrl"]}</span> : <span className="field-help" id="yeastar-base-url-help">Enter only the scheme, host, and optional port. Do not add an API path.</span>}
          </label>

          <label>
            <span>Client ID</span>
            <input
              id="yeastar-client-id"
              autoComplete="off"
              spellCheck={false}
              value={value.Settings.ClientId}
              placeholder={clientIdConfigured ? "Configured. Leave blank to keep the existing Client ID." : "Enter the Client ID"}
              required={!clientIdConfigured}
              aria-invalid={Boolean(errors["Settings.ClientId"])}
              aria-describedby={errors["Settings.ClientId"] ? "yeastar-client-id-error" : "yeastar-client-id-help"}
              onChange={(event) => updateSettings({ ClientId: event.target.value })}
            />
            {errors["Settings.ClientId"] ? <span className="field-error" id="yeastar-client-id-error">{errors["Settings.ClientId"]}</span> : <span className="field-help" id="yeastar-client-id-help">{clientIdConfigured ? "Leave blank to keep the configured Client ID." : "Provided by your Yeastar administrator."}</span>}
          </label>

          <label>
            <span>Client Secret</span>
            <input
              id="yeastar-client-secret"
              type="password"
              autoComplete="off"
              spellCheck={false}
              value={value.Settings.ClientSecret}
              placeholder={clientSecretConfigured ? "Configured. Leave blank to keep the existing Client Secret." : "Enter the Client Secret"}
              required={!clientSecretConfigured}
              aria-invalid={Boolean(errors["Settings.ClientSecret"])}
              aria-describedby={errors["Settings.ClientSecret"] ? "yeastar-client-secret-error" : "yeastar-client-secret-help"}
              onChange={(event) => updateSettings({ ClientSecret: event.target.value })}
            />
            {errors["Settings.ClientSecret"] ? <span className="field-error" id="yeastar-client-secret-error">{errors["Settings.ClientSecret"]}</span> : <span className="field-help" id="yeastar-client-secret-help">{clientSecretConfigured ? "Leave blank to keep the configured secret. Its saved value cannot be viewed here." : "Its saved value will never be shown in this interface."}</span>}
          </label>

          <label>
            <span>Date format</span>
            <input
              id="yeastar-date-format"
              spellCheck={false}
              value={value.Settings.DateFormat}
              placeholder="MM/dd/yyyy HH:mm:ss"
              required
              aria-invalid={Boolean(errors["Settings.DateFormat"])}
              aria-describedby={errors["Settings.DateFormat"] ? "yeastar-date-format-error" : "yeastar-date-format-help"}
              onChange={(event) => updateSettings({ DateFormat: event.target.value })}
            />
            {errors["Settings.DateFormat"] ? <span className="field-error" id="yeastar-date-format-error">{errors["Settings.DateFormat"]}</span> : <span className="field-help" id="yeastar-date-format-help">Use the phone system&apos;s format. MM is month; mm is minute.</span>}
          </label>

          <label>
            <span>Page size</span>
            <input
              id="yeastar-page-size"
              type="number"
              min={1}
              max={10000}
              value={value.Settings.PageSize}
              required
              aria-invalid={Boolean(errors["Settings.PageSize"])}
              aria-describedby={errors["Settings.PageSize"] ? "yeastar-page-size-error" : "yeastar-page-size-help"}
              onChange={(event) => updateSettings({ PageSize: Number(event.target.value) })}
            />
            {errors["Settings.PageSize"] ? <span className="field-error" id="yeastar-page-size-error">{errors["Settings.PageSize"]}</span> : <span className="field-help" id="yeastar-page-size-help">Choose a value from 1 to 10,000. The recommended value is 500.</span>}
          </label>

          <label className="checkbox-row span-full" htmlFor="yeastar-ignore-ssl-errors">
            <input
              id="yeastar-ignore-ssl-errors"
              type="checkbox"
              checked={value.Settings.IgnoreSslErrors}
              aria-invalid={Boolean(errors["Settings.IgnoreSslErrors"])}
              aria-describedby={errors["Settings.IgnoreSslErrors"] ? "yeastar-ignore-ssl-errors-error" : "yeastar-ignore-ssl-errors-help"}
              onChange={(event) => updateSettings({ IgnoreSslErrors: event.target.checked })}
            />
            <span>
              Ignore SSL certificate errors
              {errors["Settings.IgnoreSslErrors"] ? <span className="field-error block-help" id="yeastar-ignore-ssl-errors-error">{errors["Settings.IgnoreSslErrors"]}</span> : <span className="field-help block-help" id="yeastar-ignore-ssl-errors-help">This affects only the phone-system connection. Use it only with IT approval.</span>}
            </span>
          </label>
        </fieldset>

        <div className="form-actions">
          <button className="button primary" type="submit" disabled={disabled || saving}>
            {saving ? "Saving connection details..." : "Save connection details"}
          </button>
        </div>
      </form>
    </section>
  );
}
