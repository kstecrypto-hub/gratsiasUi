import { expect, Page, test } from "@playwright/test";

type Handler = (page: Page, request: { method: string; pathname: string; search: string; body: unknown; headers: Record<string, string> }) => Promise<{ status?: number; body?: unknown; contentType?: string } | undefined>;

test("retranscription queues the displayed transcript and opens its processing job", async ({ page }) => {
  let submitted: unknown;
  await mockApi(page, async (_page, request) => {
    if (request.pathname === "/features") return { body: { transcription_v2_enabled: true } };
    if (request.pathname === "/calls/call-quality") return { body: {
      id: "call-quality", transcript_id: "transcript-old", processing_status: "completed",
      transcript_segments: [{
        id: "continuous-segment", original_text: "Sample conversation.",
        start_timestamp: 1, end_timestamp: 3, speaker_label: "Unknown",
        speaker_source: "unknown", quality_flags: ["approximate_timestamps", "speaker_alignment_uncertain"],
      }], matches: [],
    } };
    if (request.pathname === "/calls/call-quality/reprocess") {
      expect(request.method).toBe("POST");
      expect(request.headers["x-csrf-token"]).toBe("test-csrf");
      submitted = request.body;
      return { status: 202, body: { id: "quality-job", status: "queued" } };
    }
    if (request.pathname === "/jobs/quality-job") return { body: {
      id: "quality-job", status: "queued", selected_operator_ids: [], items: [],
    } };
  });
  await page.goto("/calls/call-quality");
  await expect(page.getByText("Speaker timestamps are approximate", { exact: false })).toBeVisible();
  await expect(page.getByText("Sample conversation.", { exact: true })).toBeVisible();
  await page.getByRole("button", { name: "Retranscribe audio" }).click();
  await expect(page).toHaveURL(/\/processing\/quality-job$/);
  expect(submitted).toEqual({ transcript_id: "transcript-old", pipeline_version: "pipeline-v2" });
});

test("retranscription keeps the transcript visible when an active job blocks it", async ({ page }) => {
  await mockApi(page, async (_page, request) => {
    if (request.pathname === "/features") return { body: { transcription_v2_enabled: true } };
    if (request.pathname === "/calls/call-quality") return { body: {
      id: "call-quality", transcript_id: "transcript-old", processing_status: "completed",
      transcript_segments: [], matches: [],
    } };
    if (request.pathname === "/calls/call-quality/reprocess") return {
      status: 409, body: { detail: "Another analysis is already active." },
    };
  });
  await page.goto("/calls/call-quality");
  await page.getByRole("button", { name: "Retranscribe audio" }).click();
  await expect(page.getByRole("alert").filter({ hasText: "Another analysis is already active." })).toBeVisible();
  await expect(page.getByRole("button", { name: "Retranscribe audio" })).toBeEnabled();
  await expect(page).toHaveURL(/\/calls\/call-quality$/);
});

test("retranscription stays hidden when the upgraded pipeline is disabled", async ({ page }) => {
  await mockApi(page, async (_page, request) => {
    if (request.pathname === "/features") return { body: { transcription_v2_enabled: false } };
    if (request.pathname === "/calls/call-quality") return { body: {
      id: "call-quality", transcript_id: "transcript-old", processing_status: "completed",
      transcript_segments: [], matches: [],
    } };
  });
  await page.goto("/calls/call-quality");
  await expect(page.getByRole("heading", { name: "Transcript", exact: true })).toBeVisible();
  await expect(page.getByRole("button", { name: "Retranscribe audio" })).toHaveCount(0);
});

async function mockApi(page: Page, handler?: Handler) {
  await page.route("**/api/**", async (route) => {
    const request = route.request();
    const url = new URL(request.url());
    const method = request.method();
    let body: unknown;
    try { body = request.postDataJSON(); } catch { body = request.postData(); }
    const custom = await handler?.(page, { method, pathname: url.pathname.replace(/^\/api/, ""), search: url.search, body, headers: request.headers() });
    if (custom) {
      await route.fulfill({ status: custom.status || 200, contentType: custom.contentType || "application/json", body: custom.contentType === "text/csv" ? String(custom.body || "") : JSON.stringify(custom.body ?? {}) });
      return;
    }
    const defaults: Record<string, unknown> = {
      "GET /auth/me": { id: 1, email: "admin@example.test" },
      "GET /auth/csrf": { csrf_token: "test-csrf" },
      "GET /operators": [],
      "GET /keyword-categories": [],
      "GET /keywords": [],
      "GET /jobs": { items: [], total: 0, page: 1, page_size: 50 },
      "GET /jobs/current": null,
      "GET /jobs/active": null,
      "GET /results": { items: [], total: 0, page: 1, page_size: 25 },
      "GET /settings/yeastar/status": {
        status: "not_configured",
        configured: false,
        last_tested_at: null,
        last_successful_connection_at: null,
        model_name: null,
        firmware_version: null,
        capabilities: { extensions: null, cdr_v2: null, recordings: null },
        message: "Phone system not configured",
      },
      "GET /settings/yeastar/configuration": {
        Name: "",
        Settings: {
          BaseUrl: "",
          ClientId: "[NOT CONFIGURED]",
          ClientSecret: "[NOT CONFIGURED]",
          DateFormat: "MM/dd/yyyy HH:mm:ss",
          PageSize: 500,
          IgnoreSslErrors: true,
        },
      },
      "GET /settings/openai/configuration": {
        api_key: "[NOT CONFIGURED]",
      },
    };
    const key = `${method} ${url.pathname.replace(/^\/api/, "")}`;
    if (key in defaults) await route.fulfill({ status: 200, contentType: "application/json", body: JSON.stringify(defaults[key]) });
    else await route.fulfill({ status: 404, contentType: "application/json", body: JSON.stringify({ detail: "Not found" }) });
  });
}

test("administrator signs in with a CSRF-protected request", async ({ page }) => {
  let signedIn = false;
  let loginRequest: { body: unknown; headers: Record<string, string> } | undefined;
  await mockApi(page, async (_page, request) => {
    if (request.method === "GET" && request.pathname === "/auth/me") return signedIn ? { body: { id: 1, email: "admin@example.test" } } : { status: 401, body: { detail: "Not authenticated" } };
    if (request.method === "POST" && request.pathname === "/auth/login") {
      loginRequest = { body: request.body, headers: request.headers };
      signedIn = true;
      return { body: { user: { id: 1, email: "admin@example.test" }, csrf_token: "rotated-csrf" } };
    }
    if (request.method === "GET" && request.pathname === "/dashboard") return { body: {} };
    if (request.method === "GET" && request.pathname === "/configuration") return { body: { yeastar: { status: "not_configured" }, openai: { status: "not_configured" }, database: { status: "ready" }, processing: { status: "ready" } } };
  });
  await page.goto("/login");
  await page.getByLabel("Email").fill("admin@example.test");
  await page.getByLabel("Password").fill("correct horse battery staple");
  await page.getByRole("button", { name: "Sign in" }).click();
  await expect(page).toHaveURL(/\/dashboard$/);
  expect(loginRequest?.body).toEqual({ email: "admin@example.test", password: "correct horse battery staple" });
  expect(loginRequest?.headers["x-csrf-token"]).toBe("test-csrf");
});

test("dashboard shows genuine empty and missing-configuration states", async ({ page }) => {
  await mockApi(page, async (_page, request) => {
    if (request.method === "GET" && request.pathname === "/dashboard") return { body: {} };
    if (request.method === "GET" && request.pathname === "/configuration") return { body: {
      yeastar: { status: "not_configured" },
      openai: { status: "not_configured" },
      database: { status: "ready" },
      processing: { status: "ready" },
    } };
  });
  await page.goto("/dashboard");
  await expect(page.getByRole("heading", { name: "No calls have been analyzed yet" })).toBeVisible();
  await expect(page.getByText("Phone system not configured")).toBeVisible();
  await expect(page.getByText("Add the connection details in Settings, then use Test connection.")).toBeVisible();
  await expect(page.locator(".metric")).toHaveCount(0);
});

test("missing Yeastar credentials disable operator synchronization", async ({ page }) => {
  await mockApi(page, async (_page, request) => {
    if (request.method === "GET" && request.pathname === "/configuration") return { body: {
      yeastar: { status: "not_configured" },
      openai: { status: "connected" },
      database: { status: "ready" },
      processing: { status: "ready" },
    } };
  });
  await page.goto("/operators");
  await expect(page.getByRole("button", { name: "Refresh operators" })).toBeDisabled();
  await expect(page.getByText("No operators are available until the phone system is configured.")).toBeVisible();
});

test("settings saves the exact safe Yeastar contract before test and reset actions", async ({ page }) => {
  let connected = false;
  let releaseSave: (() => void) | undefined;
  const saveGate = new Promise<void>((resolve) => { releaseSave = resolve; });
  let releaseTest: (() => void) | undefined;
  const testGate = new Promise<void>((resolve) => { releaseTest = resolve; });
  let testRequests = 0;
  let submittedConfiguration: unknown;
  let phoneConfiguration = {
    Name: "Main Yeastar PBX",
    Settings: {
      BaseUrl: "https://pbx.example.test:8088",
      ClientId: "[CONFIGURED]",
      ClientSecret: "[REDACTED]",
      DateFormat: "MM/dd/yyyy HH:mm:ss",
      PageSize: 500,
      IgnoreSslErrors: true,
    },
  };

  await mockApi(page, async (_page, request) => {
    if (request.method === "GET" && request.pathname === "/configuration") return { body: {
      yeastar: { status: connected ? "connected" : "not_tested" },
      openai: { status: "connected" },
      database: { status: "ready" },
      processing: { status: "ready" },
    } };
    if (request.method === "GET" && request.pathname === "/settings") return { body: {
      default_language: "el",
      transcript_retention_days: 90,
      delete_audio_after_transcription: true,
      max_parallel_transcriptions: 3,
      company_vocabulary: "",
      default_timezone: "Europe/Athens",
    } };
    if (request.method === "GET" && request.pathname === "/settings/yeastar/configuration") return { body: phoneConfiguration };
    if (request.method === "GET" && request.pathname === "/settings/yeastar/status") return { body: {
      status: connected ? "connected" : "not_tested",
      configured: true,
      last_tested_at: connected ? "2026-07-15T11:35:00Z" : null,
      last_successful_connection_at: connected ? "2026-07-15T11:35:00Z" : null,
      model_name: connected ? "P-Series Cloud Edition" : null,
      firmware_version: connected ? "84.23.0.123" : null,
      capabilities: { extensions: connected, cdr_v2: connected, recordings: connected },
      message: connected ? "Phone system connected" : "Connection has not been tested",
    } };
    if (request.method === "PUT" && request.pathname === "/settings/yeastar/configuration") {
      submittedConfiguration = request.body;
      await saveGate;
      const submitted = request.body as {
        Name: string;
        Settings: { BaseUrl: string; DateFormat: string; PageSize: number; IgnoreSslErrors: boolean };
      };
      connected = false;
      phoneConfiguration = {
        Name: submitted.Name,
        Settings: {
          BaseUrl: submitted.Settings.BaseUrl,
          ClientId: "[CONFIGURED]",
          ClientSecret: "[REDACTED]",
          DateFormat: submitted.Settings.DateFormat,
          PageSize: submitted.Settings.PageSize,
          IgnoreSslErrors: submitted.Settings.IgnoreSslErrors,
        },
      };
      return { body: { valid: true, errors: [], configuration: phoneConfiguration } };
    }
    if (request.method === "POST" && request.pathname === "/settings/yeastar/test") {
      testRequests += 1;
      await testGate;
      connected = true;
      return { body: {
        configurationAccepted: true,
        configuration: phoneConfiguration,
        connection: { status: "connected", model: "P-Series Cloud Edition", firmwareVersion: "84.23.0.123" },
      } };
    }
    if (request.method === "POST" && request.pathname === "/settings/yeastar/reset") {
      connected = false;
      return { body: {
        status: "not_tested", configured: true, last_tested_at: null, last_successful_connection_at: null,
        model_name: null, firmware_version: null, capabilities: { extensions: null, cdr_v2: null, recordings: null }, message: "Connection has not been tested",
      } };
    }
  });

  await page.goto("/settings");
  await expect(page.getByRole("heading", { name: "Phone system" })).toBeVisible();
  await expect(page.getByText("Certificate verification is disabled for the phone-system connection.")).toBeVisible();
  await expect(page.getByText("Use this only when approved by your IT administrator.")).toBeVisible();
  await expect(page.getByText("Connection has not been tested")).toBeVisible();

  await expect(page.getByLabel("Connection name")).toHaveValue("Main Yeastar PBX");
  await expect(page.getByLabel("Base URL")).toHaveValue("https://pbx.example.test:8088");
  await expect(page.getByLabel("Client ID")).toHaveValue("");
  await expect(page.getByLabel("Client ID")).toHaveAttribute("placeholder", "Configured. Leave blank to keep the existing Client ID.");
  await expect(page.getByLabel("Client Secret")).toHaveValue("");
  await expect(page.getByLabel("Client Secret")).toHaveAttribute("type", "password");
  await expect(page.getByLabel("Client Secret")).toHaveAttribute("placeholder", "Configured. Leave blank to keep the existing Client Secret.");

  await page.getByLabel("Connection name").fill("Athens Support PBX");
  await page.getByLabel("Base URL").fill("https://pbx2.example.test:8088");
  await page.getByLabel("Date format").fill("dd/MM/yyyy HH:mm:ss");
  await page.getByLabel("Page size").fill("250");
  await page.getByLabel("Ignore SSL certificate errors").uncheck();
  await page.getByRole("button", { name: "Save connection details" }).click();

  await expect(page.getByRole("button", { name: "Saving connection details..." })).toBeDisabled();

  expect(submittedConfiguration).toEqual({
    Name: "Athens Support PBX",
    Settings: {
      BaseUrl: "https://pbx2.example.test:8088",
      ClientId: "",
      ClientSecret: "",
      DateFormat: "dd/MM/yyyy HH:mm:ss",
      PageSize: 250,
      IgnoreSslErrors: false,
    },
  });
  releaseSave?.();
  await expect(page.getByText("Connection details saved. Test connection is required before operator refresh or call analysis.")).toBeVisible();
  await expect(page.getByLabel("Client ID")).toHaveValue("");
  await expect(page.getByLabel("Client Secret")).toHaveValue("");
  await expect(page.getByText("Certificate verification is disabled for the phone-system connection.")).not.toBeVisible();

  const testButton = page.getByRole("button", { name: "Test connection" });
  await testButton.click();
  await expect(page.getByRole("button", { name: "Testing connection..." })).toBeDisabled();
  expect(testRequests).toBe(1);
  releaseTest?.();
  await expect(page.locator(".phone-system-summary").getByText("Phone system connected")).toBeVisible();
  await expect(page.getByText("P-Series Cloud Edition").first()).toBeVisible();
  expect(testRequests).toBe(1);

  page.once("dialog", (dialog) => void dialog.accept());
  await page.getByRole("button", { name: "Reset connection" }).click();
  await expect(page.locator(".phone-system-summary").getByText("Connection has not been tested")).toBeVisible();
  await expect(page.getByText("Phone-system connection reset. Use Test connection when you are ready to reconnect.")).toBeVisible();
});

test("settings saves and tests an OpenAI API key without ever rehydrating it", async ({ page }) => {
  let openAIConfigured = false;
  let openAIConnected = false;
  let submittedKey: unknown;
  let testRequests = 0;
  const safeOpenAIConfiguration = { api_key: "[CONFIGURED]" };

  await mockApi(page, async (_page, request) => {
    if (request.method === "GET" && request.pathname === "/configuration") return { body: {
      yeastar: { status: "connected" },
      openai: {
        status: openAIConnected ? "connected" : openAIConfigured ? "ready" : "not_configured",
        message: openAIConnected ? "OpenAI connection confirmed" : undefined,
      },
      database: { status: "ready" },
      processing: { status: "ready" },
    } };
    if (request.method === "GET" && request.pathname === "/settings") return { body: {
      default_language: "el", transcript_retention_days: 90, delete_audio_after_transcription: true,
      max_parallel_transcriptions: 3, company_vocabulary: "", default_timezone: "Europe/Athens",
    } };
    if (request.method === "GET" && request.pathname === "/settings/openai/configuration") {
      return { body: openAIConfigured ? safeOpenAIConfiguration : { api_key: "[NOT CONFIGURED]" } };
    }
    if (request.method === "PUT" && request.pathname === "/settings/openai/configuration") {
      submittedKey = request.body;
      openAIConfigured = true;
      return { body: { valid: true, errors: [], configuration: safeOpenAIConfiguration } };
    }
    if (request.method === "POST" && request.pathname === "/settings/openai/test") {
      testRequests += 1;
      openAIConnected = true;
      return { body: { configurationAccepted: true, configuration: safeOpenAIConfiguration, connection: { status: "connected" } } };
    }
  });

  await page.goto("/settings");
  const apiKey = page.getByLabel("OpenAI API key");
  await expect(apiKey).toHaveAttribute("type", "password");
  await expect(apiKey).toHaveAttribute("autocomplete", "new-password");
  await expect(apiKey).toHaveValue("");
  await expect(page.getByRole("button", { name: "Test OpenAI connection" })).toBeDisabled();

  await apiKey.fill("not-a-real-openai-api-key");
  await page.getByRole("button", { name: "Save OpenAI API key" }).click();
  await expect(page.getByText("OpenAI API key saved. The transcription service is ready.")).toBeVisible();
  expect(submittedKey).toEqual({ api_key: "not-a-real-openai-api-key" });
  await expect(apiKey).toHaveValue("");
  await expect(apiKey).toHaveAttribute("placeholder", "Configured. Leave blank to keep the existing API key.");

  await page.getByRole("button", { name: "Test OpenAI connection" }).click();
  await expect(page.getByText("OpenAI connection is ready for transcription.")).toBeVisible();
  expect(testRequests).toBe(1);
});

test("settings shows only safe OpenAI API-key validation errors and clears a rejected key", async ({ page }) => {
  let putHeaders: Record<string, string> | undefined;
  await mockApi(page, async (_page, request) => {
    if (request.method === "GET" && request.pathname === "/configuration") return { body: {
      yeastar: { status: "not_configured" }, openai: { status: "not_configured" }, database: { status: "ready" }, processing: { status: "ready" },
    } };
    if (request.method === "GET" && request.pathname === "/settings") return { body: {
      default_language: "el", transcript_retention_days: 90, delete_audio_after_transcription: true,
      max_parallel_transcriptions: 3, company_vocabulary: "", default_timezone: "Europe/Athens",
    } };
    if (request.method === "PUT" && request.pathname === "/settings/openai/configuration") {
      putHeaders = request.headers;
      return { status: 422, body: {
        valid: false,
        errors: [{ field: "api_key", message: "Enter a valid OpenAI API key." }],
        configuration: { api_key: "[NOT CONFIGURED]" },
      } };
    }
  });

  await page.goto("/settings");
  const apiKey = page.getByLabel("OpenAI API key");
  await apiKey.fill("sk-rejected-value");
  await page.getByRole("button", { name: "Save OpenAI API key" }).click();

  await expect(page.getByText("Review the highlighted API key.")).toBeVisible();
  await expect(page.getByText("Enter a valid OpenAI API key.")).toBeVisible();
  await expect(apiKey).toHaveValue("");
  expect(putHeaders?.["x-csrf-token"]).toBe("test-csrf");
});

test("settings shows only safe field errors and clears a rejected secret", async ({ page }) => {
  let putHeaders: Record<string, string> | undefined;
  await mockApi(page, async (_page, request) => {
    if (request.method === "GET" && request.pathname === "/configuration") return { body: {
      yeastar: { status: "not_configured" }, openai: { status: "connected" }, database: { status: "ready" }, processing: { status: "ready" },
    } };
    if (request.method === "GET" && request.pathname === "/settings") return { body: {
      default_language: "el", transcript_retention_days: 90, delete_audio_after_transcription: true,
      max_parallel_transcriptions: 3, company_vocabulary: "", default_timezone: "Europe/Athens",
    } };
    if (request.method === "PUT" && request.pathname === "/settings/yeastar/configuration") {
      putHeaders = request.headers;
      return { status: 422, body: {
        valid: false,
        errors: [
          { field: "Settings.BaseUrl", message: "Enter only the phone-system URL, without an API path." },
          { field: "Settings.ClientSecret", message: "Check the Client Secret with your IT administrator." },
        ],
        configuration: {
          Name: "",
          Settings: { BaseUrl: "", ClientId: "[NOT CONFIGURED]", ClientSecret: "[NOT CONFIGURED]", DateFormat: "MM/dd/yyyy HH:mm:ss", PageSize: 500, IgnoreSslErrors: true },
        },
      } };
    }
  });

  await page.goto("/settings");
  await page.getByLabel("Base URL").fill("https://pbx.example.test/openapi");
  await page.getByLabel("Client ID").fill("application-client");
  await page.getByLabel("Client Secret").fill("test-only-secret");
  await page.getByRole("button", { name: "Save connection details" }).click();

  await expect(page.getByText("Review the highlighted connection details.")).toBeVisible();
  await expect(page.getByText("Enter only the phone-system URL, without an API path.")).toBeVisible();
  await expect(page.getByText("Check the Client Secret with your IT administrator.")).toBeVisible();
  await expect(page.getByLabel("Client Secret")).toHaveValue("");
  await expect(page.getByLabel("Client ID")).toHaveValue("");
  expect(putHeaders?.["x-csrf-token"]).toBe("test-csrf");
});

test("settings never keeps a stale connected badge when save refresh fails", async ({ page }) => {
  let saved = false;
  const phoneConfiguration = {
    Name: "Main Yeastar PBX",
    Settings: {
      BaseUrl: "https://pbx.example.test:8088",
      ClientId: "[CONFIGURED]",
      ClientSecret: "[REDACTED]",
      DateFormat: "MM/dd/yyyy HH:mm:ss",
      PageSize: 500,
      IgnoreSslErrors: false,
    },
  };
  await mockApi(page, async (_page, request) => {
    if (request.method === "GET" && request.pathname === "/configuration") return { body: {
      yeastar: { status: "connected" }, openai: { status: "connected" }, database: { status: "ready" }, processing: { status: "ready" },
    } };
    if (request.method === "GET" && request.pathname === "/settings") return { body: {
      default_language: "el", transcript_retention_days: 90, delete_audio_after_transcription: true,
      max_parallel_transcriptions: 3, company_vocabulary: "", default_timezone: "Europe/Athens",
    } };
    if (request.method === "GET" && request.pathname === "/settings/yeastar/configuration") {
      return saved ? { status: 503, body: { detail: "Unavailable" } } : { body: phoneConfiguration };
    }
    if (request.method === "GET" && request.pathname === "/settings/yeastar/status") {
      return saved ? { status: 503, body: { detail: "Unavailable" } } : { body: {
        status: "connected", configured: true, last_tested_at: "2026-07-15T11:35:00Z", last_successful_connection_at: "2026-07-15T11:35:00Z",
        model_name: "P-Series Cloud Edition", firmware_version: "84.23.0.123",
        capabilities: { extensions: true, cdr_v2: true, recordings: true }, message: "Phone system connected",
      } };
    }
    if (request.method === "PUT" && request.pathname === "/settings/yeastar/configuration") {
      saved = true;
      return { body: { valid: true, errors: [], configuration: phoneConfiguration } };
    }
  });

  await page.goto("/settings");
  await expect(page.locator(".phone-system-summary").getByText("Phone system connected")).toBeVisible();
  await page.getByRole("button", { name: "Save connection details" }).click();

  await expect(page.getByText("Connection details were saved, but the latest status could not be refreshed. Reload this page before continuing.")).toBeVisible();
  await expect(page.locator(".phone-system-summary").getByText("Phone system connected")).not.toBeVisible();
  await expect(page.getByText("Connection status has not been refreshed. Reload this page before continuing.")).toBeVisible();
});

test("operator refresh waits for a successful connection test", async ({ page }) => {
  let syncRequests = 0;
  await mockApi(page, async (_page, request) => {
    if (request.method === "GET" && request.pathname === "/configuration") return { body: {
      yeastar: { status: "not_tested" }, openai: { status: "connected" }, database: { status: "ready" }, processing: { status: "ready" },
    } };
    if (request.method === "GET" && request.pathname === "/settings/yeastar/status") return { body: {
      status: "not_tested", configured: true, last_tested_at: null, last_successful_connection_at: null,
      model_name: null, firmware_version: null, capabilities: { extensions: null, cdr_v2: null, recordings: null }, message: "Connection has not been tested",
    } };
    if (request.method === "POST" && request.pathname === "/operators/sync") syncRequests += 1;
  });

  await page.goto("/operators");
  const refresh = page.getByRole("button", { name: "Refresh operators" });
  await expect(refresh).toBeDisabled();
  await expect(page.getByText("Use Test connection in Settings before refreshing operators.")).toBeVisible();
  await expect(page.getByText("No operators are available until the phone-system connection has been tested.")).toBeVisible();
  expect(syncRequests).toBe(0);
});

test("primary navigation starts a full-day transcription-only analysis and filters operators", async ({ page }) => {
  let submitted: Record<string, unknown> | undefined;
  let categoryRequests = 0;
  await mockApi(page, async (_page, request) => {
    if (request.method === "GET" && request.pathname === "/dashboard") return { body: {} };
    if (request.method === "GET" && request.pathname === "/configuration") return { body: {
      yeastar: { status: "connected" }, openai: { status: "connected" }, database: { status: "ready" }, processing: { status: "ready" },
    } };
    if (request.method === "GET" && request.pathname === "/operators") return { body: [
      { id: 7, extension_number: "204", display_name: "Maria", enabled: true },
      { id: 8, extension_number: "315", display_name: "Eleni", enabled: true },
      { id: 9, extension_number: "401", display_name: "Disabled operator", enabled: false },
    ] };
    if (request.method === "GET" && request.pathname === "/keyword-categories") {
      categoryRequests += 1;
      return { body: [{ id: 11, name: "Required wording", active: true, keyword_count: 2 }] };
    }
    if (request.method === "POST" && request.pathname === "/jobs") { submitted = request.body as Record<string, unknown>; return { status: 201, body: { id: "job-123", status: "queued", date_from: String(submitted.date_from), date_to: String(submitted.date_to) } }; }
    if (request.method === "GET" && request.pathname === "/jobs/job-123") return { body: { id: "job-123", status: "queued", date_from: "2026-06-30T21:00:00.000Z", date_to: "2026-07-01T20:59:59.000Z", operator_ids: [7], calls_found: 0, recordings_found: 0, calls_completed: 0, calls_failed: 0 } };
  });

  await page.goto("/dashboard");
  await page.getByRole("link", { name: "Analyze Calls", exact: true }).click();
  await expect(page).toHaveURL(/\/analyze$/);
  await expect(page.getByRole("heading", { name: "Keyword categories" })).toHaveCount(0);
  await expect(page.getByLabel("Direction")).toHaveCount(0);
  await expect(page.getByLabel("From date and time *")).toHaveCount(0);
  await expect(page.getByLabel("To date and time *")).toHaveCount(0);
  await expect(page.getByLabel("Day *")).toHaveAttribute("type", "date");
  await expect(page.getByLabel("Search operators")).toBeVisible();
  await page.getByLabel("Search operators").fill("204");
  await expect(page.getByRole("checkbox", { name: /Maria.*Extension 204/ })).toBeVisible();
  await expect(page.getByRole("checkbox", { name: /Eleni.*Extension 315/ })).toHaveCount(0);
  await page.getByLabel("Day *").fill("2026-07-01");
  await page.getByRole("checkbox", { name: /Maria.*Extension 204/ }).check();
  await expect(page.getByText("1 operator selected.")).toBeVisible();
  await expect(page.getByText("Each matching recording is transcribed. Saved keywords are checked automatically against safely identified operator speech;")).toBeVisible();
  await page.getByRole("button", { name: "Analyze calls" }).click();

  await expect(page).toHaveURL(/\/processing\/job-123$/);
  expect(submitted).toMatchObject({
    date_from: "2026-06-30T21:00:00.000Z",
    date_to: "2026-07-01T20:59:59.000Z",
    operator_ids: [7],
    keyword_category_ids: [],
    recording_available: null,
    include_all_speakers: false,
  });
  expect(categoryRequests).toBe(0);
  await expect(page.locator(".progress-header").getByText("Waiting to start")).toBeVisible();
});

test("results table reports an empty filtered result set", async ({ page }) => {
  const currentJob = { id: "job-current", is_current: true, status: "completed", date_from: "2026-07-01T00:00:00Z", date_to: "2026-07-01T23:59:59Z", operator_ids: [7], created_at: "2026-07-02T00:00:00Z" };
  await mockApi(page, async (_page, request) => {
    if (request.method === "GET" && request.pathname === "/jobs/current") return { body: currentJob };
    if (request.method === "GET" && request.pathname === "/jobs") return { body: { items: [currentJob], total: 1, page: 1, page_size: 50 } };
  });
  await page.goto("/results");
  await expect(page.getByText("No calls found. Try another word or clear the search.")).toBeVisible();
  await expect(page.getByRole("button", { name: "Export CSV" })).toBeDisabled();
});

test("results searches completed transcript text without requiring a saved keyword", async ({ page }) => {
  let transcriptQuery: string | null = null;
  let resultJobId: string | null = null;
  const currentJob = { id: "job-current", is_current: true, status: "completed", date_from: "2026-07-01T00:00:00Z", date_to: "2026-07-01T23:59:59Z", operator_ids: [7], created_at: "2026-07-02T00:00:00Z" };
  await mockApi(page, async (_page, request) => {
    if (request.method === "GET" && request.pathname === "/jobs/current") return { body: currentJob };
    if (request.method === "GET" && request.pathname === "/jobs") return { body: { items: [currentJob], total: 1, page: 1, page_size: 50 } };
    if (request.method === "GET" && request.pathname === "/results") {
      const params = new URLSearchParams(request.search);
      transcriptQuery = params.get("transcript_query");
      resultJobId = params.get("job_id");
      return { body: { items: [], total: 0, page: 1, page_size: 25 } };
    }
  });

  await page.goto("/results");
  await page.getByLabel("Find words in transcripts").fill("refund request");
  await page.getByRole("button", { name: "Search calls" }).click();
  await expect.poll(() => transcriptQuery).toBe("refund request");
  expect(resultJobId).toBe("job-current");
  await expect(page.getByText("More filters")).toBeVisible();
  await expect(page.getByLabel("Saved keyword")).not.toBeVisible();
});

test("results keeps filters scoped to the chosen current or historical analysis", async ({ page }) => {
  let resultParams = new URLSearchParams();
  const currentJob = { id: "job-current", is_current: true, status: "completed", date_from: "2026-07-20T00:00:00Z", date_to: "2026-07-20T23:59:59Z", operator_ids: [7, 8], created_at: "2026-07-21T00:00:00Z" };
  const oldJob = { id: "job-old", is_current: false, status: "completed", date_from: "2026-07-10T00:00:00Z", date_to: "2026-07-10T23:59:59Z", operator_ids: [7], created_at: "2026-07-11T00:00:00Z" };
  await mockApi(page, async (_page, request) => {
    if (request.method === "GET" && request.pathname === "/jobs/current") return { body: currentJob };
    if (request.method === "GET" && request.pathname === "/jobs") return { body: { items: [currentJob, oldJob], total: 2, page: 1, page_size: 50 } };
    if (request.method === "GET" && request.pathname === "/operators") return { body: [
      { id: 7, extension_number: "204", display_name: "Maria", enabled: true },
      { id: 8, extension_number: "315", display_name: "Eleni", enabled: true },
    ] };
    if (request.method === "GET" && request.pathname === "/keyword-categories") return { body: [{ id: 11, name: "Required wording", active: true }] };
    if (request.method === "GET" && request.pathname === "/results") {
      resultParams = new URLSearchParams(request.search);
      return { body: { items: [
        { call_id: "shared-call", operator_id: 7, started_at: "2026-07-20T09:00:00Z", operator_name: "Maria", duration_seconds: 40, match_count: 1, processing_status: "completed" },
        { call_id: "shared-call", operator_id: 8, started_at: "2026-07-20T09:00:00Z", operator_name: "Eleni", duration_seconds: 40, match_count: 0, processing_status: "completed" },
      ], total: 2, page: 1, page_size: 25 } };
    }
  });

  await page.goto("/results");
  await expect(page.getByLabel("Choose analysis")).toHaveValue("job-current");
  await expect(page.getByRole("cell", { name: "Maria", exact: true })).toBeVisible();
  await expect(page.getByRole("cell", { name: "Eleni", exact: true })).toBeVisible();

  await page.getByLabel("Find words in transcripts").fill(" refund request ");
  await page.getByLabel("From date").fill("2026-07-20");
  await page.getByLabel("To date").fill("2026-07-20");
  await page.locator('form[aria-label="Result filters"] select').first().selectOption("7");
  await page.getByText("More filters").click();
  await page.getByLabel("Saved keyword").fill("saved phrase");
  await expect(page.getByLabel("Phrase matches").locator('option[value="false"]')).toHaveAttribute("disabled", "");
  await page.getByLabel("Keyword category").selectOption("11");
  await page.getByLabel("Direction").selectOption("inbound");
  await page.getByLabel("Phrase matches").selectOption("true");
  await page.getByRole("button", { name: "Search calls" }).click();

  await expect.poll(() => resultParams.get("transcript_query")).toBe("refund request");
  expect(Object.fromEntries(resultParams)).toMatchObject({
    job_id: "job-current",
    date_from: "2026-07-20",
    date_to: "2026-07-20",
    operator_id: "7",
    keyword: "saved phrase",
    category_id: "11",
    direction: "inbound",
    has_matches: "true",
  });
  await expect(page.getByRole("link", { name: "View call" }).first()).toHaveAttribute("href", /job_id=job-current.*transcript_query=refund\+request/);

  await page.getByLabel("Choose analysis").selectOption("job-old");
  await expect.poll(() => resultParams.get("job_id")).toBe("job-old");
  await page.getByRole("button", { name: "Clear search" }).click();
  await expect.poll(() => resultParams.get("transcript_query")).toBeNull();
  expect(resultParams.get("job_id")).toBe("job-old");
  expect(resultParams.get("operator_id")).toBeNull();
  await expect(page).toHaveURL(/\/results\?.*job_id=job-old/);
});

test("a slower previous result request cannot overwrite a newer search", async ({ page }) => {
  let slowRequestStarted = false;
  const currentJob = { id: "job-current", is_current: true, status: "completed", date_from: "2026-07-20T00:00:00Z", date_to: "2026-07-20T23:59:59Z", operator_ids: [7], created_at: "2026-07-21T00:00:00Z" };
  await mockApi(page, async (_page, request) => {
    if (request.method === "GET" && request.pathname === "/jobs/current") return { body: currentJob };
    if (request.method === "GET" && request.pathname === "/jobs") return { body: { items: [currentJob], total: 1, page: 1, page_size: 50 } };
    if (request.method === "GET" && request.pathname === "/results") {
      const query = new URLSearchParams(request.search).get("transcript_query");
      if (query === "slow") {
        slowRequestStarted = true;
        await new Promise((resolve) => setTimeout(resolve, 300));
        return { body: { items: [{ call_id: "slow-call", operator_id: 7, operator_name: "Stale result", processing_status: "completed" }], total: 1, page: 1, page_size: 25 } };
      }
      return { body: { items: [{ call_id: "current-call", operator_id: 7, operator_name: "Current result", processing_status: "completed" }], total: 1, page: 1, page_size: 25 } };
    }
  });

  await page.goto("/results");
  await expect(page.getByText("Current result")).toBeVisible();
  await page.getByLabel("Find words in transcripts").fill("slow");
  await page.getByRole("button", { name: "Search calls" }).click();
  await expect.poll(() => slowRequestStarted).toBe(true);
  await page.getByRole("button", { name: "Clear search" }).click();
  await expect(page.getByText("Current result")).toBeVisible();
  await page.waitForTimeout(350);
  await expect(page.getByText("Stale result")).not.toBeVisible();
});

test("analyze directs a nontechnical user to the one active analysis", async ({ page }) => {
  const activeJob = { id: "job-active", is_current: true, status: "waiting_for_connection", date_from: "2026-07-20T00:00:00Z", date_to: "2026-07-20T23:59:59Z", operator_ids: [7], created_at: "2026-07-21T00:00:00Z" };
  await mockApi(page, async (_page, request) => {
    if (request.method === "GET" && request.pathname === "/configuration") return { body: { yeastar: { status: "connected" }, openai: { status: "connected" }, database: { status: "ready" }, processing: { status: "ready" } } };
    if (request.method === "GET" && request.pathname === "/operators") return { body: [{ id: 7, extension_number: "204", display_name: "Maria", enabled: true }] };
    if (request.method === "GET" && request.pathname === "/jobs/active") return { body: activeJob };
  });

  await page.goto("/analyze");
  await expect(page.getByRole("heading", { name: "An analysis is already in progress" })).toBeVisible();
  await expect(page.getByLabel("Day *")).toHaveCount(0);
  await expect(page.getByRole("button", { name: "Analyze calls" })).toHaveCount(0);
  await expect(page.getByRole("link", { name: "View analysis progress" })).toHaveAttribute("href", "/processing/job-active");
  await expect(page.getByRole("link", { name: "View available results" })).toHaveAttribute("href", "/results?job_id=job-active");
});

test("waiting analyses keep polling and link completed results to their saved job", async ({ page }) => {
  let requests = 0;
  let allowCompletion = false;
  await mockApi(page, async (_page, request) => {
    if (request.method === "GET" && request.pathname === "/jobs/job-progress") {
      requests += 1;
      const completed = allowCompletion;
      return { body: { id: "job-progress", status: completed ? "completed" : "waiting_for_connection", date_from: "2026-07-20T00:00:00Z", date_to: "2026-07-20T23:59:59Z", operator_ids: [7], calls_found: 2, recordings_found: 2, calls_completed: completed ? 2 : 0, calls_failed: 0, progress_percent: completed ? 100 : 10 } };
    }
  });

  await page.goto("/processing/job-progress");
  await expect(page.getByRole("button", { name: "Cancel analysis" })).toBeVisible();
  allowCompletion = true;
  await expect.poll(() => requests, { timeout: 5_000 }).toBeGreaterThan(1);
  await expect(page.getByRole("link", { name: "View results" })).toHaveAttribute("href", "/results?job_id=job-progress");
});

test("call detail accepts backend aliases, highlights matches, and seeks authenticated audio", async ({ page }) => {
  let callDetailJobId: string | null = null;
  await page.addInitScript(() => {
    Object.defineProperty(HTMLMediaElement.prototype, "play", { configurable: true, value: () => Promise.resolve() });
  });
  await mockApi(page, async (_page, request) => {
    if (request.method === "GET" && request.pathname === "/calls/call-1") {
      callDetailJobId = new URLSearchParams(request.search).get("job_id");
      return { body: {
      id: "call-1",
      started_at: "2026-07-01T06:00:00Z",
      caller_number: "+302101234567",
      callee_number: "204",
      duration_seconds: 94,
      direction: "inbound",
      queue_name: "Support",
      processing_status: "completed",
      has_audio: true,
      participants: [{ display_name: "Maria" }],
      transcript_segments: [{
        id: "segment-1", speaker_label: "Operator", speaker_source: "stereo_channel", start_seconds: "10.25", end_seconds: "18.5", original_text: "Σας ενημερώνω για την προσφορά σήμερα.",
        matches: [{ id: "match-1", keyword_id: "keyword-1", keyword: "προσφορά", category: "Sales", original_matched_text: "προσφορά", context_before: "για την", context_after: "σήμερα", start_seconds: "12.5", end_seconds: "13.1", match_method: "exact_phrase", match_score: "1" }],
      }],
      processing_history: [],
      } };
    }
    if (request.method === "GET" && request.pathname === "/calls/call-1/audio") return { body: "", contentType: "audio/mpeg" };
  });
  await page.goto("/calls/call-1?job_id=job-current&transcript_query=refund");
  await expect.poll(() => callDetailJobId).toBe("job-current");
  await expect(page.getByText("+302101234567")).toBeVisible();
  await expect(page.locator("mark")).toHaveText("προσφορά");
  const audio = page.locator("audio");
  await expect(audio).toHaveAttribute("src", "/api/calls/call-1/audio");
  await expect(page.getByRole("link", { name: "Back to results" })).toHaveAttribute("href", "/results?job_id=job-current&transcript_query=refund");
  await page.getByRole("button", { name: "Play recording from 0:13" }).click();
  await expect.poll(() => audio.evaluate((element: HTMLAudioElement) => element.currentTime)).toBeCloseTo(12.5, 1);
});

test("call detail shows speaker-assignment controls for unknown dual-channel transcripts", async ({ page }) => {
  await mockApi(page, async (_page, request) => {
    if (request.method === "GET" && request.pathname === "/calls/assign-unknown") return { body: {
      id: "assign-unknown",
      transcript_id: "transcript-assign",
      started_at: "2026-07-01T06:00:00Z",
      direction: "inbound",
      processing_status: "completed",
      transcription_mode: "dual_channel",
      speaker_attribution_status: "channel_unknown",
      speaker_assignment_required: true,
      available_channels: [0, 1],
      participants: [{ operator_id: 7, operator_name: "Maria", extension: "204" }],
      transcript_segments: [
        { id: "segment-0", speaker_label: "Channel A", speaker_source: "unknown", channel_index: 0, start_seconds: 0, end_seconds: 5, original_text: "hello" },
        { id: "segment-1", speaker_label: "Channel B", speaker_source: "unknown", channel_index: 1, start_seconds: 5, end_seconds: 10, original_text: "world" },
      ],
      processing_history: [],
    } };
  });

  await page.goto("/calls/assign-unknown");
  await expect(page.getByText("The operator channel could not be identified automatically.")).toBeVisible();
  await expect(page.getByRole("button", { name: "Operator is Channel A" })).toBeVisible();
  await expect(page.getByRole("button", { name: "Operator is Channel B" })).toBeVisible();
  await expect(page.getByRole("button", { name: "Leave unassigned" })).toBeVisible();
});

test("call detail hides speaker-assignment controls for mono diarization", async ({ page }) => {
  await mockApi(page, async (_page, request) => {
    if (request.method === "GET" && request.pathname === "/calls/mono") return { body: {
      id: "mono",
      started_at: "2026-07-01T06:00:00Z",
      direction: "inbound",
      processing_status: "completed",
      transcription_mode: "mono_diarization",
      speaker_attribution_status: "anonymous_diarization",
      speaker_assignment_required: false,
      available_channels: [],
      transcript_segments: [],
      processing_history: [],
    } };
  });

  await page.goto("/calls/mono");
  await expect(page.getByText("No transcript is available")).toBeVisible();
  await expect(page.getByRole("button", { name: "Operator is Channel A" })).toHaveCount(0);
  await expect(page.getByText("The operator channel could not be identified automatically.")).toHaveCount(0);
});

test("speaker assignment sends the selected channel and reloads the call", async ({ page }) => {
  let submitted: Record<string, unknown> | undefined;
  let getRequests = 0;
  await mockApi(page, async (_page, request) => {
    if (request.method === "GET" && request.pathname === "/calls/assign-send") {
      getRequests += 1;
      return { body: {
        id: "assign-send",
        transcript_id: "transcript-assign",
        started_at: "2026-07-01T06:00:00Z",
        direction: "inbound",
        processing_status: "completed",
        transcription_mode: "dual_channel",
        speaker_attribution_status: "channel_unknown",
        speaker_assignment_required: true,
        available_channels: [0, 1],
        participants: [{ operator_id: 7, operator_name: "Maria", extension: "204" }],
        transcript_segments: [],
        processing_history: [],
      } };
    }
    if (request.method === "PATCH" && request.pathname === "/calls/assign-send/speaker-assignment") {
      submitted = request.body as Record<string, unknown>;
      return { body: {
        transcript_id: "transcript-assign",
        transcription_mode: "dual_channel",
        speaker_attribution_status: "manually_assigned",
        speaker_assignment_required: false,
        available_channels: [0, 1],
        operator_id: 7,
        operator_channel_index: 1,
      } };
    }
  });

  await page.goto("/calls/assign-send");
  await expect(page.getByRole("button", { name: "Operator is Channel B" })).toBeEnabled();
  await page.getByRole("button", { name: "Operator is Channel B" }).click();
  await expect.poll(() => submitted).toEqual({
    transcript_id: "transcript-assign",
    operator_id: "7",
    operator_channel_index: 1,
  });
  await expect.poll(() => getRequests).toBeGreaterThan(1);
});

test("speaker assignment surfaces backend errors without reloading", async ({ page }) => {
  let getRequests = 0;
  await mockApi(page, async (_page, request) => {
    if (request.method === "GET" && request.pathname === "/calls/assign-error") {
      getRequests += 1;
      return { body: {
        id: "assign-error",
        transcript_id: "transcript-assign",
        started_at: "2026-07-01T06:00:00Z",
        direction: "inbound",
        processing_status: "completed",
        transcription_mode: "dual_channel",
        speaker_attribution_status: "channel_unknown",
        speaker_assignment_required: true,
        available_channels: [0, 1],
        participants: [{ operator_id: 7, operator_name: "Maria", extension: "204" }],
        transcript_segments: [],
        processing_history: [],
      } };
    }
    if (request.method === "PATCH" && request.pathname === "/calls/assign-error/speaker-assignment") {
      return { status: 409, body: { detail: "The selected operator was not part of this call." } };
    }
  });

  await page.goto("/calls/assign-error");
  await expect(page.getByRole("button", { name: "Operator is Channel A" })).toBeEnabled();
  const beforeErrorRequests = getRequests;
  await page.getByRole("button", { name: "Operator is Channel A" }).click();
  await expect(page.getByText("The selected operator was not part of this call.")).toBeVisible();
  expect(getRequests).toBe(beforeErrorRequests);
});

test("existing manual assignment asks for confirmation before reassignment", async ({ page }) => {
  let patchRequests = 0;
  await mockApi(page, async (_page, request) => {
    if (request.method === "GET" && request.pathname === "/calls/assign-confirm") return { body: {
      id: "assign-confirm",
      transcript_id: "transcript-assign",
      started_at: "2026-07-01T06:00:00Z",
      direction: "inbound",
      processing_status: "completed",
      transcription_mode: "dual_channel",
      speaker_attribution_status: "manually_assigned",
      speaker_assignment_required: false,
      available_channels: [0, 1],
      participants: [{ operator_id: 7, operator_name: "Maria", extension: "204" }],
      transcript_segments: [],
      processing_history: [],
    } };
    if (request.method === "PATCH" && request.pathname === "/calls/assign-confirm/speaker-assignment") {
      patchRequests += 1;
      return { body: {} };
    }
  });

  await page.goto("/calls/assign-confirm");
  await page.once("dialog", (dialog) => dialog.dismiss());
  await page.getByRole("button", { name: "Operator is Channel B" }).click();
  await expect.poll(() => patchRequests).toBe(0);

  await page.once("dialog", (dialog) => dialog.accept());
  await page.getByRole("button", { name: "Operator is Channel B" }).click();
  await expect.poll(() => patchRequests).toBe(1);
});
