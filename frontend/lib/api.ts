import type {
  EvaluationDetail,
  EvaluationFilter,
  EvaluationKeyword,
  EvaluationList,
  HumanReference,
  ReferenceDraft,
  Administrator,
  CallDetail,
  ConnectionState,
  Configuration,
  CreateJobInput,
  DashboardData,
  Keyword,
  KeywordCategory,
  KeywordMatch,
  JobListFilters,
  Operator,
  OpenAIConfigurationValidation,
  OpenAIConnectionConfiguration,
  OpenAIConnectionConfigurationInput,
  OpenAIConnectionTestResult,
  Paginated,
  ProcessingJob,
  ResultFilters,
  ResultRow,
  Settings,
  SpeakerAssignmentInput,
  SpeakerAssignmentResult,
  TranscriptSegment,
  YeastarConfigurationValidation,
  YeastarConnectionConfiguration,
  YeastarConnectionConfigurationInput,
  YeastarConnectionStatus,
  YeastarConnectionTestResult,
} from "@/lib/types";

const configuredBase = process.env.NEXT_PUBLIC_API_BASE_URL?.replace(/\/$/, "");
export const API_BASE = configuredBase || "/api";

export class ApiError extends Error {
  constructor(
    message: string,
    public readonly status: number,
    public readonly details?: unknown,
  ) {
    super(message);
    this.name = "ApiError";
  }
}

type JsonRecord = Record<string, unknown>;
let csrfToken: string | null = null;

function isJsonRecord(value: unknown): value is JsonRecord {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function errorMessage(payload: unknown, fallback: string): string {
  if (!isJsonRecord(payload)) return fallback;
  if (typeof payload.detail === "string") return payload.detail;
  if (typeof payload.message === "string") return payload.message;
  if (Array.isArray(payload.detail)) {
    return payload.detail
      .map((item) => (isJsonRecord(item) && typeof item.msg === "string" ? item.msg : null))
      .filter(Boolean)
      .join(" ") || fallback;
  }
  return fallback;
}

async function parseResponse(response: Response): Promise<unknown> {
  if (response.status === 204) return undefined;
  const type = response.headers.get("content-type") || "";
  if (!type.includes("application/json")) return response.text();
  return response.json();
}

async function request<T>(path: string, options: RequestInit = {}): Promise<T> {
  const response = await fetch(apiUrl(path), {
    ...options,
    cache: "no-store",
    credentials: "include",
    headers: {
      Accept: "application/json",
      ...(options.body ? { "Content-Type": "application/json" } : {}),
      ...options.headers,
    },
  });
  const payload = await parseResponse(response);

  if (!response.ok) {
    if (response.status === 401 && typeof window !== "undefined") {
      window.dispatchEvent(new Event("session-expired"));
    }
    throw new ApiError(
      errorMessage(payload, response.status >= 500 ? "The service is temporarily unavailable." : "The request could not be completed."),
      response.status,
      payload,
    );
  }
  return payload as T;
}

export function apiUrl(path: string): string {
  const normalized = path.startsWith("/") ? path : `/${path}`;
  return `${API_BASE}${normalized}`;
}

export async function getCsrfToken(force = false): Promise<string> {
  if (csrfToken && !force) return csrfToken;
  const payload = await request<{ csrf_token: string }>("/auth/csrf");
  csrfToken = payload.csrf_token;
  return csrfToken;
}

async function mutate<T>(path: string, method: "POST" | "PUT" | "PATCH" | "DELETE", body?: unknown): Promise<T> {
  const token = await getCsrfToken();
  try {
    return await request<T>(path, {
      method,
      headers: { "X-CSRF-Token": token },
      ...(body === undefined ? {} : { body: JSON.stringify(body) }),
    });
  } catch (error) {
    if (error instanceof ApiError && error.status === 403) csrfToken = null;
    throw error;
  }
}

export const api = {
  features: () => request<{ evaluation_ui_enabled: boolean; transcription_v2_enabled?: boolean }>("/features"),
  evaluation: {
    list: (filter: EvaluationFilter = "all") => request<EvaluationList>(`/evaluation?filter=${filter}`),
    get: (id: string) => request<EvaluationDetail>(`/evaluation/${encodeURIComponent(id)}`),
    keywords: () => request<EvaluationKeyword[]>("/evaluation/keywords"),
    audioUrl: (id: string, channel?: 0 | 1) => apiUrl(`/evaluation/${encodeURIComponent(id)}/audio${channel === undefined ? "" : `/channel/${channel}`}`),
    save: (id: string, input: ReferenceDraft, revision: string) => mutate<HumanReference>(`/evaluation/${encodeURIComponent(id)}/reference`, "PUT", { ...input, revision }),
    verify: (id: string, revision: string) => mutate<HumanReference>(`/evaluation/${encodeURIComponent(id)}/verify`, "POST", { revision }),
  },
  auth: {
    me: () => request<Administrator>("/auth/me"),
    async login(email: string, password: string) {
      const token = await getCsrfToken(true);
      const payload = await request<Administrator | { user: Administrator; csrf_token?: string }>("/auth/login", {
        method: "POST",
        headers: { "X-CSRF-Token": token },
        body: JSON.stringify({ email, password }),
      });
      if (isJsonRecord(payload) && "user" in payload && isJsonRecord(payload.user)) {
        if (typeof payload.csrf_token === "string") csrfToken = payload.csrf_token;
        return payload.user as Administrator;
      }
      return payload as Administrator;
    },
    async logout() {
      await mutate<void>("/auth/logout", "POST");
      csrfToken = null;
    },
  },
  async configuration() {
    const raw = await request<Configuration & { processing_service?: ConnectionState }>("/configuration");
    return { ...raw, processing: raw.processing || raw.processing_service || { status: "unavailable" } };
  },
  dashboard: async () => normalizeDashboard(await request<DashboardData>("/dashboard")),
  settings: {
    async get() {
      return normalizeSettings(await request<RawSettings>("/settings"));
    },
    async update(input: Settings) {
      const raw = await mutate<RawSettings>("/settings", "PATCH", {
        default_language: input.default_language,
        transcript_retention_days: input.transcript_retention_days,
        delete_audio_after_transcription: input.delete_audio_after_transcription,
        max_parallel_transcriptions: input.max_parallel_transcriptions,
        company_vocabulary: input.company_vocabulary,
        default_timezone: input.default_timezone,
      });
      return normalizeSettings(raw);
    },
  },
  yeastar: {
    status: () => request<YeastarConnectionStatus>("/settings/yeastar/status"),
    configuration: () => request<YeastarConnectionConfiguration>("/settings/yeastar/configuration"),
    async updateConfiguration(input: YeastarConnectionConfigurationInput) {
      return mutate<YeastarConfigurationValidation>("/settings/yeastar/configuration", "PUT", input);
    },
    validateConfiguration: () => request<YeastarConfigurationValidation>("/settings/yeastar/configuration/validate"),
    testConnection: () => mutate<YeastarConnectionTestResult>("/settings/yeastar/test", "POST"),
    resetConnection: () => mutate<YeastarConnectionStatus>("/settings/yeastar/reset", "POST"),
  },
  openai: {
    async configuration() {
      return normalizeOpenAIConfiguration(await request<RawOpenAIConnectionConfiguration>("/settings/openai/configuration"));
    },
    async updateConfiguration(input: OpenAIConnectionConfigurationInput) {
      const response = await mutate<RawOpenAIConfigurationValidation>("/settings/openai/configuration", "PUT", input);
      return normalizeOpenAIConfigurationValidation(response);
    },
    async testConnection() {
      return normalizeOpenAIConnectionTestResult(
        await mutate<RawOpenAIConnectionTestResult>("/settings/openai/test", "POST"),
      );
    },
  },
  operators: {
    list: () => request<Operator[] | Paginated<Operator>>("/operators"),
    sync: () => mutate<{ synchronized?: number; operators?: Operator[] }>("/operators/sync", "POST"),
    update: (id: string | number, enabled: boolean) => mutate<Operator>(`/operators/${encodeURIComponent(id)}`, "PATCH", { enabled }),
  },
  categories: {
    list: () => request<KeywordCategory[] | Paginated<KeywordCategory>>("/keyword-categories"),
    create: (input: Pick<KeywordCategory, "name" | "description">) => mutate<KeywordCategory>("/keyword-categories", "POST", input),
    update: (id: string | number, input: Partial<KeywordCategory>) => mutate<KeywordCategory>(`/keyword-categories/${encodeURIComponent(id)}`, "PATCH", input),
    remove: (id: string | number) => mutate<void>(`/keyword-categories/${encodeURIComponent(id)}`, "DELETE"),
  },
  keywords: {
    async list() {
      const payload = await request<Keyword[] | Paginated<Keyword>>("/keywords");
      if (Array.isArray(payload)) return payload.map(normalizeKeyword);
      return { ...payload, items: payload.items.map(normalizeKeyword) };
    },
    async create(input: Omit<Keyword, "id" | "category_name">) {
      return normalizeKeyword(await mutate<Keyword>("/keywords", "POST", input));
    },
    async update(id: string | number, input: Partial<Keyword>) {
      return normalizeKeyword(await mutate<Keyword>(`/keywords/${encodeURIComponent(id)}`, "PATCH", input));
    },
    remove: (id: string | number) => mutate<void>(`/keywords/${encodeURIComponent(id)}`, "DELETE"),
  },
  jobs: {
    async list(filters: JobListFilters = {}) {
      const query = resultQueryString(filters);
      const payload = await request<ProcessingJob[] | Paginated<ProcessingJob>>(`/jobs${query ? `?${query}` : ""}`);
      if (Array.isArray(payload)) return payload.map(normalizeJob);
      return { ...payload, items: payload.items.map(normalizeJob) };
    },
    async current() {
      const payload = await request<ProcessingJob | null>("/jobs/current");
      return payload ? normalizeJob(payload) : null;
    },
    async active() {
      const payload = await request<ProcessingJob | null>("/jobs/active");
      return payload ? normalizeJob(payload) : null;
    },
    async create(input: CreateJobInput) {
      const { recording_availability, ...rest } = input;
      const recording_available = recording_availability === "available" ? true : recording_availability === "unavailable" ? false : null;
      return normalizeJob(await mutate<ProcessingJob>("/jobs", "POST", { ...rest, recording_available }));
    },
    async get(id: string | number) { return normalizeJob(await request<ProcessingJob>(`/jobs/${encodeURIComponent(id)}`)); },
    async retry(id: string | number) { return normalizeJob(await mutate<ProcessingJob>(`/jobs/${encodeURIComponent(id)}/retry`, "POST")); },
    async cancel(id: string | number) { return normalizeJob(await mutate<ProcessingJob>(`/jobs/${encodeURIComponent(id)}/cancel`, "POST")); },
  },
  results: {
    list(filters: ResultFilters) {
      return request<Paginated<ResultRow> | ResultRow[]>(`/results?${resultQueryString(filters)}`);
    },
    async export(filters: ResultFilters) {
      const response = await fetch(apiUrl(`/results/export.csv?${resultQueryString(filters)}`), {
        credentials: "include",
        headers: { Accept: "text/csv" },
      });
      if (!response.ok) {
        const payload = await parseResponse(response);
        throw new ApiError(errorMessage(payload, "The export could not be prepared."), response.status, payload);
      }
      const blob = await response.blob();
      const url = URL.createObjectURL(blob);
      const link = document.createElement("a");
      link.href = url;
      link.download = "call-results.csv";
      document.body.appendChild(link);
      link.click();
      link.remove();
      URL.revokeObjectURL(url);
    },
  },
  calls: {
    async get(id: string | number, jobId?: string) {
      const query = jobId ? `?job_id=${encodeURIComponent(jobId)}` : "";
      return normalizeCall(await request<CallDetail>(`/calls/${encodeURIComponent(id)}${query}`));
    },
    retry: (id: string | number) => mutate<{ message: string }>(`/calls/${encodeURIComponent(id)}/retry`, "POST"),
    reprocess: (id: string | number, transcriptId: string | number) =>
      mutate<ProcessingJob>(`/calls/${encodeURIComponent(id)}/reprocess`, "POST", {
        transcript_id: transcriptId,
        pipeline_version: "pipeline-v2",
      }),
    assignSpeaker: (id: string | number, input: SpeakerAssignmentInput) =>
      mutate<SpeakerAssignmentResult>(`/calls/${encodeURIComponent(id)}/speaker-assignment`, "PATCH", input),
    audioUrl: (id: string | number) => apiUrl(`/calls/${encodeURIComponent(id)}/audio`),
  },
};

export function resultQueryString(filters: Record<string, unknown>): string {
  const params = new URLSearchParams();
  for (const [key, value] of Object.entries(filters)) {
    if (value !== undefined && value !== null && value !== "") params.set(key, String(value));
  }
  return params.toString();
}

export function asList<T>(payload: T[] | Paginated<T> | undefined | null): T[] {
  if (!payload) return [];
  return Array.isArray(payload) ? payload : Array.isArray(payload.items) ? payload.items : [];
}

export function asPage<T>(payload: T[] | Paginated<T>): Paginated<T> {
  if (Array.isArray(payload)) return { items: payload, total: payload.length, page: 1, page_size: payload.length || 25, pages: 1 };
  return {
    items: Array.isArray(payload.items) ? payload.items : [],
    total: Number(payload.total) || 0,
    page: Number(payload.page) || 1,
    page_size: Number(payload.page_size) || 25,
    pages: payload.pages,
  };
}

export function messageFromError(error: unknown, fallback = "Something went wrong. Try again."): string {
  return error instanceof Error && error.message ? error.message : fallback;
}

export function phoneSystemMessageFromError(error: unknown, fallback: string): string {
  if (!(error instanceof ApiError)) return fallback;
  if (error.status === 400 || error.status === 422) return "The phone-system configuration needs attention. Check the settings with your IT administrator.";
  if (error.status === 409) return "Another phone-system connection action is already running. Try again shortly.";
  if (error.status === 423) return "Connection attempts have been paused for safety. Ask your IT administrator to resolve the connection issue, then use Test connection.";
  if (error.status === 502 || error.status === 504) return "Could not reach the phone system. Check the network connection with your IT administrator.";
  if (error.status === 503) return "The phone system is temporarily unavailable. Try again after your IT administrator confirms it is available.";
  return fallback;
}

export function openAIMessageFromError(error: unknown, fallback: string): string {
  if (!(error instanceof ApiError)) return fallback;
  if (error.status === 400 || error.status === 422) return "The OpenAI API key needs attention. Check it and try again.";
  if (error.status === 401 || error.status === 403) return "The OpenAI API key was rejected. Check the key and its project permissions.";
  if (error.status === 409) return "Another OpenAI connection action is already running. Try again shortly.";
  if (error.status === 429) return "OpenAI is rate limiting this connection test. Try again later.";
  if (error.status === 502 || error.status === 503 || error.status === 504) return "The OpenAI service is temporarily unavailable. Try again later.";
  return fallback;
}

type RawSettings = Partial<Settings> & { maximum_simultaneous_transcriptions?: number };

type RawOpenAIConnectionConfiguration = {
  api_key?: unknown;
  ApiKey?: unknown;
  configured?: unknown;
};

type RawOpenAIConfigurationValidation = Partial<OpenAIConfigurationValidation> & {
  configuration?: RawOpenAIConnectionConfiguration;
  errors?: unknown;
};

type RawOpenAIConnectionTestResult = Partial<OpenAIConnectionTestResult> & {
  configuration?: RawOpenAIConnectionConfiguration;
  connection?: unknown;
};

function normalizeOpenAIConfiguration(raw: RawOpenAIConnectionConfiguration): OpenAIConnectionConfiguration {
  const candidate = raw.api_key ?? raw.ApiKey;
  // Treat any unexpected nonempty value as configured instead of allowing a
  // malformed server response to expose a saved API key in the browser.
  if (typeof candidate === "string" && candidate.trim()) {
    if (candidate === "[NOT CONFIGURED]") return { api_key: "[NOT CONFIGURED]" };
    return { api_key: "configured" };
  }
  if (raw.configured === true) return { api_key: "configured" };
  return { api_key: "[NOT CONFIGURED]" };
}

function normalizeOpenAIConfigurationValidation(raw: RawOpenAIConfigurationValidation): OpenAIConfigurationValidation {
  const errors = Array.isArray(raw.errors)
    ? raw.errors.flatMap((item) => {
        if (!isJsonRecord(item) || typeof item.field !== "string" || typeof item.message !== "string") return [];
        return [{ field: item.field, message: item.message }];
      })
    : [];
  return {
    valid: raw.valid !== false,
    errors,
    configuration: normalizeOpenAIConfiguration(raw.configuration || {}),
  };
}

function normalizeOpenAIConnectionTestResult(raw: RawOpenAIConnectionTestResult): OpenAIConnectionTestResult {
  const connection: JsonRecord = isJsonRecord(raw.connection) ? raw.connection : {};
  return {
    configurationAccepted: raw.configurationAccepted === true,
    configuration: normalizeOpenAIConfiguration(raw.configuration || {}),
    connection: {
      status: typeof connection.status === "string" ? connection.status : "unavailable",
      message: typeof connection.message === "string" ? connection.message : undefined,
    },
  };
}

function normalizeSettings(raw: RawSettings): Settings {
  return {
    default_language: raw.default_language ?? "",
    transcript_retention_days: raw.transcript_retention_days ?? 0,
    delete_audio_after_transcription: raw.delete_audio_after_transcription ?? false,
    max_parallel_transcriptions: raw.max_parallel_transcriptions ?? raw.maximum_simultaneous_transcriptions ?? 0,
    company_vocabulary: raw.company_vocabulary ?? "",
    default_timezone: raw.default_timezone ?? "",
  };
}

function normalizeDashboard(raw: DashboardData): DashboardData {
  const normalizeBreakdown = (rows: DashboardData["results_by_operator"]) =>
    rows?.map((row) => {
      const aliases = row as typeof row & { call_count?: number; match_count?: number };
      return { ...row, count: row.count ?? aliases.call_count ?? aliases.match_count ?? 0 };
    });
  return {
    ...raw,
    results_by_operator: normalizeBreakdown(raw.results_by_operator),
    results_by_keyword_category: normalizeBreakdown(raw.results_by_keyword_category),
  };
}

function normalizeKeyword(keyword: Keyword): Keyword {
  const threshold = keyword.fuzzy_threshold;
  return { ...keyword, fuzzy_threshold: typeof threshold === "number" && threshold > 1 ? threshold / 100 : threshold };
}

function normalizeJob(raw: ProcessingJob): ProcessingJob {
  const aliases = raw as ProcessingJob & { selected_operator_ids?: ProcessingJob["operator_ids"]; queue_name?: string | null; last_error_message?: string | null };
  return {
    ...raw,
    operator_ids: raw.operator_ids || aliases.selected_operator_ids,
    queue: raw.queue ?? aliases.queue_name,
    error_message: raw.error_message ?? aliases.last_error_message,
  };
}

function normalizeCall(raw: CallDetail): CallDetail {
  const record = raw as CallDetail & {
    caller_number?: string | null;
    callee_number?: string | null;
    queue_name?: string | null;
    has_audio?: boolean;
    transcript?: { segments?: unknown[] };
    participants?: unknown[];
  };
  const segmentValues = Array.isArray(raw.transcript_segments) ? raw.transcript_segments : Array.isArray(record.transcript?.segments) ? record.transcript.segments : [];
  const transcript_segments = segmentValues.map((value) => {
    const segment = value as TranscriptAlias;
    return {
      ...segment,
      start_timestamp: numeric(segment.start_timestamp ?? segment.start_seconds),
      end_timestamp: numeric(segment.end_timestamp ?? segment.end_seconds ?? segment.start_timestamp ?? segment.start_seconds),
      original_text: segment.original_text ?? segment.text ?? "",
    };
  });
  const nestedMatches = segmentValues.flatMap((value) => {
    const segment = value as TranscriptAlias;
    return Array.isArray(segment.matches) ? segment.matches.map((match) => normalizeMatch(match, segment)) : [];
  });
  const topLevelMatches = Array.isArray(raw.matches) ? raw.matches.map((match) => normalizeMatch(match)) : [];
  // Segment-nested matches carry the association needed for transcript highlighting.
  // Prefer them when available; the top-level list remains a compatibility fallback.
  const matches = nestedMatches.length ? nestedMatches : topLevelMatches;
  const participants = Array.isArray(record.participants) ? record.participants : [];
  const participantNames = participants.flatMap((participant) => {
    if (!isJsonRecord(participant)) return [];
    const candidate = participant.operator_name ?? participant.display_name ?? (isJsonRecord(participant.operator) ? participant.operator.display_name : undefined);
    return typeof candidate === "string" && candidate.trim() ? [candidate.trim()] : [];
  });
  return {
    ...raw,
    caller: raw.caller ?? record.caller_number,
    callee: raw.callee ?? record.callee_number,
    queue: raw.queue ?? record.queue_name,
    audio_available: raw.audio_available ?? record.has_audio,
    operator_name: raw.operator_name || Array.from(new Set(participantNames)).join(", ") || undefined,
    transcript_segments,
    matches,
    processing_history: Array.isArray(raw.processing_history)
      ? raw.processing_history.map((entry) => {
          const aliases = entry as typeof entry & { updated_at?: string; stage?: string; error?: string | null };
          return {
            ...entry,
            created_at: entry.created_at ?? aliases.updated_at,
            message: entry.message ?? ([aliases.stage, aliases.error].filter(Boolean).join(" — ") || undefined),
          };
        })
      : [],
  };
}

type TranscriptAlias = TranscriptSegment & {
  start_seconds?: number;
  end_seconds?: number;
  text?: string;
  matches?: MatchAlias[];
};

type MatchAlias = KeywordMatch & { start_seconds?: number; end_seconds?: number };

function normalizeMatch(match: MatchAlias, segment?: TranscriptAlias): KeywordMatch {
  return {
    ...match,
    transcript_segment_id: match.transcript_segment_id ?? segment?.id,
    start_timestamp: numeric(match.start_timestamp ?? match.start_seconds ?? segment?.start_timestamp ?? segment?.start_seconds),
    end_timestamp: numericOptional(match.end_timestamp ?? match.end_seconds ?? segment?.end_timestamp ?? segment?.end_seconds),
  };
}

function numeric(value: unknown): number {
  const number = typeof value === "number" ? value : Number(value);
  return Number.isFinite(number) ? number : 0;
}

function numericOptional(value: unknown): number | undefined {
  if (value === undefined || value === null || value === "") return undefined;
  return numeric(value);
}
