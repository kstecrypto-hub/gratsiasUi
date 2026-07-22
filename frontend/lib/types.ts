export type Identifier = string | number;

export type ConnectionState = {
  status: string;
  message?: string | null;
};

export type YeastarConnectionState =
  | "not_configured"
  | "not_tested"
  | "connected"
  | "auth_rejected"
  | "token_refresh_failed"
  | "ip_blocked"
  | "ip_not_allowed"
  | "api_disabled"
  | "permission_denied"
  | "unsupported_api_version"
  | "unsupported_firmware"
  | "network_unavailable"
  | "temporarily_unavailable";

export type YeastarCapabilities = {
  extensions: boolean | null;
  cdr_v2: boolean | null;
  recordings: boolean | null;
};

export type YeastarConnectionStatus = {
  status: YeastarConnectionState;
  configured: boolean;
  last_tested_at: string | null;
  last_successful_connection_at: string | null;
  model_name: string | null;
  firmware_version: string | null;
  capabilities: YeastarCapabilities;
  message: string;
  last_error_reference?: string | null;
};

export type YeastarSafeCredentialMarker =
  | "configured"
  | "[CONFIGURED]"
  | "[NOT CONFIGURED]"
  | "[REDACTED]";

export type YeastarConnectionSettingsView = {
  BaseUrl: string;
  ClientId: YeastarSafeCredentialMarker;
  ClientSecret: YeastarSafeCredentialMarker;
  DateFormat: string;
  PageSize: number;
  IgnoreSslErrors: boolean;
};

export type YeastarConnectionConfiguration = {
  Name: string;
  Settings: YeastarConnectionSettingsView;
};

export type YeastarConnectionSettingsInput = {
  BaseUrl: string;
  ClientId: string;
  ClientSecret: string;
  DateFormat: string;
  PageSize: number;
  IgnoreSslErrors: boolean;
};

export type YeastarConnectionConfigurationInput = {
  Name: string;
  Settings: YeastarConnectionSettingsInput;
};

export type YeastarConfigurationValidationError = {
  field: string;
  message: string;
};

export type YeastarConfigurationValidation = {
  valid: boolean;
  errors: YeastarConfigurationValidationError[];
  configuration: YeastarConnectionConfiguration;
};

export type YeastarConnectionTestResult = {
  configurationAccepted: boolean;
  configuration: YeastarConnectionConfiguration;
  connection: {
    status: YeastarConnectionState;
    model: string | null;
    firmwareVersion: string | null;
  };
};

export type OpenAISafeCredentialMarker =
  | "configured"
  | "[CONFIGURED]"
  | "[NOT CONFIGURED]"
  | "[REDACTED]";

/** A deliberately sanitized view; it must never contain the actual API key. */
export type OpenAIConnectionConfiguration = {
  api_key: OpenAISafeCredentialMarker;
};

/** An empty key retains an existing configured value. */
export type OpenAIConnectionConfigurationInput = {
  api_key: string;
};

export type OpenAIConfigurationValidationError = {
  field: string;
  message: string;
};

export type OpenAIConfigurationValidation = {
  valid: boolean;
  errors: OpenAIConfigurationValidationError[];
  configuration: OpenAIConnectionConfiguration;
};

export type OpenAIConnectionTestResult = {
  configurationAccepted: boolean;
  configuration: OpenAIConnectionConfiguration;
  connection: ConnectionState;
};

export type Configuration = {
  yeastar: ConnectionState;
  openai: ConnectionState;
  database: ConnectionState;
  processing: ConnectionState;
};

export type Administrator = {
  id?: Identifier;
  email: string;
};

export type Operator = {
  id: Identifier;
  yeastar_extension_id?: string | null;
  extension_number: string;
  display_name: string;
  email?: string | null;
  enabled: boolean;
  last_synced_at?: string | null;
};

export type KeywordCategory = {
  id: Identifier;
  name: string;
  description?: string | null;
  active?: boolean;
  keyword_count?: number;
};

export type KeywordVariant = {
  id?: Identifier;
  phrase: string;
};

export type Keyword = {
  id: Identifier;
  category_id: Identifier;
  category_name?: string;
  canonical_phrase: string;
  variants?: Array<KeywordVariant | string>;
  accent_insensitive: boolean;
  whole_word: boolean;
  exact_phrase: boolean;
  fuzzy_match: boolean;
  fuzzy_threshold?: number | null;
  active: boolean;
  severity: string;
  notes?: string | null;
};

export type Settings = {
  default_language: string;
  transcript_retention_days: number;
  delete_audio_after_transcription: boolean;
  max_parallel_transcriptions: number;
  company_vocabulary: string;
  default_timezone: string;
};

export type DashboardBreakdown = {
  id?: Identifier;
  name?: string;
  operator_name?: string;
  category_name?: string;
  count: number;
};

export type DashboardData = {
  calls_analyzed?: number;
  calls_with_recordings?: number;
  calls_transcribed?: number;
  calls_with_matches?: number;
  failed_calls?: number;
  processing_jobs?: number;
  results_by_operator?: DashboardBreakdown[];
  results_by_keyword_category?: DashboardBreakdown[];
  recent_jobs?: ProcessingJob[];
};

export type ProcessingStatus =
  | "queued"
  | "connecting"
  | "fetching_calls"
  | "fetching_call_details"
  | "finding_recordings"
  | "downloading_recordings"
  | "inspecting_audio"
  | "extracting_operator_audio"
  | "transcribing"
  | "searching_keywords"
  | "completed"
  | "completed_with_errors"
  | "failed"
  | "cancelled"
  | string;

export type ProcessingJob = {
  id: Identifier;
  is_current?: boolean;
  status: ProcessingStatus;
  date_from: string;
  date_to: string;
  operator_ids?: Identifier[];
  operators?: Array<Pick<Operator, "id" | "display_name" | "extension_number">>;
  direction?: string | null;
  queue?: string | null;
  progress_percent?: number | null;
  current_stage?: string;
  calls_found?: number;
  recordings_found?: number;
  calls_completed?: number;
  calls_failed?: number;
  total_items?: number;
  transcribed_count?: number;
  searched_count?: number;
  created_at?: string;
  started_at?: string | null;
  completed_at?: string | null;
  error_message?: string | null;
};

export type CreateJobInput = {
  date_from: string;
  date_to: string;
  operator_ids: Identifier[];
  direction?: string | null;
  queue?: string | null;
  keyword_category_ids: Identifier[];
  call_status?: string | null;
  recording_availability?: string | null;
  include_all_speakers?: boolean;
};

export type ResultRow = {
  id?: Identifier;
  job_id?: Identifier;
  call_id: Identifier;
  operator_id?: Identifier | null;
  occurred_at?: string;
  started_at?: string;
  date?: string;
  time?: string;
  operator_name?: string;
  masked_phone_number?: string;
  duration_seconds?: number;
  keywords?: string[];
  keywords_found?: string[];
  match_count?: number;
  processing_status?: string;
  status?: string;
  direction?: string;
};

export type Paginated<T> = {
  items: T[];
  total: number;
  page: number;
  page_size: number;
  pages?: number;
};

export type KeywordMatch = {
  id: Identifier;
  keyword_id?: Identifier;
  keyword?: string;
  keyword_phrase?: string;
  category?: string;
  category_name?: string;
  transcript_segment_id?: Identifier;
  original_matched_text?: string;
  normalized_match?: string;
  context_before?: string;
  context_after?: string;
  start_timestamp: number;
  end_timestamp?: number;
  match_method?: string;
  match_score?: number | null;
};

export type TranscriptSegment = {
  id: Identifier;
  operator_id?: Identifier | null;
  speaker_label: string;
  speaker_source?: string;
  start_timestamp: number;
  end_timestamp: number;
  original_text: string;
  confidence?: number | null;
  sequence_number?: number;
};

export type ProcessingHistoryEntry = {
  id?: Identifier;
  status: string;
  occurred_at?: string;
  created_at?: string;
  message?: string | null;
};

export type CallDetail = {
  id: Identifier;
  occurred_at?: string;
  started_at?: string;
  operator?: Pick<Operator, "id" | "display_name" | "extension_number">;
  operator_name?: string;
  caller?: string | null;
  callee?: string | null;
  duration_seconds?: number;
  direction?: string;
  queue?: string | null;
  status?: string;
  processing_status?: string;
  recording_available?: boolean;
  audio_available?: boolean;
  matches?: KeywordMatch[];
  transcript_segments?: TranscriptSegment[];
  processing_history?: ProcessingHistoryEntry[];
};

export type ResultFilters = {
  job_id?: string;
  date_from?: string;
  date_to?: string;
  operator_id?: string;
  transcript_query?: string;
  keyword?: string;
  category_id?: string;
  direction?: string;
  has_matches?: string;
  page?: number;
  page_size?: number;
  sort?: string;
  order?: "asc" | "desc";
};

export type JobListFilters = {
  page?: number;
  page_size?: number;
};
