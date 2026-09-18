export interface Env {
  STORE: R2Bucket;
  TELEMETRY: AnalyticsEngineDataset;
  /** Schema-3 session reports (Phase 5b), a separate dataset so the load columns keep their meaning. */
  SESSIONS: AnalyticsEngineDataset;
  STATS_CACHE: KVNamespace;
  /** HMAC key for split session tokens (secret; the split server holds the same key). */
  SPLIT_SIGNING_KEY?: string;
  /**
   * JSON map model id -> { ws, plan }: where sessions for that model connect and that server's public /plan
   * (Phase 7; replaces SPLIT_WS_URL / SPLIT_PLAN_URL / SPLIT_MODELS). The first entry is the default model.
   * e.g. {"qwen2.5-0.5b-instruct":{"ws":"wss://split.dianome.dev","plan":"https://split.dianome.dev/plan"}}
   */
  SPLIT_SERVERS?: string;
  /** Comma list of Origins allowed to mint session tokens (the demo origin). */
  SPLIT_ALLOWED_ORIGINS?: string;
  /** Account that owns the `dianome_loads` dataset (secret). */
  CF_ACCOUNT_ID?: string;
  /** API token with Account Analytics: Read, for the Analytics Engine SQL API (secret). */
  CF_ANALYTICS_TOKEN?: string;
  /** Git sha baked at build via `wrangler deploy --var GIT_SHA:…`. */
  GIT_SHA?: string;
}

/** Hand-mirrored from schemas/telemetry.v1.json and telemetry.v2.json — keep the three in sync. */
export const LOAD_SOURCES = ["network", "per-site-cache", "cross-site-cache", "mixed"] as const;
export const BROWSERS = ["chrome", "firefox", "safari", "other"] as const;
export const CACHE_MODES = ["per-site", "cross-site", "none"] as const;
export type LoadSource = (typeof LOAD_SOURCES)[number];
export type Browser = (typeof BROWSERS)[number];
export type CacheMode = (typeof CACHE_MODES)[number];

export interface LoadReportV1 {
  schema: 1;
  model: string;
  variant: string;
  bytes: number;
  chunks: number;
  ms: number;
  source: LoadSource;
  cache_hits: number;
  browser: Browser;
  webgpu: boolean;
}

/** Schema 2: the v1 fields plus optional measurements (schemas/telemetry.v2.json). */
export interface LoadReportV2 extends Omit<LoadReportV1, "schema"> {
  schema: 2;
  bytes_per_second?: number;
  verify_ms?: number;
  transfer_ms?: number;
  quota_bytes?: number;
  max_buffer_size?: number;
  cache_mode?: CacheMode;
}

export type LoadReport = LoadReportV1 | LoadReportV2;

/** Schema 3 (Phase 5b): one split/local/server generation session. No prompt content, ever. */
export const SESSION_MODES = ["local", "split", "server"] as const;
export const PLAN_POLICIES = ["cost", "latency", "local", "server"] as const;
export type SessionMode = (typeof SESSION_MODES)[number];
export type PlanPolicy = (typeof PLAN_POLICIES)[number];

export interface SessionReportV3 {
  schema: 3;
  model: string;
  variant: string;
  mode: SessionMode;
  N: number;
  L: number;
  prompt_tokens: number;
  new_tokens: number;
  client_ms: number;
  server_busy_ms: number;
  rtt_ms: number;
  tok_per_s: number;
  plan_policy: PlanPolicy;
  cache_mode: CacheMode;
  browser: Browser;
  webgpu: boolean;
}

/** GET /v1/stats/sessions (Phase 7): aggregates of the schema-3 reports. */
export interface ModelModeStats { model: string; mode: string; sessions: number; p50_tok_per_s: number; p50_n: number; L: number; p50_server_busy_ms: number; p50_rtt_ms: number; new_tokens: number }
export interface PolicyStats { policy: string; mode: string; sessions: number }
export interface BrowserSessionStats { browser: string; sessions: number; p50_tok_per_s: number; webgpu_share: number }
export interface SessionStats {
  since: string;
  window_hours: number;
  computed_at: string;
  by_model_mode: ModelModeStats[];
  by_policy: PolicyStats[];
  by_browser: BrowserSessionStats[];
  degraded?: true;
}

export interface CountryStats { country: string; loads: number; p50_ms: number; p90_ms: number; cache_hit_rate: number }
export interface ModelVariantStats { model: string; variant: string; loads: number; p50_ms: number; bytes: number }
export interface SourceStats { source: string; loads: number; p50_ms: number }

export interface LoadStats {
  since: string;
  window_hours: number;
  /** When the aggregation ran (ISO 8601). A KV hit returns the cached value, so this lags by up to STATS_TTL_SECONDS. */
  computed_at: string;
  by_country: CountryStats[];
  by_model_variant: ModelVariantStats[];
  by_source: SourceStats[];
  degraded?: true;
}
