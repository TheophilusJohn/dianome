export interface Env {
  STORE: R2Bucket;
  TELEMETRY: AnalyticsEngineDataset;
  STATS_CACHE: KVNamespace;
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
