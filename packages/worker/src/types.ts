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

/** Hand-mirrored from schemas/telemetry.v1.json — keep the two in sync. */
export const LOAD_SOURCES = ["network", "per-site-cache", "cross-site-cache", "mixed"] as const;
export const BROWSERS = ["chrome", "firefox", "safari", "other"] as const;
export type LoadSource = (typeof LOAD_SOURCES)[number];
export type Browser = (typeof BROWSERS)[number];

export interface LoadReport {
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

export interface CountryStats { country: string; loads: number; p50_ms: number; p90_ms: number; cache_hit_rate: number }
export interface ModelVariantStats { model: string; variant: string; loads: number; p50_ms: number; bytes: number }
export interface SourceStats { source: string; loads: number; p50_ms: number }

export interface LoadStats {
  since: string;
  window_hours: number;
  by_country: CountryStats[];
  by_model_variant: ModelVariantStats[];
  by_source: SourceStats[];
  degraded?: true;
}
