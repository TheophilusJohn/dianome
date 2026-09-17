// Public types of the `dianome` entry point.

import type { LoadedEntry } from "./assemble";
import type { CacheMode, ChunkSource } from "./cache/types";
import type { Manifest, VariantName } from "./manifest";
import type { LoadReport, LoadSource } from "./telemetry";

export interface DianomeOptions {
  /** Default https://api.dianome.dev */
  api?: string;
  /** Default https://cdn.dianome.dev */
  cdn?: string;
  /** Default true; false sends nothing. */
  telemetry?: boolean;
  /** Default "auto": cross-site if enabled and supported, else per-site. */
  cache?: "auto" | "per-site" | "none";
  /** Parallel chunk fetches (default 6). */
  concurrency?: number;
  /** Retries per chunk on network errors and 5xx (default 3). */
  retries?: number;
  /** Overrides `${cdn}/frame/v1/index.html` for the cross-site frame. */
  frameUrl?: string;
  /** Injected fetch (tests). */
  fetch?: (input: string, init?: RequestInit) => Promise<Response>;
}

export interface StreamOptions {
  /** Model manifests: which variant (default: the smallest present, q4 > q8 > fp16). Ignored for files manifests. */
  variant?: VariantName;
  /** Files manifests: load only these paths, in this order (default: every file in manifest order). */
  files?: string[];
  signal?: AbortSignal;
  onProgress?: (p: Progress) => void;
}

export interface Progress {
  bytesDone: number;
  bytesTotal: number;
  chunksDone: number;
  chunksTotal: number;
  /** Group of the chunk that just completed. */
  group: string;
  /** Where that chunk came from. */
  source: ChunkSource;
  elapsedMs: number;
  bytesPerSecond: number;
}

export interface LoadedGroup {
  name: string;
  index: number;
  bytes: number;
  tied: boolean;
  entries: Map<string, LoadedEntry>;
}

export interface LoadSummary {
  model: string;
  variant: string;
  bytes: number;
  chunks: number;
  ms: number;
  bytesPerSecond: number;
  source: LoadSource;
  cacheHits: number;
  cacheMode: CacheMode;
  /** Per-chunk sources in completion order. */
  sources: ChunkSource[];
  verifyMs: number;
  transferMs: number;
  /** True when a QuotaExceededError switched this session to `cache: "none"` during the load. */
  cacheDisabled: boolean;
  /** The telemetry report that was sent (null when telemetry is off). */
  report: LoadReport | null;
  /** HTTP status of the telemetry POST, null when it failed or was not sent. */
  telemetryStatus: number | null;
}

export interface LoadedModel {
  manifest: Manifest;
  variant: string;
  groups: Map<string, LoadedGroup>;
  group(name: string): LoadedGroup;
  entry(name: string): LoadedEntry;
  summary: LoadSummary;
}

export type CrossSiteState = "granted" | "unsupported" | "denied" | "needs-visit" | "needs-click";

export interface CrossSiteResult {
  state: CrossSiteState;
  /** `needs-visit`: the top-level opt-in page on the CDN origin the developer should link to. */
  visitUrl?: string;
  /**
   * `needs-click`: the browser wants a click inside the CDN frame itself. `mount(el)` shows the frame's single
   * "Enable shared model cache" button inside `el` and resolves once the user has clicked it.
   */
  mount?: (container: HTMLElement) => Promise<CrossSiteResult>;
  /** Why the grant failed, when it did (for logs). */
  reason?: string;
}

export interface CacheStatus {
  mode: CacheMode;
  quotaBytes: number | null;
  usageBytes: number;
  chunks: number;
}
