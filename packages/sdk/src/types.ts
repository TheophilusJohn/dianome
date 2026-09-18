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
  /**
   * Phase 6: an API key (`dk_live_…` from POST /v1/keys or the dashboard). Sent as `Authorization: Bearer` on the
   * session mint and on telemetry posts; the session report then carries the key's id (never the key) and the
   * dashboard meters the session. Without it everything behaves as before (the demo origins can still mint sessions).
   */
  apiKey?: string;
  /**
   * Phase 6 self-host: connect run() straight to your own split server per model instead of minting a session
   * through the API, so hidden states never reach Dianome. `token` is that server's static SPLIT_TOKEN (anyone who
   * can load your page can read it; for a public site point the Worker's SPLIT_SERVERS at your host instead and share
   * SPLIT_SIGNING_KEY). `plan` defaults to the ws URL as http(s) plus /plan.
   */
  split?: { servers: Record<string, SplitServerOverride> };
}

export interface SplitServerOverride { ws: string; token?: string; plan?: string }

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
  /** Median postMessage hop per chunk delivered by the cross-site frame (ms); 0 when no chunk came through the frame. */
  transferMs: number;
  /** Every per-chunk hop behind `transferMs`, in completion order. */
  transferSamples: number[];
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

/** Stages `mount()` reports while the user's click and the browser's decision are pending. */
export type CrossSiteProgress = "waiting-click" | "clicked" | "requesting";

export interface CrossSiteResult {
  state: CrossSiteState;
  /** `needs-visit`: the top-level opt-in page on the CDN origin the developer should link to. */
  visitUrl?: string;
  /**
   * `needs-click`: the browser wants a click inside the CDN frame itself. `mount(el)` shows the frame's single
   * "Enable shared model cache" button inside `el` and resolves once the user has clicked it and the browser has
   * decided; `onProgress` reports the stages in between (a permission prompt can keep it pending).
   */
  mount?: (container: HTMLElement, opts?: { onProgress?: (stage: CrossSiteProgress) => void }) => Promise<CrossSiteResult>;
  /** How the frame reached storage: a storage-access handle, or the plain grant with the frame's globals (detected, not assumed from the browser). */
  path?: "chrome-handle" | "plain-globals";
  /** Why the grant ended the way it did (rejection name and message, marker probe, persisted outcome). */
  reason?: string;
  /** `granted`: per-site chunks copied into the shared cache (skipped = the frame already had them). */
  adopted?: { chunks: number; bytes: number; skipped: number; stopped?: string };
}

export interface CacheStatus {
  mode: CacheMode;
  quotaBytes: number | null;
  usageBytes: number;
  chunks: number;
}
