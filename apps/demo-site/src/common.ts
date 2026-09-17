// Shared by the three demo pages: endpoint overrides, a Dianome instance, byte formatting.
import { Dianome } from "dianome";

const params = new URLSearchParams(location.search);
export const API = params.get("api") ?? "https://api.dianome.dev";
export const CDN = params.get("cdn") ?? "https://cdn.dianome.dev";
export const MODEL = params.get("model") ?? "qwen2.5-0.5b-instruct";

export function makeDianome(opts: { telemetry?: boolean; cache?: "auto" | "per-site" | "none" } = {}): Dianome {
  return new Dianome({ api: API, cdn: CDN, telemetry: opts.telemetry ?? true, cache: opts.cache ?? "auto" });
}

export const fmtBytes = (n: number): string => (n >= 1e9 ? `${(n / 1e9).toFixed(2)} GB` : n >= 1e6 ? `${(n / 1e6).toFixed(1)} MB` : n >= 1e3 ? `${(n / 1e3).toFixed(0)} KB` : `${n} B`);
export const fmtRate = (bps: number): string => `${(bps / 1e6).toFixed(1)} MB/s`;

export function el<T extends HTMLElement = HTMLElement>(id: string): T {
  const e = document.getElementById(id);
  if (!e) throw new Error(`no #${id}`);
  return e as T;
}

/** Counts fetches of a URL prefix made by this document since the call (Resource Timing). */
export function requestCounter(prefix: string): () => number {
  performance.setResourceTimingBufferSize?.(8192);
  const before = performance.getEntriesByType("resource").filter((e) => e.name.startsWith(prefix)).length;
  return () => performance.getEntriesByType("resource").filter((e) => e.name.startsWith(prefix)).length - before;
}
