// A scripted fetch: serves chunks from a map at `${cdn}/chunks/<sha>`, a manifest at the API path, records every
// request, and lets a test inject failures per URL (network error, HTTP status, corrupt body) for the first N hits.

import type { Manifest } from "../../src/manifest";

export type Fault = { kind: "network" } | { kind: "status"; status: number } | { kind: "corrupt" } | { kind: "short" };

export class FakeFetch {
  readonly requests: { url: string; method: string }[] = [];
  readonly faults = new Map<string, Fault[]>();
  readonly telemetry: unknown[] = [];
  telemetryStatus = 202;
  /** Optional delay per chunk in ms (setTimeout based) to exercise concurrency/ordering. */
  delayMs = 0;
  inFlight = 0;
  maxInFlight = 0;
  constructor(public readonly api: string, public readonly cdn: string, public chunks: Map<string, Uint8Array>, public manifests: Map<string, Manifest> = new Map(), public manifestSha = "") {}

  /** Queue faults for a chunk: consumed in order, one per request. */
  fail(sha: string, ...faults: Fault[]): void {
    const url = `${this.cdn}/chunks/${sha}`;
    this.faults.set(url, [...(this.faults.get(url) ?? []), ...faults]);
  }

  chunkRequests(sha?: string): number {
    return this.requests.filter((r) => r.url.startsWith(`${this.cdn}/chunks/`) && (!sha || r.url.endsWith(sha))).length;
  }

  readonly fetch = async (url: string, init?: RequestInit): Promise<Response> => {
    this.requests.push({ url, method: init?.method ?? "GET" });
    const signal = init?.signal ?? undefined;
    if (signal?.aborted) throw new DOMException("aborted", "AbortError");
    this.inFlight++; this.maxInFlight = Math.max(this.maxInFlight, this.inFlight);
    try {
      if (this.delayMs > 0) {
        await new Promise<void>((resolve, reject) => {
          const t = setTimeout(resolve, this.delayMs);
          signal?.addEventListener("abort", () => { clearTimeout(t); reject(new DOMException("aborted", "AbortError")); }, { once: true });
        });
      }
      const fault = this.faults.get(url)?.shift();
      if (fault?.kind === "network") throw new TypeError("Failed to fetch");
      if (fault?.kind === "status") return new Response("nope", { status: fault.status });
      const m = /^(.*)\/v1\/models\/([^/]+)\/manifest$/.exec(url);
      if (m && m[1] === this.api) {
        const man = this.manifests.get(m[2]!);
        if (!man) return new Response(JSON.stringify({ error: "not_found" }), { status: 404 });
        return new Response(JSON.stringify(man), { status: 200, headers: { "Content-Type": "application/json", "X-Dianome-Manifest-Sha": this.manifestSha, ETag: '"etag"' } });
      }
      if (url === `${this.api}/v1/telemetry/load`) {
        this.telemetry.push(JSON.parse(String(init?.body)));
        return new Response(null, { status: this.telemetryStatus });
      }
      const c = /^(.*)\/chunks\/([0-9a-f]{64})$/.exec(url);
      if (c && c[1] === this.cdn) {
        const body = this.chunks.get(c[2]!);
        if (!body) return new Response("missing", { status: 404 });
        if (fault?.kind === "corrupt") { const b = body.slice(); b[0] = (b[0]! + 1) & 0xff; return new Response(b); }
        if (fault?.kind === "short") return new Response(body.slice(0, body.length - 1));
        return new Response(body.slice(), { headers: { "Content-Type": "application/octet-stream" } });
      }
      return new Response("not found", { status: 404 });
    } finally {
      this.inFlight--;
    }
  };
}
