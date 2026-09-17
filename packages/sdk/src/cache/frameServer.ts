// The frame side of the protocol, with the browser surface injected so every path is unit-testable:
// Chrome (`requestStorageAccess({all: true})` → handle.caches), a grant without a handle (Firefox, Safari: cookie
// access only, `caches` stays partitioned → unsupported), denied, needs-visit (no first-party interaction yet), and
// the marker probe that catches a handle that still sees partitioned storage. packages/cache-frame wires it to the
// real document. See docs/spikes/00-storage-partitioning.md, "Correction (2026-09-17)".
//
// Structural rule: nothing touches `caches`/`indexedDB` before the grant. The store is constructed only inside
// `grant()`, from the handle; the frame's own globals are never used for chunk storage.

import { isQuotaError } from "../errors";
import { PerSiteStore } from "./persite";
import { hopSince, markerUrlFor, MARKER_PATH, PROTOCOL_VERSION, type FrameErrorCode, type FrameRequest, type FrameResponse, type FrameResultOf, type GrantMode, type GrantProgress, type GrantResult, type HelloResult } from "./protocol";
import { CACHE_NAME, type ChunkStore } from "./types";

/** What `requestStorageAccess({all: true})` resolves to in Chrome (subset we use). */
export interface StorageAccessHandle {
  caches?: CacheStorage;
  indexedDB?: IDBFactory;
  estimate?: () => Promise<StorageEstimate>;
}

export interface FramePlatform {
  origin: string;
  /** Diagnostic sink (the frame logs to its console; the parent sees it in DevTools). */
  log?: (message: string) => void;
  /** Present when document.requestStorageAccess exists. `all: true` returns a handle in Chrome, nothing elsewhere. */
  requestStorageAccess?: (opts?: { all: true }) => Promise<StorageAccessHandle | undefined | void>;
  hasStorageAccess?: () => Promise<boolean>;
  /** navigator.permissions.query({ name: "storage-access" }).state */
  permissionState?: () => Promise<"granted" | "prompt" | "denied">;
  /** Globals, read lazily so nothing is touched before the grant. */
  globals: () => { caches: CacheStorage; indexedDB: IDBFactory; storage?: Pick<StorageManager, "estimate"> };
  /** Same-origin fetch of `/chunks/<sha>` (relative to the CDN origin). */
  fetch: (url: string, init?: RequestInit) => Promise<Response>;
  /** Runs `run` synchronously inside the next click on the frame's own button (await-click grants), so that
   * requestStorageAccess is the first statement of the gesture. */
  inGesture: <T>(run: () => Promise<T>) => Promise<T>;
  now?: () => number;
}

export interface FrameServerOptions {
  platform: FramePlatform;
  /** Overrides the store constructed after a grant (tests). */
  makeStore?: (o: { caches: CacheStorage; indexedDB: IDBFactory; storage?: Pick<StorageManager, "estimate"> }) => ChunkStore;
}

class FrameError extends Error {
  constructor(readonly code: FrameErrorCode, message: string) { super(message); }
}

export class FrameServer {
  private store: ChunkStore | null = null;
  private grantResult: GrantResult | null = null;
  private readonly p: FramePlatform;
  private readonly now: () => number;
  private readonly makeStore: NonNullable<FrameServerOptions["makeStore"]>;

  constructor(opts: FrameServerOptions) {
    this.p = opts.platform;
    this.now = opts.platform.now ?? (() => (typeof performance !== "undefined" ? performance.now() : Date.now()));
    this.makeStore = opts.makeStore ?? ((o) => new PerSiteStore({ cdn: this.p.origin, caches: o.caches, indexedDB: o.indexedDB, storage: o.storage }));
  }

  get granted(): boolean { return this.store !== null; }

  /** `receivedAt` (epoch ms) defaults to now; pass the value taken in the message handler for the tightest hop. */
  async handle(req: FrameRequest, receivedAt?: number, progress?: (stage: GrantProgress) => void): Promise<FrameResponse> {
    const t = this.now();
    const hopIn = hopSince(req.sentAt, receivedAt);
    const hop = hopIn === null ? {} : { hopInMs: hopIn };
    try {
      const result = await this.dispatch(req, progress);
      return { v: PROTOCOL_VERSION, id: req.id, ok: true, result, ms: this.now() - t, ...hop };
    } catch (e) {
      const code: FrameErrorCode = e instanceof FrameError ? e.code : isQuotaError(e) ? "quota" : "internal";
      return { v: PROTOCOL_VERSION, id: req.id, ok: false, error: (e as Error)?.message ?? String(e), code, ms: this.now() - t, ...hop };
    }
  }

  private need(): ChunkStore {
    if (!this.store) throw new FrameError("no_grant", "storage access not granted in this frame");
    return this.store;
  }

  private async dispatch(req: FrameRequest, progress?: (stage: GrantProgress) => void): Promise<unknown> {
    switch (req.op) {
      case "hello": return this.hello();
      case "grant": return this.grant(req.mode, progress);
      case "get": return this.need().get(req.sha, req.modelId);
      case "put": await this.need().put(req.sha, req.buf, req.modelId); return null;
      case "has": return this.need().has(req.sha);
      case "status": return this.need().status();
      case "evict": await this.need().evict(req.shas); return null;
      case "evictModel": await this.need().evictModel(req.modelId); return null;
      case "clear": await this.need().clear(); return null;
      case "fetch": return this.fetchChunk(req.sha, req.bytes, req.modelId);
    }
  }

  private async hello(): Promise<HelloResult> {
    let hasStorageAccess: boolean | null = null;
    try { hasStorageAccess = this.p.hasStorageAccess ? await this.p.hasStorageAccess() : null; } catch { hasStorageAccess = null; }
    return { v: PROTOCOL_VERSION, hasStorageAccess, granted: this.granted };
  }

  /**
   * The grant, in the brief's order. `silent` runs it now (works when the permission is already persisted);
   * `await-click` waits for the frame's button and runs it inside that gesture with requestStorageAccess as the
   * first statement of the click continuation (Safari rule, harmless elsewhere).
   */
  async grant(mode: GrantMode, progress?: (stage: GrantProgress) => void): Promise<FrameResultOf<"grant">> {
    if (this.store && this.grantResult) return this.grantResult;
    if (!this.p.requestStorageAccess) return this.remember({ state: "unsupported", reason: "document.requestStorageAccess missing" });
    const rsa = this.p.requestStorageAccess;
    // The request is the first statement inside the gesture. `{all: true}` yields a storage-access handle in
    // Chrome; a browser that resolves without one has granted cookie access only.
    const attempt = (): Promise<StorageAccessHandle | undefined | void> => {
      progress?.("requesting");
      return rsa({ all: true });
    };
    let handle: StorageAccessHandle | undefined | void;
    try {
      if (mode === "await-click") {
        progress?.("waiting-click");
        handle = await this.p.inGesture(() => { progress?.("clicked"); return attempt(); });
      } else {
        handle = await attempt();
      }
      this.log(`requestStorageAccess({all: true}) resolved (${mode}): handle=${handle && handle.caches ? "with caches" : String(handle)}`);
    } catch (e) {
      this.log(`requestStorageAccess rejected (${mode}): ${(e as Error)?.name ?? "Error"}: ${(e as Error)?.message ?? String(e)}`);
      return this.afterRejection(e, mode);
    }
    if (handle && handle.caches) {
      const g = this.p.globals();
      return this.adopt({ caches: handle.caches, indexedDB: handle.indexedDB ?? g.indexedDB, ...(handle.estimate ? { storage: { estimate: handle.estimate } } : g.storage ? { storage: g.storage } : {}) }, "chrome-handle");
    }
    // Resolved without a handle (Firefox, Safari): cookie access only; `caches` stays partitioned (spike 00 correction).
    return this.remember({ state: "unsupported", reason: "requestStorageAccess resolved without a storage-access handle: this browser does not unpartition the Cache API" });
  }

  private async afterRejection(e: unknown, mode: GrantMode): Promise<GrantResult> {
    const reason = `${(e as Error)?.name ?? "Error"}: ${(e as Error)?.message ?? String(e)}`;
    let perm: "granted" | "prompt" | "denied" | "unknown" = "unknown";
    try { perm = this.p.permissionState ? await this.p.permissionState() : "unknown"; } catch { perm = "unknown"; }
    if (perm === "denied") return this.remember({ state: "denied", reason });
    // Without a gesture the request is expected to fail unless the permission is persisted: ask for the click.
    if (mode === "silent") return this.remember({ state: "needs-click", reason });
    // Rejected inside a gesture with the permission still at "prompt": Chrome has no first-party interaction with
    // this origin yet. The top-level opt-in visit provides it.
    return this.remember({ state: "needs-visit", reason: `${reason} (permission: ${perm})` });
  }

  private async adopt(o: { caches: CacheStorage; indexedDB: IDBFactory; storage?: Pick<StorageManager, "estimate"> }, path: "chrome-handle"): Promise<GrantResult> {
    // Probe: the opt-in page wrote a 1 KB marker top-level. If it is not visible through the granted storage,
    // either the storage is still partitioned (Safari-like) or the visit never happened; the visit fixes both
    // cases where they are fixable, so ask for it.
    let marker: Response | undefined;
    try {
      marker = await (await o.caches.open(CACHE_NAME)).match(markerUrlFor(this.p.origin));
    } catch (e) {
      return this.remember({ state: "unsupported", path, reason: `probe failed: ${(e as Error)?.message ?? String(e)}` });
    }
    if (!marker) return this.remember({ state: "needs-visit", path, reason: `marker ${MARKER_PATH} not visible after grant (no top-level visit yet, or storage still partitioned)` });
    this.store = this.makeStore(o);
    return this.remember({ state: "granted", path });
  }

  private remember(r: GrantResult): GrantResult {
    if (r.state === "granted") this.grantResult = r;
    this.log(`grant → ${r.state}${r.path ? ` via ${r.path}` : ""}${r.reason ? `: ${r.reason}` : ""}`);
    return r;
  }

  private log(message: string): void { try { this.p.log?.(`[dianome frame] ${message}`); } catch { /* never let logging break a grant */ } }

  private async fetchChunk(sha: string, bytes: number, modelId: string): Promise<FrameResultOf<"fetch">> {
    const store = this.need();
    const cached = await store.get(sha, modelId);
    if (cached) return { buf: cached, fromCache: true };
    let res: Response;
    try { res = await this.p.fetch(`/chunks/${sha}`); }
    catch (e) { throw new FrameError("network", `network error: ${(e as Error)?.message ?? String(e)}`); }
    if (!res.ok) throw new FrameError(`http_${res.status}`, `HTTP ${res.status}`);
    let buf: ArrayBuffer;
    try { buf = await res.arrayBuffer(); } catch (e) { throw new FrameError("network", `body read failed: ${(e as Error)?.message ?? String(e)}`); }
    if (buf.byteLength !== bytes) throw new FrameError("network", `length ${buf.byteLength}, expected ${bytes}`);
    // The parent verifies the SHA-256 before it trusts the bytes; the frame caches what the CDN served.
    let quota: true | undefined;
    try { await store.put(sha, buf.slice(0), modelId); }
    catch (e) { if (isQuotaError(e)) quota = true; /* other put errors: the chunk is still delivered */ }
    return quota ? { buf, fromCache: false, quota } : { buf, fromCache: false };
  }
}
