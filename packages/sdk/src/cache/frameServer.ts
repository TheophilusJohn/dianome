// The frame side of the protocol, with the browser surface injected so every path is unit-testable. Capability
// detection, never browser names: `requestStorageAccess({all: true})` either returns a storage-access handle (Chrome
// today) or resolves as a plain grant, after which the document's own globals are tried. Either way the marker
// probe decides whether the storage reached is unpartitioned: the opt-in page wrote a marker into the Cache API and
// a "visited" flag into localStorage on the CDN origin, top-level. Marker visible → granted. Flag visible but marker
// not → the grant reaches unpartitioned localStorage but the Cache API stays partitioned (Firefox as of the 2026-09-17
// correction in docs/spikes/00) → unsupported. Neither → needs-visit. A browser that later unpartitions the Cache API
// on a plain grant is picked up by the same probe without an SDK change.
//
// Structural rule: nothing touches `caches`/`indexedDB`/`localStorage` before the grant; the store is constructed
// only inside `grant()`, from the handle or the globals the grant made usable.

import { isQuotaError } from "../errors";
import { PerSiteStore } from "./persite";
import { hopSince, markerUrlFor, MARKER_PATH, PROTOCOL_VERSION, type FrameErrorCode, type FrameRequest, type FrameResponse, type FrameResultOf, type GrantMode, type GrantPath, type GrantProgress, type GrantResult, type HelloResult } from "./protocol";
import { CACHE_NAME, type ChunkStore } from "./types";

/** What `requestStorageAccess({all: true})` resolves to in Chrome (subset we use). */
export interface StorageAccessHandle {
  caches?: CacheStorage;
  indexedDB?: IDBFactory;
  estimate?: () => Promise<StorageEstimate>;
}

export interface FramePlatform {
  origin: string;
  /** The opt-in page's "visited" flag in the CDN origin's localStorage; read only after a grant. */
  visitedFlag?: () => string | null;
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
    // The request is the first statement inside the gesture. `{all: true}` yields a storage-access handle where the
    // browser supports one; elsewhere the argument is ignored and this is the plain grant.
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
    // Plain grant: no handle; the globals are whatever the grant made of them. The marker probe decides.
    let has = true;
    try { has = this.p.hasStorageAccess ? await this.p.hasStorageAccess() : true; } catch { /* assume granted */ }
    if (!has) return this.remember({ state: "unsupported", reason: "requestStorageAccess resolved without a handle and hasStorageAccess() is false" });
    return this.adopt(this.p.globals(), "plain-globals");
  }

  private async afterRejection(e: unknown, mode: GrantMode): Promise<GrantResult> {
    const reason = `${(e as Error)?.name ?? "Error"}: ${(e as Error)?.message ?? String(e)}`;
    let perm: "granted" | "prompt" | "denied" | "unknown" = "unknown";
    try { perm = this.p.permissionState ? await this.p.permissionState() : "unknown"; } catch { perm = "unknown"; }
    if (perm === "denied") return this.remember({ state: "denied", reason });
    // Without a gesture the request is expected to fail unless the permission is persisted: ask for the click.
    if (mode === "silent") return this.remember({ state: "needs-click", reason });
    // Rejected inside a gesture with the permission still at "prompt" (or no permissions API): the browser has no
    // first-party interaction with this origin yet. The top-level opt-in visit provides it.
    return this.remember({ state: "needs-visit", reason: `${reason} (permission: ${perm})` });
  }

  private async adopt(o: { caches: CacheStorage; indexedDB: IDBFactory; storage?: Pick<StorageManager, "estimate"> }, path: GrantPath): Promise<GrantResult> {
    // Probe: the opt-in page wrote a 1 KB marker into the Cache API top-level. Visible through the granted storage
    // → unpartitioned → granted.
    let marker: Response | undefined;
    try {
      marker = await (await o.caches.open(CACHE_NAME)).match(markerUrlFor(this.p.origin));
    } catch (e) {
      return this.remember({ state: "unsupported", path, reason: `probe failed: ${(e as Error)?.message ?? String(e)}` });
    }
    if (marker) {
      this.store = this.makeStore(o);
      return this.remember({ state: "granted", path });
    }
    // No marker. The opt-in page also set a "visited" flag in localStorage on this origin. If the grant lets this
    // document read that flag, the visit happened and the Cache API is what stays partitioned: unsupported, and the
    // user is not sent to the opt-in page again. Otherwise the visit is what is missing.
    let visited: string | null = null;
    try { visited = this.p.visitedFlag ? this.p.visitedFlag() : null; } catch { visited = null; }
    if (visited) return this.remember({ state: "unsupported", path, reason: `visited flag (${visited}) visible but marker ${MARKER_PATH} not: the Cache API stays partitioned after the grant` });
    return this.remember({ state: "needs-visit", path, reason: `marker ${MARKER_PATH} not visible after grant (no top-level visit yet, or storage still partitioned)` });
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
