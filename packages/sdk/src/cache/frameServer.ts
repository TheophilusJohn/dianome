// The frame side of the protocol, with the browser surface injected so every path is unit-testable:
// Chrome (`requestStorageAccess({all: true})` → handle.caches), Firefox (plain `requestStorageAccess()` then
// globals), denied, needs-visit (no first-party interaction yet), and the marker probe that catches partitioned
// grants (Safari-like). packages/cache-frame wires it to the real document.
//
// Structural rule (Phase 0, Firefox): nothing touches `caches`/`indexedDB` before the grant, because objects
// obtained earlier stay partitioned for the life of the document. The store is constructed only inside `grant()`
// from the handle or the globals the grant makes usable.

import { isQuotaError } from "../errors";
import { PerSiteStore } from "./persite";
import { markerUrlFor, MARKER_PATH, PROTOCOL_VERSION, type FrameErrorCode, type FrameRequest, type FrameResponse, type FrameResultOf, type GrantMode, type GrantResult, type HelloResult } from "./protocol";
import { CACHE_NAME, type ChunkStore } from "./types";

/** What `requestStorageAccess({all: true})` resolves to in Chrome (subset we use). */
export interface StorageAccessHandle {
  caches?: CacheStorage;
  indexedDB?: IDBFactory;
  estimate?: () => Promise<StorageEstimate>;
}

export interface FramePlatform {
  origin: string;
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

  async handle(req: FrameRequest): Promise<FrameResponse> {
    const t = this.now();
    try {
      const result = await this.dispatch(req);
      return { v: PROTOCOL_VERSION, id: req.id, ok: true, result, ms: this.now() - t };
    } catch (e) {
      const code: FrameErrorCode = e instanceof FrameError ? e.code : isQuotaError(e) ? "quota" : "internal";
      return { v: PROTOCOL_VERSION, id: req.id, ok: false, error: (e as Error)?.message ?? String(e), code, ms: this.now() - t };
    }
  }

  private need(): ChunkStore {
    if (!this.store) throw new FrameError("no_grant", "storage access not granted in this frame");
    return this.store;
  }

  private async dispatch(req: FrameRequest): Promise<unknown> {
    switch (req.op) {
      case "hello": return this.hello();
      case "grant": return this.grant(req.mode);
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
  async grant(mode: GrantMode): Promise<FrameResultOf<"grant">> {
    if (this.store && this.grantResult) return this.grantResult;
    if (!this.p.requestStorageAccess) return this.remember({ state: "unsupported", reason: "document.requestStorageAccess missing" });
    // The request is the first statement inside the gesture. `{all: true}` yields a handle in Chrome; Firefox
    // ignores the argument and resolves without one, which is also its grant.
    const rsa = this.p.requestStorageAccess;
    const attempt = (): Promise<StorageAccessHandle | undefined | void> => rsa({ all: true });
    let handle: StorageAccessHandle | undefined | void;
    try {
      handle = mode === "await-click" ? await this.p.inGesture(attempt) : await attempt();
    } catch (e) {
      return this.afterRejection(e, mode);
    }
    if (handle && handle.caches) {
      const g = this.p.globals();
      return this.adopt({ caches: handle.caches, indexedDB: handle.indexedDB ?? g.indexedDB, ...(handle.estimate ? { storage: { estimate: handle.estimate } } : g.storage ? { storage: g.storage } : {}) }, "chrome-handle");
    }
    // Firefox path: the grant resolved without a handle; the globals are unpartitioned from here on.
    let has = true;
    try { has = this.p.hasStorageAccess ? await this.p.hasStorageAccess() : true; } catch { /* assume granted */ }
    if (!has) return this.remember({ state: "unsupported", reason: "requestStorageAccess resolved without a handle and hasStorageAccess() is false" });
    return this.adopt(this.p.globals(), "firefox-globals");
  }

  private async afterRejection(e: unknown, mode: GrantMode): Promise<GrantResult> {
    const reason = `${(e as Error)?.name ?? "Error"}: ${(e as Error)?.message ?? String(e)}`;
    let perm: "granted" | "prompt" | "denied" | "unknown" = "unknown";
    try { perm = this.p.permissionState ? await this.p.permissionState() : "unknown"; } catch { perm = "unknown"; }
    if (perm === "denied") return this.remember({ state: "denied", reason });
    // Without a gesture the request is expected to fail unless the permission is persisted: ask for the click.
    if (mode === "silent") return { state: "needs-click", reason };
    // Rejected inside a gesture with the permission still at "prompt": Chrome has no first-party interaction
    // with this origin yet (no top-level visit). The opt-in page provides it.
    return { state: "needs-visit", reason };
  }

  private async adopt(o: { caches: CacheStorage; indexedDB: IDBFactory; storage?: Pick<StorageManager, "estimate"> }, path: "chrome-handle" | "firefox-globals"): Promise<GrantResult> {
    // Probe: the opt-in page wrote a 1 KB marker top-level. If it is not visible through the granted storage,
    // either the storage is still partitioned (Safari-like) or the visit never happened; the visit fixes both
    // cases where they are fixable, so ask for it.
    let marker: Response | undefined;
    try {
      marker = await (await o.caches.open(CACHE_NAME)).match(markerUrlFor(this.p.origin));
    } catch (e) {
      return this.remember({ state: "unsupported", path, reason: `probe failed: ${(e as Error)?.message ?? String(e)}` });
    }
    if (!marker) return { state: "needs-visit", path, reason: `marker ${MARKER_PATH} not visible after grant` };
    this.store = this.makeStore(o);
    return this.remember({ state: "granted", path });
  }

  private remember(r: GrantResult): GrantResult {
    if (r.state === "granted") this.grantResult = r;
    return r;
  }

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
