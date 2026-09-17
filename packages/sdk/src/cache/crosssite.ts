// Cross-site store: the SDK side of the frame protocol. A hidden iframe on the CDN origin owns the shared cache;
// the parent talks to it over postMessage with explicit targetOrigin, validates event.origin and event.source,
// and transfers ArrayBuffers. The frame fetches chunks itself (same-origin), so the parent never fetches on this
// path. See docs/briefs/03-sdk-brief.md and docs/spikes/00-storage-partitioning.md for the browser rules.

import { detectBrowser, type Browser } from "../device";
import { AbortedError, ChunkError, DianomeError } from "../errors";
import type { CrossSiteResult } from "../types";
import { epochNow, frameUrlFor, hopSince, optinUrlFor, parseProgress, parseResponse, sameOrigin, stampSent, transferablesOf, PROTOCOL_VERSION, type FrameErrorCode, type FrameOp, type FrameRequest, type FrameResultOf, type GrantProgress, type GrantResult } from "./protocol";
import type { ChunkStore, ChunkStoreStatus } from "./types";

export const CROSSSITE_KEY = "dianome:crosssite";
export type PersistedCrossSite = "granted" | "denied" | "unsupported";

export function readPersisted(storage: Storage | undefined = typeof localStorage !== "undefined" ? localStorage : undefined): PersistedCrossSite | null {
  try {
    const v = storage?.getItem(CROSSSITE_KEY);
    return v === "granted" || v === "denied" || v === "unsupported" ? v : null;
  } catch { return null; }
}

export function writePersisted(v: PersistedCrossSite | null, storage: Storage | undefined = typeof localStorage !== "undefined" ? localStorage : undefined): void {
  try {
    if (v === null) storage?.removeItem(CROSSSITE_KEY);
    else storage?.setItem(CROSSSITE_KEY, v);
  } catch { /* storage disabled */ }
}

// ---- transport -------------------------------------------------------------------------------------------------

/** One end of the parent ↔ frame channel. The port enforces origin and source; the client sees only valid data. */
export interface FramePort {
  /** The element behind a DOM port (so a mounted button frame can be collapsed after the grant). */
  iframe?: unknown;
  post(msg: FrameRequest, transfer: ArrayBuffer[]): void;
  listen(handler: (data: unknown) => void): () => void;
  close(): void;
}

type WindowLike = Pick<Window, "addEventListener" | "removeEventListener">;
type IframeLike = { contentWindow: { postMessage: (msg: unknown, targetOrigin: string, transfer?: Transferable[]) => void } | null; remove?: () => void };

/** A port over a real iframe: accepts only messages whose origin is the frame's and whose source is its window. */
export function iframePort(iframe: IframeLike, frameOrigin: string, win: WindowLike): FramePort {
  const handlers = new Set<(data: unknown) => void>();
  const onMessage = (ev: MessageEvent) => {
    if (!sameOrigin(ev.origin, frameOrigin)) return;
    if (!iframe.contentWindow || ev.source !== (iframe.contentWindow as unknown)) return;
    for (const h of handlers) h(ev.data);
  };
  win.addEventListener("message", onMessage as EventListener);
  return {
    iframe,
    post(msg, transfer) {
      const w = iframe.contentWindow;
      if (!w) throw new DianomeError("frame_error", "frame window is gone");
      w.postMessage(msg, frameOrigin, transfer);
    },
    listen(handler) { handlers.add(handler); return () => handlers.delete(handler); },
    close() { win.removeEventListener("message", onMessage as EventListener); handlers.clear(); iframe.remove?.(); },
  };
}

export class FrameCallError extends DianomeError {
  constructor(readonly frameCode: FrameErrorCode, message: string) { super("frame_error", message); this.name = "FrameCallError"; }
}

export interface OpTiming {
  count: number;
  /** Parent-measured round trip (sum, max). */
  totalMs: number;
  maxMs: number;
  /** Time the frame itself reported spending inside the op (sum). */
  frameMs: number;
  /** postMessage hops alone (sums): frame → parent for replies, parent → frame for requests. Null-hop replies are not counted. */
  hopOutMs: number;
  hopInMs: number;
}

export interface CallOptions { timeoutMs?: number | null; signal?: AbortSignal; /** Progress notes the frame posts for a pending op (await-click grants). */ onProgress?: ((stage: GrantProgress) => void) | undefined }

/** What a call resolves to when the caller wants the timing breakdown as well as the result. */
export interface CallMeta<T> {
  result: T;
  roundTripMs: number;
  /** Frame → parent postMessage hop for this reply (the transferred buffer rides on it); null if the frame did not stamp it. */
  hopOutMs: number | null;
  /** Parent → frame hop of the request, as measured by the frame; null if not reported. */
  hopInMs: number | null;
  frameMs: number;
}

export const DEFAULT_CALL_TIMEOUT_MS = 20_000;
export const FETCH_CALL_TIMEOUT_MS = 180_000;
export const HELLO_TIMEOUT_MS = 8_000;

const now = (): number => (typeof performance !== "undefined" ? performance.now() : Date.now());

/** Request/response correlation over a FramePort with per-op timeouts and timing stats. */
export class FrameClient {
  private nextId = 1;
  private readonly pending = new Map<number, { resolve: (v: CallMeta<unknown>) => void; reject: (e: unknown) => void; timer: ReturnType<typeof setTimeout> | null; op: FrameOp; t0: number; onProgress?: ((stage: GrantProgress) => void) | undefined }>();
  private readonly stats = new Map<string, OpTiming>();
  private readonly unlisten: () => void;
  private closed = false;

  constructor(readonly port: FramePort, private readonly timeoutMs = DEFAULT_CALL_TIMEOUT_MS) {
    this.unlisten = port.listen((data) => this.onMessage(data));
  }

  private onMessage(data: unknown): void {
    const receivedAt = epochNow(); // first thing: the hop ends when the message reaches us, not when we finish bookkeeping
    const prog = parseProgress(data);
    if (prog) { this.pending.get(prog.id)?.onProgress?.(prog.progress); return; }
    const res = parseResponse(data);
    if (!res) return;
    const p = this.pending.get(res.id);
    if (!p) return;
    this.pending.delete(res.id);
    if (p.timer) clearTimeout(p.timer);
    const ms = now() - p.t0;
    const hopOutMs = hopSince(res.sentAt, receivedAt);
    const hopInMs = typeof res.hopInMs === "number" ? res.hopInMs : null;
    const s = this.stats.get(p.op) ?? { count: 0, totalMs: 0, maxMs: 0, frameMs: 0, hopOutMs: 0, hopInMs: 0 };
    s.count++; s.totalMs += ms; s.maxMs = Math.max(s.maxMs, ms); s.frameMs += res.ms;
    if (hopOutMs !== null) s.hopOutMs += hopOutMs;
    if (hopInMs !== null) s.hopInMs += hopInMs;
    this.stats.set(p.op, s);
    if (res.ok) p.resolve({ result: res.result, roundTripMs: ms, hopOutMs, hopInMs, frameMs: res.ms });
    else p.reject(new FrameCallError(res.code, `frame ${p.op}: ${res.error}`));
  }

  async call<O extends FrameOp>(req: Omit<Extract<FrameRequest, { op: O }>, "v" | "id">, opts: CallOptions = {}): Promise<FrameResultOf<O>> {
    return (await this.callWithMeta<O>(req, opts)).result;
  }

  callWithMeta<O extends FrameOp>(req: Omit<Extract<FrameRequest, { op: O }>, "v" | "id">, opts: CallOptions = {}): Promise<CallMeta<FrameResultOf<O>>> {
    if (this.closed) return Promise.reject(new DianomeError("frame_error", "frame client closed"));
    const id = this.nextId++;
    const msg = { v: PROTOCOL_VERSION, id, ...req } as FrameRequest;
    const timeoutMs = opts.timeoutMs === undefined ? this.timeoutMs : opts.timeoutMs;
    return new Promise<CallMeta<FrameResultOf<O>>>((resolve, reject) => {
      const entry = { resolve: resolve as (v: CallMeta<unknown>) => void, reject, timer: null as ReturnType<typeof setTimeout> | null, op: req.op, t0: now(), onProgress: opts.onProgress };
      if (timeoutMs !== null) {
        entry.timer = setTimeout(() => {
          this.pending.delete(id);
          reject(new DianomeError("frame_timeout", `frame ${req.op}: no reply after ${timeoutMs} ms`));
        }, timeoutMs);
      }
      if (opts.signal) {
        const onAbort = () => { if (this.pending.delete(id)) { if (entry.timer) clearTimeout(entry.timer); reject(new AbortedError()); } };
        if (opts.signal.aborted) return onAbort();
        opts.signal.addEventListener("abort", onAbort, { once: true });
      }
      this.pending.set(id, entry);
      try { this.port.post(stampSent(msg), transferablesOf(msg)); }
      catch (e) { this.pending.delete(id); if (entry.timer) clearTimeout(entry.timer); reject(e); }
    });
  }

  /** Per-op round-trip timings as measured by the parent, plus the frame's own reported time. */
  timings(): Record<string, OpTiming> {
    return Object.fromEntries([...this.stats].map(([k, v]) => [k, { ...v }]));
  }

  close(): void {
    if (this.closed) return;
    this.closed = true;
    this.unlisten();
    for (const [, p] of this.pending) { if (p.timer) clearTimeout(p.timer); p.reject(new DianomeError("frame_error", "frame client closed")); }
    this.pending.clear();
    this.port.close();
  }
}

// ---- store -----------------------------------------------------------------------------------------------------

export class CrossSiteStore implements ChunkStore {
  readonly mode = "cross-site" as const;
  constructor(readonly client: FrameClient) {}
  get(sha: string, modelId?: string): Promise<ArrayBuffer | null> { return this.client.call<"get">({ op: "get", sha, ...(modelId !== undefined ? { modelId } : {}) }); }
  async put(sha: string, buf: ArrayBuffer, modelId?: string): Promise<void> {
    try { await this.client.call<"put">({ op: "put", sha, buf, ...(modelId !== undefined ? { modelId } : {}) }); }
    catch (e) { if (e instanceof FrameCallError && e.frameCode === "quota") { const q = new DOMException(e.message, "QuotaExceededError"); throw q; } throw e; }
  }
  has(sha: string): Promise<boolean> { return this.client.call<"has">({ op: "has", sha }); }
  status(): Promise<ChunkStoreStatus> { return this.client.call<"status">({ op: "status" }); }
  async evict(shas: string[]): Promise<void> { await this.client.call<"evict">({ op: "evict", shas }); }
  async evictModel(modelId: string): Promise<void> { await this.client.call<"evictModel">({ op: "evictModel", modelId }); }
  async clear(): Promise<void> { await this.client.call<"clear">({ op: "clear" }); }
  /** `transferMs` is the frame → parent postMessage hop that carried the buffer (null when the frame did not stamp it). */
  async fetch(sha: string, bytes: number, modelId: string, signal?: AbortSignal): Promise<{ buf: ArrayBuffer; fromCache: boolean; transferMs: number | null; quota?: true }> {
    try {
      const m = await this.client.callWithMeta<"fetch">({ op: "fetch", sha, bytes, modelId }, { timeoutMs: FETCH_CALL_TIMEOUT_MS, ...(signal ? { signal } : {}) });
      const r = m.result;
      return { buf: r.buf, fromCache: r.fromCache, transferMs: m.hopOutMs, ...(r.quota ? { quota: true as const } : {}) };
    } catch (e) {
      if (e instanceof FrameCallError) {
        const m = /^http_(\d+)$/.exec(e.frameCode);
        if (m) throw new ChunkError("chunk_http", sha, e.message, Number(m[1]), { cause: e });
        if (e.frameCode === "network") throw new ChunkError("chunk_network", sha, e.message, undefined, { cause: e });
      }
      throw e;
    }
  }
  timings(): Record<string, OpTiming> { return this.client.timings(); }
  close(): void { this.client.close(); }
}

// ---- connection ------------------------------------------------------------------------------------------------

export interface FrameOpener {
  /** Opens the frame document and completes the `hello` handshake. `visibleIn` renders it as the opt-in button. */
  (url: string, visibleIn: HTMLElement | null): Promise<FrameClient>;
}

export interface CrossSiteConnectOptions {
  cdn: string;
  /** Overrides `${cdn}/frame/v1/index.html`. */
  frameUrl?: string | undefined;
  /** True for enableCrossSiteCache(); false for the silent `cache: "auto"` attempt (never prompts, never re-tries a persisted denied/unsupported). */
  explicit: boolean;
  timeoutMs?: number | undefined;
  /** Called when a grant completes after `mount()`, so the owner can adopt the store. */
  onGranted?: ((store: CrossSiteStore) => void) | undefined;
  // Injection points (tests).
  browser?: Browser | undefined;
  storage?: Storage | undefined;
  openFrame?: FrameOpener | undefined;
  returnUrl?: string | undefined;
  hasDocument?: boolean | undefined;
  hasStorageAccessApi?: boolean | undefined;
}

export interface CrossSiteConnection { store: CrossSiteStore | null; result: CrossSiteResult }

const hiddenStyle = "position:absolute;width:0;height:0;border:0;overflow:hidden;visibility:hidden;left:-9999px;top:0";
const visibleStyle = "display:block;width:100%;max-width:360px;height:48px;border:0";

/** Real opener: an iframe with allow="storage-access", awaited `load`, then `hello`. */
export const domFrameOpener: FrameOpener = async (url, visibleIn) => {
  const doc = document;
  const iframe = doc.createElement("iframe");
  iframe.setAttribute("allow", "storage-access");
  iframe.referrerPolicy = "strict-origin-when-cross-origin";
  iframe.title = "Dianome shared model cache";
  if (visibleIn) iframe.setAttribute("style", visibleStyle);
  else { iframe.setAttribute("style", hiddenStyle); iframe.setAttribute("aria-hidden", "true"); iframe.tabIndex = -1; }
  const loaded = new Promise<void>((resolve, reject) => {
    iframe.addEventListener("load", () => resolve(), { once: true });
    iframe.addEventListener("error", () => reject(new DianomeError("frame_error", "frame failed to load")), { once: true });
  });
  iframe.src = url;
  (visibleIn ?? doc.body ?? doc.documentElement).appendChild(iframe);
  await loaded;
  const client = new FrameClient(iframePort(iframe, new URL(url).origin, window));
  // A blocked or missing frame still fires `load` (error page), so the handshake is what detects it.
  await client.call<"hello">({ op: "hello" }, { timeoutMs: HELLO_TIMEOUT_MS });
  return client;
};

/** One hidden frame per document and URL, shared by every Dianome instance. */
const hiddenClients = new Map<string, Promise<FrameClient>>();

async function hiddenClient(url: string, open: FrameOpener): Promise<FrameClient> {
  let p = hiddenClients.get(url);
  if (!p) {
    p = open(url, null);
    hiddenClients.set(url, p);
    p.catch(() => hiddenClients.delete(url));
  }
  return p;
}

function dropHidden(url: string, client: FrameClient): void {
  hiddenClients.delete(url);
  client.close();
}

export async function connectCrossSite(opts: CrossSiteConnectOptions): Promise<CrossSiteConnection> {
  const storage = opts.storage ?? (typeof localStorage !== "undefined" ? localStorage : undefined);
  const hasDoc = opts.hasDocument ?? typeof document !== "undefined";
  if (!hasDoc) return { store: null, result: { state: "unsupported", reason: "no document" } };
  const hasApi = opts.hasStorageAccessApi ?? (typeof document !== "undefined" && typeof document.requestStorageAccess === "function");
  if (!hasApi) { writePersisted("unsupported", storage); return { store: null, result: { state: "unsupported", reason: "document.requestStorageAccess missing" } }; }
  const browser = opts.browser ?? detectBrowser(typeof navigator !== "undefined" ? navigator.userAgent : "");
  if (browser === "safari") { writePersisted("unsupported", storage); return { store: null, result: { state: "unsupported", reason: "WebKit keeps storage partitioned after a grant (Phase 0)" } }; }
  if (!opts.explicit) {
    const p = readPersisted(storage);
    if (p === "denied" || p === "unsupported") return { store: null, result: { state: p, reason: "persisted" } };
  }
  const open = opts.openFrame ?? domFrameOpener;
  const url = opts.frameUrl ?? frameUrlFor(opts.cdn);
  const returnUrl = opts.returnUrl ?? (typeof location !== "undefined" ? location.href : "");
  const visitUrl = optinUrlFor(new URL(url).origin, returnUrl);

  let client: FrameClient;
  try { client = await hiddenClient(url, open); }
  catch (e) { return { store: null, result: { state: "unsupported", reason: `frame unavailable: ${(e as Error)?.message ?? String(e)}` } }; }

  const settle = (g: GrantResult, c: FrameClient): CrossSiteConnection => {
    const extra = { ...(g.reason ? { reason: g.reason } : {}), ...(g.path ? { path: g.path } : {}) };
    switch (g.state) {
      case "granted": writePersisted("granted", storage); return { store: new CrossSiteStore(c), result: { state: "granted", ...extra } };
      case "denied": writePersisted("denied", storage); return { store: null, result: { state: "denied", ...extra } };
      case "unsupported": writePersisted("unsupported", storage); return { store: null, result: { state: "unsupported", ...extra } };
      case "needs-visit": return { store: null, result: { state: "needs-visit", visitUrl, ...extra } };
      case "needs-click": return { store: null, result: { state: "needs-click", ...extra } };
    }
  };

  let g: GrantResult;
  try { g = await client.call<"grant">({ op: "grant", mode: "silent" }, { timeoutMs: opts.timeoutMs ?? DEFAULT_CALL_TIMEOUT_MS }); }
  catch (e) { return { store: null, result: { state: "unsupported", reason: `grant failed: ${(e as Error)?.message ?? String(e)}` } }; }
  if (g.state !== "needs-click" || !opts.explicit) {
    const conn = settle(g, client);
    if (g.state === "unsupported" || g.state === "denied") dropHidden(url, client);
    return conn;
  }

  // The browser wants the click inside the frame: the developer mounts the frame's button somewhere visible.
  const mount = async (container: HTMLElement, mountOpts: { onProgress?: (stage: GrantProgress) => void } = {}): Promise<CrossSiteResult> => {
    let visible: FrameClient;
    try { visible = await open(url, container); }
    catch (e) { return { state: "unsupported", reason: `button frame unavailable: ${(e as Error)?.message ?? String(e)}` }; }
    let vg: GrantResult;
    try { vg = await visible.call<"grant">({ op: "grant", mode: "await-click" }, { timeoutMs: null, onProgress: mountOpts.onProgress }); }
    catch (e) { visible.close(); return { state: "unsupported", reason: `grant failed: ${(e as Error)?.message ?? String(e)}` }; }
    if (vg.state !== "granted") { visible.close(); return settle(vg, visible).result; }
    // Granted in the visible frame. The permission is now persisted, so the hidden frame should be able to grant
    // silently; if it can, it becomes the store and the button frame goes away. Otherwise keep the visible
    // frame (collapsed) as the store for this document.
    let conn: CrossSiteConnection;
    try {
      const hg = await client.call<"grant">({ op: "grant", mode: "silent" });
      conn = hg.state === "granted" ? settle(hg, client) : settle(vg, visible);
    } catch { conn = settle(vg, visible); }
    if (conn.store?.client === client) visible.close();
    else { try { (visible.port.iframe as HTMLIFrameElement | undefined)?.setAttribute("style", hiddenStyle); } catch { /* best effort */ } }
    if (conn.store) opts.onGranted?.(conn.store);
    return conn.result;
  };
  return { store: null, result: { state: "needs-click", mount, ...(g.reason ? { reason: g.reason } : {}) } };
}
