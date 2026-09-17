import { describe, expect, it, vi } from "vitest";
import { connectCrossSite, CrossSiteStore, FrameClient, iframePort, readPersisted, writePersisted, type FramePort, CROSSSITE_KEY } from "../src/cache/crosssite";
import { FrameServer } from "../src/cache/frameServer";
import { parseRequest, transferablesOf, PROTOCOL_VERSION } from "../src/cache/protocol";
import { AbortedError, ChunkError } from "../src/errors";
import { fetchChunks, type ChunkResult } from "../src/fetcher";
import type { PlanChunk } from "../src/plan";
import { sha256 } from "../src/sha";
import { fakePlatform, type FakePlatformOptions } from "./helpers/fakePlatform";

/** An in-memory port wired straight to a FrameServer (what the real iframe does, minus the browser). */
function loopback(server: FrameServer, o: { delayMs?: number; drop?: (op: string) => boolean } = {}): FramePort {
  const handlers = new Set<(data: unknown) => void>();
  return {
    post(msg) {
      const req = parseRequest(msg);
      if (!req || o.drop?.(req.op)) return;
      void server.handle(req).then(async (res) => {
        if (o.delayMs) await new Promise((r) => setTimeout(r, o.delayMs));
        // Structured-clone the response the way postMessage would (transfer list aside).
        void transferablesOf(res);
        for (const h of handlers) h(JSON.parse(JSON.stringify(res, (_k, v) => (v instanceof ArrayBuffer ? { __buf: [...new Uint8Array(v)] } : v)), (_k, v) => (v && typeof v === "object" && "__buf" in v ? new Uint8Array(v.__buf).buffer : v)));
      });
    },
    listen(h) { handlers.add(h); return () => handlers.delete(h); },
    close() { handlers.clear(); },
  };
}

function memStorage(): Storage {
  const m = new Map<string, string>();
  return { getItem: (k) => m.get(k) ?? null, setItem: (k, v) => void m.set(k, v), removeItem: (k) => void m.delete(k), clear: () => m.clear(), key: () => null, length: 0 } as Storage;
}

describe("iframePort", () => {
  it("accepts only messages from the frame's origin and window, posts with an explicit targetOrigin", () => {
    const listeners = new Set<(ev: MessageEvent) => void>();
    const win = { addEventListener: (_: string, l: EventListener) => listeners.add(l as never), removeEventListener: (_: string, l: EventListener) => listeners.delete(l as never) };
    const frameWindow = { postMessage: vi.fn() };
    const iframe = { contentWindow: frameWindow, remove: vi.fn() };
    const port = iframePort(iframe, "https://cdn.dianome.dev", win);
    const got: unknown[] = [];
    port.listen((d) => got.push(d));
    const fire = (origin: string, source: unknown, data: unknown) => { for (const l of listeners) l({ origin, source, data } as MessageEvent); };
    fire("https://cdn.dianome.dev", frameWindow, { ok: 1 });
    fire("https://evil.example", frameWindow, { ok: 2 });
    fire("https://cdn.dianome.dev", { other: true }, { ok: 3 });
    fire("null", frameWindow, { ok: 4 });
    expect(got).toEqual([{ ok: 1 }]);
    const buf = new ArrayBuffer(2);
    port.post({ v: 1, id: 1, op: "put", sha: "a".repeat(64), buf }, [buf]);
    expect(frameWindow.postMessage).toHaveBeenCalledWith({ v: 1, id: 1, op: "put", sha: "a".repeat(64), buf }, "https://cdn.dianome.dev", [buf]);
    port.close();
    expect(listeners.size).toBe(0);
    expect(iframe.remove).toHaveBeenCalled();
  });
});

describe("FrameClient", () => {
  it("correlates replies, records timings, times out, and rejects pending calls on close", async () => {
    const fp = await fakePlatform({ silent: "grant" });
    const server = new FrameServer({ platform: fp.platform });
    const client = new FrameClient(loopback(server), 50);
    expect(await client.call<"hello">({ op: "hello" })).toMatchObject({ v: PROTOCOL_VERSION });
    expect(await client.call<"grant">({ op: "grant", mode: "silent" })).toMatchObject({ state: "granted" });
    const t = client.timings();
    expect(t.hello!.count).toBe(1);
    expect(t.grant!.frameMs).toBeGreaterThanOrEqual(0);
    const slow = new FrameClient(loopback(server, { drop: (op) => op === "status" }), 20);
    await expect(slow.call<"status">({ op: "status" })).rejects.toMatchObject({ code: "frame_timeout" });
    const p = slow.call<"status">({ op: "status" }, { timeoutMs: null });
    slow.close();
    await expect(p).rejects.toMatchObject({ code: "frame_error" });
    await expect(slow.call<"hello">({ op: "hello" })).rejects.toMatchObject({ code: "frame_error" });
  });

  it("honours an AbortSignal on a call", async () => {
    const fp = await fakePlatform({ silent: "grant" });
    const server = new FrameServer({ platform: fp.platform });
    const client = new FrameClient(loopback(server, { drop: () => true }));
    const ac = new AbortController();
    const p = client.call<"hello">({ op: "hello" }, { signal: ac.signal });
    ac.abort();
    await expect(p).rejects.toBeInstanceOf(AbortedError);
  });
});

describe("CrossSiteStore", () => {
  async function store(o: FakePlatformOptions = {}) {
    const fp = await fakePlatform({ silent: "grant", ...o });
    const server = new FrameServer({ platform: fp.platform });
    const client = new FrameClient(loopback(server));
    await client.call<"grant">({ op: "grant", mode: "silent" });
    return { fp, server, store: new CrossSiteStore(client) };
  }

  it("implements the ChunkStore ops over the frame", async () => {
    const { store: s } = await store();
    const a = new Uint8Array([1, 2]);
    const sha = await sha256(a);
    expect(s.mode).toBe("cross-site");
    await s.put(sha, a.buffer.slice(0), "m");
    expect(await s.has(sha)).toBe(true);
    expect(new Uint8Array((await s.get(sha, "m"))!)).toEqual(a);
    expect(await s.status()).toMatchObject({ chunks: 1, usageBytes: 2 });
    await s.evictModel("m");
    expect(await s.has(sha)).toBe(false);
    await s.put(sha, a.buffer.slice(0));
    await s.evict([sha]);
    await s.clear();
    expect(await s.status()).toMatchObject({ chunks: 0 });
    expect(Object.keys(s.timings()).sort()).toEqual(["clear", "evict", "evictModel", "get", "grant", "has", "put", "status"]);
  });

  it("put maps the frame's quota code to a QuotaExceededError", async () => {
    const { store: s } = await store({ capacity: 1024 + 4 });
    const big = new Uint8Array(10);
    await expect(s.put(await sha256(big), big.buffer, "m")).rejects.toMatchObject({ name: "QuotaExceededError" });
  });

  it("fetch reports source and transfer time, and maps http/network codes to ChunkError", async () => {
    const body = new Uint8Array([7, 7, 7]);
    const sha = await sha256(body);
    const { fp, store: s } = await store({ chunks: new Map([[sha, body]]) });
    const r1 = await s.fetch(sha, 3, "m");
    expect(r1.fromCache).toBe(false);
    expect(r1.transferMs).toBeGreaterThanOrEqual(0);
    expect(new Uint8Array(r1.buf)).toEqual(body);
    expect((await s.fetch(sha, 3, "m")).fromCache).toBe(true);
    const other = "d".repeat(64);
    fp.faults.set(other, [502, "network"]);
    await expect(s.fetch(other, 3, "m")).rejects.toMatchObject({ code: "chunk_http", status: 502 });
    await expect(s.fetch(other, 3, "m")).rejects.toMatchObject({ code: "chunk_network" });
    await expect(s.fetch(other, 3, "m")).rejects.toMatchObject({ code: "chunk_http", status: 404 });
  });

  it("fetchChunks on the cross-site path: the parent never fetches, verifies what the frame returns, refetches a corrupt cached chunk once, retries 5xx", async () => {
    const chunks = new Map<string, Uint8Array>();
    const order: PlanChunk[] = [];
    for (let i = 0; i < 4; i++) { const b = new Uint8Array(100).fill(i + 1); const sha = await sha256(b); chunks.set(sha, b); order.push({ sha, bytes: 100, group: `g${i}`, groupIndex: i }); }
    const { fp, store: s } = await store({ chunks });
    const parentFetch = vi.fn();
    const results: ChunkResult[] = [];
    fp.faults.set(order[2]!.sha, [500]);
    const stats = await fetchChunks(order, { cdn: "https://cdn.test", modelId: "m", store: s, fetch: parentFetch, sleep: async () => {}, onChunk: (r) => results.push(r) });
    expect(parentFetch).not.toHaveBeenCalled();
    expect(results.map((r) => r.source)).toEqual(["network", "network", "network", "network"]);
    expect(results.find((r) => r.sha === order[2]!.sha)!.attempts).toBe(2);
    expect(stats.transferMs).toBeGreaterThanOrEqual(0);

    // Corrupt one cached body inside the frame's storage; the second load evicts and refetches only that chunk.
    const victim = order[1]!.sha;
    const cache = fp.handleCaches.caches.get("dianome-v1")!;
    cache.entries.get(`https://cdn.test/chunks/${victim}`)!.body[0] = 99;
    const again: ChunkResult[] = [];
    const st2 = await fetchChunks(order, { cdn: "https://cdn.test", modelId: "m", store: s, fetch: parentFetch, sleep: async () => {}, onChunk: (r) => again.push(r) });
    expect(new Map(again.map((r) => [r.sha, r.source]))).toEqual(new Map(order.map((c) => [c.sha, c.sha === victim ? "network" : "cross-site-cache"])));
    expect(st2.corruptEvicted).toBe(1);
    expect(fp.fetches.filter((u) => u.endsWith(victim))).toHaveLength(2);
  });

  it("fetchChunks: a quota-flagged frame fetch switches the session to cache none", async () => {
    const b = new Uint8Array(400).fill(1);
    const sha = await sha256(b);
    const { store: s } = await store({ chunks: new Map([[sha, b]]), capacity: 1024 + 10 });
    let disabled = 0;
    const results: ChunkResult[] = [];
    const st = await fetchChunks([{ sha, bytes: 400, group: "g", groupIndex: 0 }], { cdn: "https://cdn.test", modelId: "m", store: s, fetch: vi.fn(), onChunk: (r) => results.push(r), onCacheDisabled: () => disabled++ });
    expect(st.cacheDisabled).toBe(true);
    expect(disabled).toBe(1);
    expect(results[0]!.source).toBe("network");
  });
});

describe("connectCrossSite", () => {
  function harness(o: FakePlatformOptions = {}) {
    const storage = memStorage();
    const servers: FrameServer[] = [];
    const opened: { url: string; visible: boolean }[] = [];
    let fp: Awaited<ReturnType<typeof fakePlatform>>;
    const openFrame = async (url: string, visibleIn: HTMLElement | null) => {
      opened.push({ url, visible: visibleIn !== null });
      fp ??= await fakePlatform(o);
      const server = new FrameServer({ platform: fp.platform });
      servers.push(server);
      const c = new FrameClient(loopback(server));
      await c.call<"hello">({ op: "hello" });
      return c;
    };
    const base = { cdn: "https://cdn.test", frameUrl: `https://cdn.test/frame/v1/index.html?t=${Math.random()}`, storage, openFrame, returnUrl: "https://site-a.pages.dev/", hasDocument: true, hasStorageAccessApi: true, browser: "chrome" as const };
    return { storage, opened, servers, base, fp: () => fp };
  }

  it("unsupported without the API or on Safari, persisted; auto mode honours persisted denied/unsupported without opening a frame", async () => {
    const h = harness();
    expect((await connectCrossSite({ ...h.base, explicit: true, hasStorageAccessApi: false })).result.state).toBe("unsupported");
    expect(readPersisted(h.storage)).toBe("unsupported");
    writePersisted(null, h.storage);
    expect((await connectCrossSite({ ...h.base, explicit: true, browser: "safari" })).result.state).toBe("unsupported");
    expect(readPersisted(h.storage)).toBe("unsupported");
    expect(h.opened).toHaveLength(0);
    writePersisted("denied", h.storage);
    expect((await connectCrossSite({ ...h.base, explicit: false })).result).toMatchObject({ state: "denied", reason: "persisted" });
    expect(h.opened).toHaveLength(0);
    expect((await connectCrossSite({ ...h.base, explicit: false, hasDocument: false })).result.state).toBe("unsupported");
  });

  it("silent grant (persisted permission) → granted store, persisted", async () => {
    const h = harness({ silent: "grant" });
    const c = await connectCrossSite({ ...h.base, explicit: false });
    expect(c.result.state).toBe("granted");
    expect(c.store).toBeInstanceOf(CrossSiteStore);
    expect(readPersisted(h.storage)).toBe("granted");
    expect(h.opened).toEqual([{ url: h.base.frameUrl, visible: false }]);
    // The hidden frame is shared: a second connection reuses it.
    await connectCrossSite({ ...h.base, explicit: true });
    expect(h.opened).toHaveLength(1);
  });

  it("needs-click: auto mode returns it without mounting; explicit mode mounts, grants in the visible frame, then adopts the hidden frame", async () => {
    const h = harness();
    const auto = await connectCrossSite({ ...h.base, explicit: false });
    expect(auto.result.state).toBe("needs-click");
    expect(auto.result.mount).toBeUndefined();
    expect(readPersisted(h.storage)).toBeNull();

    const adopted: CrossSiteStore[] = [];
    const ex = await connectCrossSite({ ...h.base, explicit: true, onGranted: (s) => adopted.push(s) });
    expect(ex.result.state).toBe("needs-click");
    const pending = ex.result.mount!({} as HTMLElement);
    await new Promise((r) => setTimeout(r, 5));
    expect(h.opened).toEqual([{ url: h.base.frameUrl, visible: false }, { url: h.base.frameUrl, visible: true }]);
    h.fp().click();
    const r = await pending;
    expect(r.state).toBe("granted");
    expect(readPersisted(h.storage)).toBe("granted");
    expect(adopted).toHaveLength(1);
    // The permission is now persisted at the browser level, so the hidden frame granted silently and is the store.
    expect(h.servers[0]!.granted).toBe(true);
    expect(await adopted[0]!.status()).toMatchObject({ chunks: 0 });
    // Later auto connections hit the persisted "granted" and the hidden frame's silent grant.
    const later = await connectCrossSite({ ...h.base, explicit: false });
    expect(later.result.state).toBe("granted");
  });

  it("needs-visit carries the opt-in URL with the return address; denied is persisted", async () => {
    const h = harness({ gesture: "reject", permission: "prompt" });
    const ex = await connectCrossSite({ ...h.base, explicit: true });
    const pending = ex.result.mount!({} as HTMLElement);
    await new Promise((r) => setTimeout(r, 5));
    h.fp().click();
    const r = await pending;
    expect(r).toMatchObject({ state: "needs-visit", visitUrl: "https://cdn.test/frame/v1/optin.html?return=https%3A%2F%2Fsite-a.pages.dev%2F" });
    expect(readPersisted(h.storage)).toBeNull();

    // A browser-level denial is detected on the silent attempt (permission state "denied"): no click needed.
    const d = harness({ gesture: "reject", permission: "denied" });
    const ex2 = await connectCrossSite({ ...d.base, explicit: true });
    expect(ex2.result).toMatchObject({ state: "denied" });
    expect(ex2.result.mount).toBeUndefined();
    expect(d.storage.getItem(CROSSSITE_KEY)).toBe("denied");
    // Explicit calls re-try a persisted denial (the user may have changed the site setting); auto does not.
    const again = await connectCrossSite({ ...d.base, explicit: true });
    expect(again.result.state).toBe("denied");
    expect(d.opened).toHaveLength(2);
  });
});
