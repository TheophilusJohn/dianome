import { describe, expect, it, vi } from "vitest";
import { connectCrossSite, CrossSiteStore, FrameClient, iframePort, readPersisted, writePersisted, type FramePort, CROSSSITE_KEY } from "../src/cache/crosssite";
import { FrameServer } from "../src/cache/frameServer";
import { epochNow, parseRequest, stampSent, transferablesOf, PROTOCOL_VERSION } from "../src/cache/protocol";
import { median } from "../src/telemetry";
import { AbortedError, ChunkError } from "../src/errors";
import { fetchChunks, type ChunkResult } from "../src/fetcher";
import type { PlanChunk } from "../src/plan";
import { sha256 } from "../src/sha";
import { fakePlatform, type FakePlatformOptions } from "./helpers/fakePlatform";

/** An in-memory port wired straight to a FrameServer (what the real iframe does, minus the browser). */
function loopback(server: FrameServer, o: { delayMs?: number; drop?: (op: string) => boolean; hopDelayMs?: number } = {}): FramePort {
  const handlers = new Set<(data: unknown) => void>();
  return {
    post(msg) {
      const receivedAt = epochNow();
      const req = parseRequest(msg);
      if (!req || o.drop?.(req.op)) return;
      const progress = (stage: string) => { for (const h of handlers) h({ v: 1, id: req.id, progress: stage }); };
      void server.handle(req, receivedAt, progress as never).then(async (res) => {
        if (o.delayMs) await new Promise((r) => setTimeout(r, o.delayMs)); // "frame work" after the op (not part of the hop)
        // Structured-clone the response the way postMessage would (transfer list aside), stamped right before delivery.
        void transferablesOf(res);
        const wire = JSON.stringify(stampSent(res), (_k, v) => (v instanceof ArrayBuffer ? { __buf: [...new Uint8Array(v)] } : v));
        if (o.hopDelayMs) await new Promise((r) => setTimeout(r, o.hopDelayMs)); // simulated postMessage latency (is part of the hop)
        for (const h of handlers) h(JSON.parse(wire, (_k, v) => (v && typeof v === "object" && "__buf" in v ? new Uint8Array(v.__buf).buffer : v)));
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

  it("fetch reports source and the postMessage hop, and maps http/network codes to ChunkError", async () => {
    const body = new Uint8Array([7, 7, 7]);
    const sha = await sha256(body);
    const { fp, store: s } = await store({ chunks: new Map([[sha, body]]) });
    const r1 = await s.fetch(sha, 3, "m");
    expect(r1.fromCache).toBe(false);
    expect(r1.transferMs).not.toBeNull();
    expect(r1.transferMs!).toBeGreaterThanOrEqual(0);
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
    expect(stats.transferSamples).toHaveLength(4);
    expect(stats.transferSamples.every((t) => t >= 0)).toBe(true);

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

  it("transferMs is the postMessage hop alone: frame work is excluded and concurrent fetches do not add up", async () => {
    const chunks = new Map<string, Uint8Array>();
    const order: PlanChunk[] = [];
    for (let i = 0; i < 12; i++) { const b = new Uint8Array(1000).fill(i + 1); const sha = await sha256(b); chunks.set(sha, b); order.push({ sha, bytes: 1000, group: "g", groupIndex: 0 }); }
    const fp = await fakePlatform({ silent: "grant", chunks });
    const server = new FrameServer({ platform: fp.platform });
    // 40 ms of "frame work" per op before the reply is stamped, 5 ms of simulated postMessage latency after.
    const client = new FrameClient(loopback(server, { delayMs: 40, hopDelayMs: 5 }));
    await client.call<"grant">({ op: "grant", mode: "silent" });
    const s = new CrossSiteStore(client);
    const t0 = Date.now();
    const stats = await fetchChunks(order, { cdn: "https://cdn.test", modelId: "m", store: s, concurrency: 6, fetch: vi.fn(), onChunk: () => {} });
    const wall = Date.now() - t0;
    expect(stats.transferSamples).toHaveLength(12);
    const med = median(stats.transferSamples);
    // Each hop is ~5 ms: never the 40 ms of frame work, never a sum over the 6 concurrent fetches (>= 45 ms would show either).
    expect(med).toBeGreaterThanOrEqual(4);
    expect(med).toBeLessThan(30);
    expect(Math.max(...stats.transferSamples)).toBeLessThan(40);
    expect(stats.transferSamples.reduce((a, b) => a + b, 0)).toBeLessThan(wall); // the old sum-of-round-trips exceeded wall time
    const tm = client.timings().fetch!;
    expect(tm.hopOutMs / tm.count).toBeLessThan(30);
    expect(tm.totalMs / tm.count).toBeGreaterThanOrEqual(40); // round trip still sees the frame work
    expect(tm.hopInMs).toBeGreaterThanOrEqual(0);
  });

  it("a frame that does not stamp its replies yields null transfer and no samples", async () => {
    const b = new Uint8Array([1, 2]);
    const sha = await sha256(b);
    const fp = await fakePlatform({ silent: "grant", chunks: new Map([[sha, b]]) });
    const server = new FrameServer({ platform: fp.platform });
    const handlers = new Set<(d: unknown) => void>();
    const unstamped: FramePort = {
      post(msg) { const req = parseRequest(msg); if (req) void server.handle(req).then((res) => { for (const h of handlers) h(JSON.parse(JSON.stringify(res, (_k, v) => (v instanceof ArrayBuffer ? { __buf: [...new Uint8Array(v)] } : v)), (_k, v) => (v && typeof v === "object" && "__buf" in v ? new Uint8Array(v.__buf).buffer : v))); }); },
      listen(h) { handlers.add(h); return () => handlers.delete(h); }, close() { handlers.clear(); },
    };
    const client = new FrameClient(unstamped);
    await client.call<"grant">({ op: "grant", mode: "silent" });
    const s = new CrossSiteStore(client);
    expect((await s.fetch(sha, 2, "m")).transferMs).toBeNull();
    const stats = await fetchChunks([{ sha, bytes: 2, group: "g", groupIndex: 0 }], { cdn: "https://cdn.test", modelId: "m", store: s, fetch: vi.fn(), onChunk: () => {} });
    expect(stats.transferSamples).toEqual([]);
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
    const stages: string[] = [];
    const pending = ex.result.mount!({} as HTMLElement, { onProgress: (st) => stages.push(st) });
    await new Promise((r) => setTimeout(r, 5));
    expect(h.opened).toEqual([{ url: h.base.frameUrl, visible: false }, { url: h.base.frameUrl, visible: true }]);
    expect(stages).toEqual(["waiting-click"]);
    h.fp().click();
    const r = await pending;
    expect(stages).toEqual(["waiting-click", "clicked", "requesting"]);
    expect(r.state).toBe("granted");
    expect(r.path).toBe("chrome-handle");
    expect(readPersisted(h.storage)).toBe("granted");
    expect(adopted).toHaveLength(1);
    // The permission is now persisted at the browser level, so the hidden frame granted silently and is the store.
    expect(h.servers[0]!.granted).toBe(true);
    expect(await adopted[0]!.status()).toMatchObject({ chunks: 0 });
    // Later auto connections hit the persisted "granted" and the hidden frame's silent grant.
    const later = await connectCrossSite({ ...h.base, explicit: false });
    expect(later.result.state).toBe("granted");
  });

  it("Firefox: the whole flow uses the plain call and reports path firefox-globals; a rejection in the gesture is needs-visit with the reason", async () => {
    const ok = harness({ mode: "firefox", browser: "firefox" });
    const c1 = await connectCrossSite({ ...ok.base, browser: "firefox", explicit: true });
    expect(c1.result.state).toBe("needs-click");
    const pending = c1.result.mount!({} as HTMLElement);
    await new Promise((r) => setTimeout(r, 5));
    ok.fp().click();
    const r1 = await pending;
    expect(r1).toMatchObject({ state: "granted", path: "firefox-globals" });
    expect(ok.fp().rsaCalls.every((c) => c.all === undefined)).toBe(true);

    const rej = harness({ mode: "firefox", browser: "firefox", gesture: "reject", permission: "throws" });
    const c2 = await connectCrossSite({ ...rej.base, browser: "firefox", explicit: true });
    const p2 = c2.result.mount!({} as HTMLElement);
    await new Promise((r) => setTimeout(r, 5));
    rej.fp().click();
    const r2 = await p2;
    expect(r2).toMatchObject({ state: "needs-visit", visitUrl: expect.stringContaining("/frame/v1/optin.html?return="), reason: "NotAllowedError: requestStorageAccess not allowed (permission: unknown)" });
    expect(r2.path).toBeUndefined();
    expect(readPersisted(rej.storage)).toBeNull();
  });

  it("needs-visit carries the opt-in URL with the return address; denied is persisted", async () => {
    const h = harness({ gesture: "reject", permission: "prompt" });
    const ex = await connectCrossSite({ ...h.base, explicit: true });
    const pending = ex.result.mount!({} as HTMLElement);
    await new Promise((r) => setTimeout(r, 5));
    h.fp().click();
    const r = await pending;
    expect(r).toMatchObject({ state: "needs-visit", visitUrl: "https://cdn.test/frame/v1/optin.html?return=https%3A%2F%2Fsite-a.pages.dev%2F", reason: expect.stringMatching(/^NotAllowedError/) });
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
