import { describe, expect, it } from "vitest";
import { FrameServer } from "../src/cache/frameServer";
import type { FrameRequest } from "../src/cache/protocol";
import { CACHE_NAME } from "../src/cache/types";
import { sha256 } from "../src/sha";
import { fakePlatform } from "./helpers/fakePlatform";

const req = <R extends Omit<FrameRequest, "v" | "id">>(r: R, id = 1): FrameRequest => ({ v: 1, id, ...r } as FrameRequest);

describe("FrameServer grant", () => {
  it("Chrome: silent grant with a persisted permission uses handle.caches, never the globals", async () => {
    const fp = await fakePlatform({ mode: "chrome", silent: "grant" });
    const s = new FrameServer({ platform: fp.platform });
    expect(await s.handle(req({ op: "hello" }))).toMatchObject({ ok: true, result: { hasStorageAccess: false, granted: false } });
    const res = await s.handle(req({ op: "grant", mode: "silent" }));
    expect(res).toMatchObject({ ok: true, result: { state: "granted", path: "chrome-handle" } });
    expect(fp.rsaCalls).toEqual([{ all: true, inGesture: false }]);
    const sha = await sha256(new Uint8Array([1, 2, 3]));
    await s.handle(req({ op: "put", sha, buf: new Uint8Array([1, 2, 3]).buffer, modelId: "m" }));
    expect(fp.handleCaches.caches.get(CACHE_NAME)!.entries.has(`https://cdn.test/chunks/${sha}`)).toBe(true);
    expect(fp.globalCaches.caches.get(CACHE_NAME)!.entries.has(`https://cdn.test/chunks/${sha}`)).toBe(false);
    expect(await s.handle(req({ op: "status" }))).toMatchObject({ ok: true, result: { quotaBytes: 1000, usageBytes: 3, chunks: 1 } });
    // A second grant is idempotent and does not call requestStorageAccess again.
    expect(await s.handle(req({ op: "grant", mode: "silent" }))).toMatchObject({ ok: true, result: { state: "granted" } });
    expect(fp.rsaCalls).toHaveLength(1);
  });

  it("Chrome: silent request rejected → needs-click; the await-click grant runs requestStorageAccess inside the gesture", async () => {
    const fp = await fakePlatform({ mode: "chrome" });
    const s = new FrameServer({ platform: fp.platform });
    expect(await s.handle(req({ op: "grant", mode: "silent" }))).toMatchObject({ ok: true, result: { state: "needs-click" } });
    expect(fp.globalsTouched).toBe(0);
    const pending = s.handle(req({ op: "grant", mode: "await-click" }, 2));
    await new Promise((r) => setTimeout(r, 5));
    expect(fp.rsaCalls).toHaveLength(1); // nothing until the click
    fp.click();
    expect(await pending).toMatchObject({ ok: true, result: { state: "granted", path: "chrome-handle" } });
    expect(fp.rsaCalls[1]).toEqual({ all: true, inGesture: true });
  });

  it("Chrome: rejected inside the gesture with permission 'prompt' → needs-visit; 'denied' → denied; no permissions API → needs-visit", async () => {
    for (const [permission, state] of [["prompt", "needs-visit"], ["denied", "denied"], ["throws", "needs-visit"]] as const) {
      const fp = await fakePlatform({ mode: "chrome", gesture: "reject", permission });
      const s = new FrameServer({ platform: fp.platform });
      const p = s.handle(req({ op: "grant", mode: "await-click" }));
      fp.click();
      expect(await p, permission).toMatchObject({ ok: true, result: { state, reason: expect.stringContaining("NotAllowedError") } });
      expect(s.granted).toBe(false);
    }
  });

  it("a grant that resolves WITHOUT a handle (Firefox, Safari) is unsupported: cookie access only, caches stays partitioned", async () => {
    const fp = await fakePlatform({ mode: "firefox", silent: "grant" });
    const s = new FrameServer({ platform: fp.platform });
    expect(await s.handle(req({ op: "grant", mode: "silent" }))).toMatchObject({ ok: true, result: { state: "unsupported", reason: expect.stringMatching(/without a storage-access handle/) } });
    expect(fp.rsaCalls).toEqual([{ all: true, inGesture: false }]); // always the {all: true} form: a handle or nothing
    expect(fp.globalsTouched).toBe(0); // the frame's own globals are never used for chunk storage
    expect(s.granted).toBe(false);
    expect(await s.handle(req({ op: "status" }))).toMatchObject({ ok: false, code: "no_grant" });
    const fp2 = await fakePlatform({ mode: "firefox" });
    const s2 = new FrameServer({ platform: fp2.platform });
    const p = s2.handle(req({ op: "grant", mode: "await-click" }));
    fp2.click();
    expect(await p).toMatchObject({ result: { state: "unsupported" } });
    expect(fp2.logs.at(-1)).toMatch(/grant → unsupported: requestStorageAccess resolved without a storage-access handle/);
  });

  it("rejection shape without a permissions API: NotAllowedError inside the gesture → needs-visit (no path), reason carries the rejection", async () => {
    const fp = await fakePlatform({ mode: "chrome", gesture: "reject", permission: "throws" });
    const s = new FrameServer({ platform: fp.platform });
    const p = s.handle(req({ op: "grant", mode: "await-click" }));
    fp.click();
    const res = await p;
    expect(res).toEqual({ v: 1, id: 1, ok: true, ms: expect.any(Number), result: { state: "needs-visit", reason: "NotAllowedError: requestStorageAccess not allowed (permission: unknown)" } });
    expect(fp.rsaCalls).toEqual([{ all: true, inGesture: true }]);
    expect(s.granted).toBe(false);
    expect(fp.logs).toEqual(expect.arrayContaining([expect.stringMatching(/requestStorageAccess rejected \(await-click\): NotAllowedError/), expect.stringMatching(/grant → needs-visit: NotAllowedError/)]));
    const fp2 = await fakePlatform({ mode: "chrome", gesture: "reject", permission: "throws" });
    expect(await new FrameServer({ platform: fp2.platform }).handle(req({ op: "grant", mode: "silent" }))).toMatchObject({ result: { state: "needs-click", reason: "NotAllowedError: requestStorageAccess not allowed" } });
  });

  it("silent grants report only the requesting stage", async () => {
    const fp = await fakePlatform({ mode: "chrome", silent: "grant" });
    const stages: string[] = [];
    await new FrameServer({ platform: fp.platform }).handle(req({ op: "grant", mode: "silent" }), undefined, (st) => stages.push(st));
    expect(stages).toEqual(["requesting"]);
  });

  it("no requestStorageAccess → unsupported", async () => {
    const fp = await fakePlatform({ mode: "no-api" });
    const s = new FrameServer({ platform: fp.platform });
    expect(await s.handle(req({ op: "grant", mode: "silent" }))).toMatchObject({ ok: true, result: { state: "unsupported" } });
  });

  it("marker probe: a grant that cannot see the top-level marker is needs-visit, not granted", async () => {
    const fp = await fakePlatform({ mode: "chrome", silent: "grant", markerInHandle: false });
    const s = new FrameServer({ platform: fp.platform });
    expect(await s.handle(req({ op: "grant", mode: "silent" }))).toMatchObject({ ok: true, result: { state: "needs-visit", path: "chrome-handle" } });
    expect(s.granted).toBe(false);
    expect(await s.handle(req({ op: "status" }))).toMatchObject({ ok: false, code: "no_grant" });
  });

  it("every storage op before a grant answers no_grant without touching caches, indexedDB or storage (structural rule)", async () => {
    const fp = await fakePlatform({ mode: "chrome" });
    const s = new FrameServer({ platform: fp.platform });
    const sha = "a".repeat(64);
    for (const r of [req({ op: "hello" }), req({ op: "get", sha }), req({ op: "has", sha }), req({ op: "status" }), req({ op: "evict", shas: [sha] }), req({ op: "evictModel", modelId: "m" }), req({ op: "clear" }), req({ op: "fetch", sha, bytes: 1, modelId: "m" }), req({ op: "put", sha, buf: new ArrayBuffer(1) })]) {
      const res = await s.handle(r);
      if (r.op !== "hello") expect(res, r.op).toMatchObject({ ok: false, code: "no_grant" });
    }
    expect(fp.globalsTouched).toBe(0);
    expect(fp.rsaCalls).toEqual([]);
    // The grant afterwards is the first thing that reaches the globals.
    const p = s.handle(req({ op: "grant", mode: "await-click" }, 99));
    fp.click();
    expect(await p).toMatchObject({ result: { state: "granted", path: "chrome-handle" } });
    expect(fp.globalsTouched).toBe(1); // only for indexedDB/estimate fallbacks next to the handle
  });
});

describe("FrameServer ops", () => {
  async function granted(o: Parameters<typeof fakePlatform>[0] = {}) {
    const fp = await fakePlatform({ mode: "chrome", silent: "grant", ...o });
    const s = new FrameServer({ platform: fp.platform });
    await s.handle(req({ op: "grant", mode: "silent" }));
    return { fp, s };
  }

  it("fetch: miss → same-origin fetch, cached, fromCache false; hit → fromCache true; errors map to codes", async () => {
    const body = new Uint8Array([5, 6, 7, 8]);
    const sha = await sha256(body);
    const { fp, s } = await granted({ chunks: new Map([[sha, body]]) });
    const r1 = await s.handle(req({ op: "fetch", sha, bytes: 4, modelId: "m" }));
    expect(r1).toMatchObject({ ok: true, result: { fromCache: false } });
    expect(new Uint8Array((r1 as { result: { buf: ArrayBuffer } }).result.buf)).toEqual(body);
    expect(fp.fetches).toEqual([`/chunks/${sha}`]);
    const r2 = await s.handle(req({ op: "fetch", sha, bytes: 4, modelId: "m" }));
    expect(r2).toMatchObject({ ok: true, result: { fromCache: true } });
    expect(fp.fetches).toHaveLength(1);
    expect(await s.handle(req({ op: "fetch", sha, bytes: 5, modelId: "m" }))).toMatchObject({ ok: true, result: { fromCache: true } }); // length is the parent's problem to verify
    const other = "b".repeat(64);
    fp.faults.set(other, [503, "network"]);
    expect(await s.handle(req({ op: "fetch", sha: other, bytes: 4, modelId: "m" }))).toMatchObject({ ok: false, code: "http_503" });
    expect(await s.handle(req({ op: "fetch", sha: other, bytes: 4, modelId: "m" }))).toMatchObject({ ok: false, code: "network" });
    expect(await s.handle(req({ op: "fetch", sha: other, bytes: 4, modelId: "m" }))).toMatchObject({ ok: false, code: "http_404" });
  });

  it("fetch under quota with nothing evictable still delivers the bytes and flags quota", async () => {
    const body = new Uint8Array(500);
    const sha = await sha256(body);
    const { s } = await granted({ chunks: new Map([[sha, body]]), capacity: 1024 + 100 });
    const r = await s.handle(req({ op: "fetch", sha, bytes: 500, modelId: "m" }));
    expect(r).toMatchObject({ ok: true, result: { fromCache: false, quota: true } });
    expect((r as { result: { buf: ArrayBuffer } }).result.buf.byteLength).toBe(500);
    expect(await s.handle(req({ op: "put", sha, buf: body.buffer.slice(0), modelId: "m" }))).toMatchObject({ ok: false, code: "quota" });
  });

  it("get/put/has/evict/evictModel/clear round-trip and time each op", async () => {
    const { s } = await granted();
    const a = new Uint8Array([1]), b = new Uint8Array([2]);
    const sa = await sha256(a), sb = await sha256(b);
    await s.handle(req({ op: "put", sha: sa, buf: a.buffer.slice(0), modelId: "m1" }));
    await s.handle(req({ op: "put", sha: sb, buf: b.buffer.slice(0), modelId: "m2" }));
    const got = await s.handle(req({ op: "get", sha: sa, modelId: "m1" }));
    expect(got.ok && got.result instanceof ArrayBuffer && new Uint8Array(got.result)[0]).toBe(1);
    expect(typeof got.ms).toBe("number");
    expect(await s.handle(req({ op: "get", sha: "c".repeat(64) }))).toMatchObject({ ok: true, result: null });
    expect(await s.handle(req({ op: "has", sha: sb }))).toMatchObject({ result: true });
    await s.handle(req({ op: "evictModel", modelId: "m2" }));
    expect(await s.handle(req({ op: "has", sha: sb }))).toMatchObject({ result: false });
    await s.handle(req({ op: "evict", shas: [sa] }));
    expect(await s.handle(req({ op: "status" }))).toMatchObject({ result: { chunks: 0 } });
    await s.handle(req({ op: "put", sha: sa, buf: a.buffer.slice(0) }));
    await s.handle(req({ op: "clear" }));
    expect(await s.handle(req({ op: "status" }))).toMatchObject({ result: { chunks: 0, usageBytes: 0 } });
  });
});
