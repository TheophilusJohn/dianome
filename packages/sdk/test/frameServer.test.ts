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

  it("Firefox: plain requestStorageAccess() as the first statement of the gesture, no handle; the store is built on the globals only after the grant", async () => {
    const fp = await fakePlatform({ mode: "firefox" });
    const s = new FrameServer({ platform: fp.platform });
    expect(fp.globalsTouched).toBe(0);
    const stages: string[] = [];
    const p = s.handle(req({ op: "grant", mode: "await-click" }), undefined, (st) => stages.push(st));
    await new Promise((r) => setTimeout(r, 2));
    expect(stages).toEqual(["waiting-click"]);
    fp.click();
    expect(await p).toMatchObject({ ok: true, result: { state: "granted", path: "firefox-globals" } });
    expect(fp.rsaCalls).toEqual([{ inGesture: true }]); // no {all: true} on Firefox
    expect(stages).toEqual(["waiting-click", "clicked", "requesting"]);
    expect(fp.globalsTouched).toBe(1);
    expect(fp.logs.some((l) => /requestStorageAccess\(\) resolved \(await-click\)/.test(l))).toBe(true);
    expect(fp.logs.at(-1)).toMatch(/grant → granted via firefox-globals/);
    const sha = await sha256(new Uint8Array([9]));
    await s.handle(req({ op: "put", sha, buf: new Uint8Array([9]).buffer }));
    expect(fp.globalCaches.caches.get(CACHE_NAME)!.entries.has(`https://cdn.test/chunks/${sha}`)).toBe(true);
    expect(fp.handleCaches.used).toBe(1024); // only the marker
  });

  it("Firefox: hasStorageAccess() already true at load → silent grant succeeds", async () => {
    const fp = await fakePlatform({ mode: "firefox", silent: "grant", hasStorageAccess: true });
    const s = new FrameServer({ platform: fp.platform });
    expect(await s.handle(req({ op: "hello" }))).toMatchObject({ result: { hasStorageAccess: true } });
    expect(await s.handle(req({ op: "grant", mode: "silent" }))).toMatchObject({ ok: true, result: { state: "granted", path: "firefox-globals" } });
  });

  it("Firefox rejection shape: NotAllowedError inside the gesture with no permissions API → needs-visit (no path), reason carries the rejection", async () => {
    // Firefox rejects requestStorageAccess() when the user has not interacted with the origin top-level (or dismissed
    // its prompt) and its permissions.query has no "storage-access" name: the frame sees a TypeError from query().
    const fp = await fakePlatform({ mode: "firefox", gesture: "reject", permission: "throws" });
    const s = new FrameServer({ platform: fp.platform });
    const p = s.handle(req({ op: "grant", mode: "await-click" }));
    fp.click();
    const res = await p;
    expect(res).toEqual({ v: 1, id: 1, ok: true, ms: expect.any(Number), result: { state: "needs-visit", reason: "NotAllowedError: requestStorageAccess not allowed (permission: unknown)" } });
    expect((res as { result: { path?: string } }).result.path).toBeUndefined();
    expect(fp.rsaCalls).toEqual([{ inGesture: true }]);
    expect(s.granted).toBe(false);
    expect(fp.logs).toEqual(expect.arrayContaining([expect.stringMatching(/requestStorageAccess rejected \(await-click\): NotAllowedError/), expect.stringMatching(/grant → needs-visit: NotAllowedError/)]));
    // The silent attempt rejecting the same way asks for the click first.
    const fp2 = await fakePlatform({ mode: "firefox", gesture: "reject", permission: "throws" });
    expect(await new FrameServer({ platform: fp2.platform }).handle(req({ op: "grant", mode: "silent" }))).toMatchObject({ result: { state: "needs-click", reason: "NotAllowedError: requestStorageAccess not allowed" } });
    expect(fp2.rsaCalls).toEqual([{ inGesture: false }]);
  });

  it("Firefox: a grant that resolves but cannot see the marker (no top-level visit yet) → needs-visit via firefox-globals", async () => {
    const fp = await fakePlatform({ mode: "firefox", silent: "grant", markerInGlobals: false });
    const s = new FrameServer({ platform: fp.platform });
    expect(await s.handle(req({ op: "grant", mode: "silent" }))).toMatchObject({ result: { state: "needs-visit", path: "firefox-globals", reason: expect.stringContaining("marker /frame/v1/marker not visible") } });
    expect(s.granted).toBe(false);
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

  it("every storage op before a grant answers no_grant", async () => {
    const fp = await fakePlatform();
    const s = new FrameServer({ platform: fp.platform });
    const sha = "a".repeat(64);
    for (const r of [req({ op: "get", sha }), req({ op: "has", sha }), req({ op: "status" }), req({ op: "evict", shas: [sha] }), req({ op: "evictModel", modelId: "m" }), req({ op: "clear" }), req({ op: "fetch", sha, bytes: 1, modelId: "m" })]) {
      expect(await s.handle(r), r.op).toMatchObject({ ok: false, code: "no_grant" });
    }
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
