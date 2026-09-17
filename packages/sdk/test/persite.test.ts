import { describe, expect, it } from "vitest";
import { sha256 } from "../src/sha";
import { clock, makeStore } from "./helpers/env";

const buf = (n: number, fillByte = 1): ArrayBuffer => new Uint8Array(n).fill(fillByte).buffer;
const shaOf = (name: string): string => name.padEnd(64, "0");

describe("PerSiteStore", () => {
  it("round-trips a chunk under the CDN URL key and tracks metadata", async () => {
    const { store, caches } = makeStore({ cdn: "https://cdn.dianome.dev", now: clock() });
    const sha = await sha256(buf(16));
    expect(await store.has(sha)).toBe(false);
    expect(await store.get(sha)).toBeNull();
    await store.put(sha, buf(16), "model-a");
    expect(await store.has(sha)).toBe(true);
    expect(new Uint8Array((await store.get(sha))!)).toEqual(new Uint8Array(16).fill(1));
    const cache = caches.caches.get("dianome-v1")!;
    expect([...cache.entries.keys()]).toEqual([`https://cdn.dianome.dev/chunks/${sha}`]);
    expect(cache.entries.get(`https://cdn.dianome.dev/chunks/${sha}`)!.headers["content-type"]).toBe("application/octet-stream");
    expect(await store.list()).toEqual([{ sha, bytes: 16, lastAccess: 1002, modelIds: ["model-a"] }]);
    expect(await store.status()).toEqual({ quotaBytes: null, usageBytes: 16, chunks: 1 });
  });

  it("get() bumps lastAccess and records the model; put() merges owners", async () => {
    const { store } = makeStore({ now: clock() });
    await store.put(shaOf("a"), buf(8), "m1");
    await store.put(shaOf("b"), buf(8), "m1");
    await store.get(shaOf("a"), "m2");
    const list = await store.list();
    expect(list.map((m) => m.sha)).toEqual([shaOf("b"), shaOf("a")]);
    expect(list[1]!.modelIds).toEqual(["m1", "m2"]);
    await store.put(shaOf("b"), buf(8), "m3");
    expect((await store.list()).find((m) => m.sha === shaOf("b"))!.modelIds).toEqual(["m1", "m3"]);
  });

  it("evicts least-recently-used chunks first, skipping the model being loaded", async () => {
    const { store, caches } = makeStore({ now: clock(), capacity: 45 });
    await store.put(shaOf("a"), buf(10), "old");
    await store.put(shaOf("b"), buf(10), "old");
    await store.put(shaOf("c"), buf(10), "cur");
    await store.put(shaOf("d"), buf(10), "old");
    await store.get(shaOf("a"), "old"); // a is now the most recent of the old ones; b is the LRU
    await store.put(shaOf("e"), buf(10), "cur"); // needs 10: evicts b
    expect((await store.list()).map((m) => m.sha)).toEqual([shaOf("c"), shaOf("d"), shaOf("a"), shaOf("e")]);
    await store.put(shaOf("f"), buf(25), "cur"); // needs 25: evicts d then a (c and e belong to cur)
    expect((await store.list()).map((m) => m.sha)).toEqual([shaOf("c"), shaOf("e"), shaOf("f")]);
    expect(caches.used).toBe(45); // 10 + 10 + 25
  });

  it("rethrows QuotaExceededError when nothing evictable is left", async () => {
    const { store } = makeStore({ now: clock(), capacity: 20 });
    await store.put(shaOf("a"), buf(10), "cur");
    await store.put(shaOf("b"), buf(10), "cur");
    await expect(store.put(shaOf("c"), buf(10), "cur")).rejects.toMatchObject({ name: "QuotaExceededError" });
    // A write larger than the whole quota also fails after evicting everything evictable.
    const { store: s2 } = makeStore({ now: clock(), capacity: 20 });
    await s2.put(shaOf("x"), buf(10), "other");
    await expect(s2.put(shaOf("y"), buf(30), "cur")).rejects.toMatchObject({ name: "QuotaExceededError" });
    expect(await s2.list()).toEqual([]);
  });

  it("evict(), evictModel() and clear()", async () => {
    const { store, caches } = makeStore({ now: clock() });
    await store.put(shaOf("a"), buf(1), "m1");
    await store.put(shaOf("b"), buf(1), "m1");
    await store.put(shaOf("b"), buf(1), "m2");
    await store.put(shaOf("c"), buf(1), "m2");
    await store.evict([shaOf("a")]);
    expect(await store.has(shaOf("a"))).toBe(false);
    await store.evictModel("m1");
    expect((await store.list()).map((m) => [m.sha, m.modelIds])).toEqual([[shaOf("b"), ["m2"]], [shaOf("c"), ["m2"]]]);
    await store.clear();
    expect(await store.list()).toEqual([]);
    expect(caches.caches.has("dianome-v1")).toBe(false);
    // Usable again after clear.
    await store.put(shaOf("d"), buf(1), "m3");
    expect(await store.has(shaOf("d"))).toBe(true);
  });

  it("status() reports the storage estimate's quota when available", async () => {
    const { store } = makeStore({ quota: 10_000 });
    await store.put(shaOf("a"), buf(100));
    expect(await store.status()).toEqual({ quotaBytes: 10_000, usageBytes: 100, chunks: 1 });
  });
});
