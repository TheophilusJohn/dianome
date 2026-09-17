import { describe, expect, it } from "vitest";
import { AbortedError, ChunkError } from "../src/errors";
import { fetchChunks, type ChunkResult } from "../src/fetcher";
import { planVariant } from "../src/plan";
import { synthModel } from "./fixtures/synth";
import { makeStore } from "./helpers/env";
import { FakeFetch } from "./helpers/fakeFetch";

const CDN = "https://cdn.test";
const noSleep = async (): Promise<void> => {};

async function setup(o: { chunkSize?: number } = {}) {
  const { manifest, chunks } = await synthModel(o);
  const plan = planVariant(manifest, "q4");
  const ff = new FakeFetch("https://api.test", CDN, chunks);
  return { plan, ff, chunks };
}

function run(plan: ReturnType<typeof planVariant>, ff: FakeFetch, extra: Partial<Parameters<typeof fetchChunks>[1]> = {}) {
  const results: ChunkResult[] = [];
  const sleeps: number[] = [];
  const p = fetchChunks(plan.order, {
    cdn: CDN, modelId: plan.modelId, store: null, fetch: ff.fetch, onChunk: (r) => results.push(r),
    sleep: async (ms) => { sleeps.push(ms); }, ...extra,
  });
  return { results, sleeps, done: p };
}

describe("fetchChunks", () => {
  it("fetches every chunk once, verified, with at most `concurrency` in flight", async () => {
    const { plan, ff } = await setup({ chunkSize: 4096 });
    ff.delayMs = 2;
    const { results, done } = run(plan, ff, { concurrency: 3 });
    const stats = await done;
    expect(results).toHaveLength(plan.order.length);
    expect(ff.chunkRequests()).toBe(plan.order.length);
    expect(ff.maxInFlight).toBeLessThanOrEqual(3);
    expect(results.every((r) => r.source === "network" && r.attempts === 1)).toBe(true);
    expect(stats.cacheDisabled).toBe(false);
    // Requests are issued in download order.
    expect(ff.requests.map((r) => r.url.slice(-64))).toEqual(plan.order.map((c) => c.sha));
  });

  it("retries network errors and 5xx with exponential backoff, then succeeds", async () => {
    const { plan, ff } = await setup();
    const sha = plan.order[1]!.sha;
    ff.fail(sha, { kind: "network" }, { kind: "status", status: 503 }, { kind: "status", status: 500 });
    const { results, sleeps, done } = run(plan, ff, { concurrency: 1, backoffMs: 100 });
    await done;
    expect(sleeps).toEqual([100, 200, 400]);
    expect(ff.chunkRequests(sha)).toBe(4);
    expect(results.find((r) => r.sha === sha)!.attempts).toBe(4);
  });

  it("gives up after `retries` retries and rejects with the last error", async () => {
    const { plan, ff } = await setup();
    const sha = plan.order[0]!.sha;
    ff.fail(sha, { kind: "network" }, { kind: "network" }, { kind: "network" }, { kind: "network" });
    const { done } = run(plan, ff, { retries: 3 });
    await expect(done).rejects.toMatchObject({ code: "chunk_network", sha });
    expect(ff.chunkRequests(sha)).toBe(4);
  });

  it("does not retry 4xx", async () => {
    const { plan, ff } = await setup();
    const sha = plan.order[0]!.sha;
    ff.fail(sha, { kind: "status", status: 404 });
    const { sleeps, done } = run(plan, ff);
    await expect(done).rejects.toMatchObject({ code: "chunk_http", status: 404 });
    expect(sleeps).toEqual([]);
    expect(ff.chunkRequests(sha)).toBe(1);
  });

  it("a corrupt or short network body fails verification and is retried", async () => {
    const { plan, ff } = await setup();
    const sha = plan.order[2]!.sha;
    ff.fail(sha, { kind: "corrupt" }, { kind: "short" });
    const { results, done } = run(plan, ff);
    await done;
    expect(ff.chunkRequests(sha)).toBe(3);
    expect(results.find((r) => r.sha === sha)!.attempts).toBe(3);
  });

  it("a failing chunk cancels the other workers and the loop rejects once", async () => {
    const { plan, ff } = await setup({ chunkSize: 4096 });
    ff.delayMs = 1;
    const sha = plan.order[3]!.sha;
    ff.fail(sha, { kind: "status", status: 403 });
    const { done } = run(plan, ff, { concurrency: 4 });
    await expect(done).rejects.toBeInstanceOf(ChunkError);
    expect(ff.chunkRequests()).toBeLessThan(plan.order.length);
  });

  it("honours AbortSignal before and during the load", async () => {
    const { plan, ff } = await setup({ chunkSize: 4096 });
    const pre = new AbortController(); pre.abort();
    await expect(run(plan, ff, { signal: pre.signal }).done).rejects.toBeInstanceOf(AbortedError);
    expect(ff.chunkRequests()).toBe(0);

    ff.delayMs = 5;
    const ac = new AbortController();
    const { results, done } = run(plan, ff, { signal: ac.signal, concurrency: 2 });
    setTimeout(() => ac.abort(), 8);
    await expect(done).rejects.toBeInstanceOf(AbortedError);
    expect(results.length).toBeLessThan(plan.order.length);
  });

  it("aborts a backoff sleep", async () => {
    const { plan, ff } = await setup();
    ff.fail(plan.order[0]!.sha, { kind: "network" });
    const ac = new AbortController();
    const p = fetchChunks(plan.order, { cdn: CDN, modelId: "m", store: null, fetch: ff.fetch, onChunk: () => {}, signal: ac.signal, backoffMs: 10_000 });
    setTimeout(() => ac.abort(), 5);
    await expect(p).rejects.toBeInstanceOf(AbortedError);
  });

  describe("with a per-site store", () => {
    it("serves from cache on the second run, verifying every cached chunk, and reports the source", async () => {
      const { plan, ff } = await setup();
      const { store } = makeStore({ cdn: CDN });
      const first = run(plan, ff, { store });
      await first.done;
      expect(first.results.every((r) => r.source === "network")).toBe(true);
      expect(await store.status()).toMatchObject({ chunks: plan.order.length });

      const ff2 = new FakeFetch("https://api.test", CDN, ff.chunks);
      const second = run(plan, ff2, { store });
      const stats = await second.done;
      expect(ff2.chunkRequests()).toBe(0);
      expect(second.results.every((r) => r.source === "per-site-cache" && r.attempts === 0)).toBe(true);
      expect(stats.verifyMs).toBeGreaterThanOrEqual(0);
    });

    it("evicts a corrupt cached chunk and refetches it once", async () => {
      const { plan, ff } = await setup();
      const { store, caches } = makeStore({ cdn: CDN });
      await run(plan, ff, { store }).done;
      // Corrupt one cached body in place.
      const victim = plan.order[1]!.sha;
      const cache = caches.caches.get("dianome-v1")!;
      const rec = cache.entries.get(`${CDN}/chunks/${victim}`)!;
      rec.body[10] = (rec.body[10]! + 1) & 0xff;

      const ff2 = new FakeFetch("https://api.test", CDN, ff.chunks);
      const { results, done } = run(plan, ff2, { store });
      const stats = await done;
      expect(stats.corruptEvicted).toBe(1);
      expect(ff2.chunkRequests()).toBe(1);
      expect(ff2.chunkRequests(victim)).toBe(1);
      const r = results.find((x) => x.sha === victim)!;
      expect(r.source).toBe("network");
      expect(results.filter((x) => x.source === "per-site-cache")).toHaveLength(plan.order.length - 1);
      // The refetched chunk was put back and is now valid.
      const again = await store.get(victim);
      expect(new Uint8Array(again!)).toEqual(ff.chunks.get(victim));
    });

    it("on QuotaExceededError with nothing evictable, switches to cache none and keeps streaming", async () => {
      const { plan, ff } = await setup();
      const total = plan.order.reduce((s, c) => s + c.bytes, 0);
      const { store, caches } = makeStore({ cdn: CDN, capacity: Math.floor(total / 2) });
      let disabled = 0;
      const { results, done } = run(plan, ff, { store, onCacheDisabled: () => disabled++ });
      const stats = await done;
      expect(results).toHaveLength(plan.order.length);
      expect(stats.cacheDisabled).toBe(true);
      expect(disabled).toBe(1);
      expect(caches.used).toBeLessThanOrEqual(Math.floor(total / 2));
      // Chunks of the model being loaded are never evicted to make room for its own later chunks.
      const listed = await store.list();
      expect(listed.every((m) => m.modelIds.includes(plan.modelId))).toBe(true);
    });
  });
});
