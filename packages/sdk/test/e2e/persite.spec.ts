import { expect, test, type Page } from "@playwright/test";

// Per-site cache path on every engine: first load from the network, second load entirely from the Cache API with
// zero chunk requests, and a corrupt cached chunk evicted and refetched (exactly one chunk request).

interface LoadOut {
  ms: number;
  summary: { bytes: number; chunks: number; source: string; cacheHits: number; cacheMode: string; sources: string[]; telemetryStatus: number | null; cacheDisabled: boolean };
  report: { schema: number; source: string; cache_hits: number; browser: string; cache_mode?: string } | null;
  groups: { name: string; entries: number; copied: number }[];
  aligned: boolean;
  tiedEqual: boolean;
  status: { mode: string; chunks: number; usageBytes: number };
}

async function chunkRequests(page: Page, run: () => Promise<LoadOut>): Promise<{ out: LoadOut; requests: number; serverHits: number }> {
  await page.request.get("/__reset");
  let requests = 0;
  const onReq = (r: { url(): string }) => { if (r.url().includes("/chunks/")) requests++; };
  page.on("request", onReq);
  const out = await run();
  page.off("request", onReq);
  const serverHits = (await (await page.request.get("/__chunk-hits")).json()).chunkHits as number;
  return { out, requests, serverHits };
}

test.beforeEach(async ({ page }) => {
  await page.goto("/");
  await page.evaluate(() => (window as unknown as { clearAll(): Promise<void> }).clearAll());
});

test("first load: network; second load: per-site cache with zero chunk requests; report matches", async ({ page, browserName }) => {
  const load = () => page.evaluate(() => (window as unknown as { load(): Promise<LoadOut> }).load());
  const first = await chunkRequests(page, load);
  expect(first.out.summary.source).toBe("network");
  expect(first.out.summary.cacheMode).toBe("per-site");
  expect(first.out.summary.cacheHits).toBe(0);
  expect(first.requests).toBe(first.out.summary.chunks);
  expect(first.serverHits).toBe(first.out.summary.chunks);
  expect(first.out.aligned).toBe(true);
  expect(first.out.tiedEqual).toBe(true);
  expect(first.out.groups.map((g) => g.name)).toEqual(["embed", ...Array.from({ length: 8 }, (_, i) => `layer.${i}`), "final_norm", "lm_head"]);
  expect(first.out.groups.some((g) => g.copied > 0)).toBe(true); // multi-chunk entries exist in this store
  expect(first.out.status.chunks).toBe(first.out.summary.chunks);
  expect(first.out.summary.telemetryStatus).toBe(202);
  const reports = (await (await page.request.get("/__telemetry")).json()) as { source: string; browser: string; schema: number; cache_mode: string }[];
  expect(reports).toHaveLength(1);
  expect(reports[0]).toMatchObject({ schema: 2, source: "network", cache_mode: "per-site", browser: { chromium: "chrome", firefox: "firefox", webkit: "safari" }[browserName] });

  const second = await chunkRequests(page, load);
  expect(second.requests).toBe(0);
  expect(second.serverHits).toBe(0);
  expect(second.out.summary.source).toBe("per-site-cache");
  expect(second.out.summary.cacheHits).toBe(second.out.summary.chunks);
  expect(second.out.tiedEqual).toBe(true);
  const reports2 = (await (await page.request.get("/__telemetry")).json()) as { source: string; cache_hits: number }[];
  expect(reports2[0]).toMatchObject({ source: "per-site-cache", cache_hits: second.out.summary.chunks });

  // Cached across a navigation too (same origin, fresh document). Playwright's ephemeral WebKit context drops
  // CacheStorage on navigation (memory-backed); the persistent-context test below covers WebKit.
  let third_ms: number | null = null;
  if (browserName !== "webkit") {
    await page.reload();
    const third = await chunkRequests(page, load);
    expect(third.requests).toBe(0);
    expect(third.out.summary.source).toBe("per-site-cache");
    third_ms = Math.round(third.out.ms);
  }
  test.info().annotations.push({ type: "timing", description: JSON.stringify({ browser: browserName, first_ms: Math.round(first.out.ms), second_ms: Math.round(second.out.ms), third_ms, bytes: first.out.summary.bytes, chunks: first.out.summary.chunks }) });
});

test("webkit: the per-site cache survives a navigation in a persistent context", async ({ browserName, playwright, baseURL }) => {
  test.skip(browserName !== "webkit", "chromium and firefox are covered by the reload step above");
  const dir = `test-results/webkit-profile-${Date.now()}`;
  const context = await playwright.webkit.launchPersistentContext(dir);
  try {
    const page = await context.newPage();
    await page.goto(`${baseURL}/`);
    await page.evaluate(() => (window as unknown as { clearAll(): Promise<void> }).clearAll());
    const load = () => page.evaluate(() => (window as unknown as { load(): Promise<LoadOut> }).load());
    const first = await chunkRequests(page, load);
    expect(first.out.summary.source).toBe("network");
    await page.goto(`${baseURL}/`);
    const second = await chunkRequests(page, load);
    expect(second.requests).toBe(0);
    expect(second.out.summary.source).toBe("per-site-cache");
    test.info().annotations.push({ type: "timing", description: JSON.stringify({ browser: "webkit-persistent", first_ms: Math.round(first.out.ms), second_ms: Math.round(second.out.ms), bytes: first.out.summary.bytes, chunks: first.out.summary.chunks }) });
  } finally {
    await context.close();
  }
});

test("a corrupt cached chunk is evicted and refetched exactly once; the rest stay cache hits", async ({ page }) => {
  const load = () => page.evaluate(() => (window as unknown as { load(): Promise<LoadOut> }).load());
  await chunkRequests(page, load);
  const shas = (await page.evaluate(() => (window as unknown as { cachedShas(): Promise<string[]> }).cachedShas())) as string[];
  expect(shas.length).toBeGreaterThan(2);
  const victim = shas[Math.floor(shas.length / 2)]!;
  expect(await page.evaluate((s) => (window as unknown as { corrupt(s: string): Promise<boolean> }).corrupt(s), victim)).toBe(true);

  const again = await chunkRequests(page, load);
  expect(again.requests).toBe(1);
  // The refetch may be answered by the browser's HTTP cache (the chunk is immutable), so the server sees 0 or 1.
  expect(again.serverHits).toBeLessThanOrEqual(1);
  expect(again.out.summary.source).toBe("mixed");
  expect(again.out.summary.cacheHits).toBe(again.out.summary.chunks - 1);
  expect(again.out.summary.sources.filter((s) => s === "network")).toHaveLength(1);
  expect(again.out.tiedEqual).toBe(true);

  // The refetched chunk is valid again: a further load is all cache.
  const clean = await chunkRequests(page, load);
  expect(clean.requests).toBe(0);
  expect(clean.out.summary.source).toBe("per-site-cache");
});

test("cache: none never touches the Cache API", async ({ page }) => {
  const load = () => page.evaluate(() => (window as unknown as { load(o: unknown): Promise<LoadOut> }).load({ cache: "none" }));
  const a = await chunkRequests(page, load);
  const b = await chunkRequests(page, load);
  expect(a.requests).toBe(a.out.summary.chunks);
  expect(b.requests).toBe(b.out.summary.chunks);
  expect(b.out.summary.cacheMode).toBe("none");
  expect(await page.evaluate(() => (window as unknown as { cachedShas(): Promise<string[]> }).cachedShas())).toEqual([]);
});
