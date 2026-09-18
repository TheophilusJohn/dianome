// Bench numbers for docs/phase-5a-notes.md: prefill and decode tok/s vs N for fp16/q8/q4, both T=1 kernels
// (and the tiled kernel forced at T=1), time to first block, GPU bytes. Three runs per row; the median run
// (by decode ms) is reported and all three are kept. Writes results/bench.json.
import { test } from "@playwright/test";
import { mkdirSync, writeFileSync } from "node:fs";
import type { Page } from "@playwright/test";

const results: unknown[] = [];
let page: Page;
const h = (name: string, ...args: unknown[]): Promise<any> => page.evaluate(([n, a]) => (window.harness as unknown as Record<string, (...x: unknown[]) => unknown>)[n as string]!(...(a as unknown[])), [name, args] as const);

test.beforeAll(async ({ browser }) => { page = await browser.newPage(); page.on("pageerror", (e) => console.log(`[pageerror] ${e.message}`)); await page.goto("/"); await page.waitForFunction(() => Boolean(window.harness)); });
test.afterAll(() => { mkdirSync("results", { recursive: true }); writeFileSync("results/bench.json", JSON.stringify({ info: null, runs: results }, null, 2)); });

test("prefill/decode tok/s vs N, variants, T=1 kernels", async () => {
  test.setTimeout(1_800_000);
  for (const variant of ["fp16", "q8", "q4"]) {
    for (const N of [1, 8, 16, 24]) {
      for (const matvec of ["seq", "lanes", "tiled"]) {
        const runs: any[] = [];
        for (let i = 0; i < 3; i++) runs.push(await h("bench", variant, N, 32, matvec));
        const med = (k: string) => { const s = runs.map((r) => r[k]).sort((a, b) => a - b); return s[1]; };
        const r = { ...runs[0], prefillTokPerS: med("prefillTokPerS"), prefillMs: med("prefillMs"), decodeTokPerS: med("decodeTokPerS"), decodeMsMedian: med("decodeMsMedian"), exportMs: med("exportMs"), runs };
        results.push({ variant, N, matvec, ...r });
        console.log(`  ${variant.padEnd(4)} N=${String(N).padStart(2)} ${matvec.padEnd(5)} prefill ${r.prefillTokPerS.toFixed(0).padStart(5)} tok/s (${r.prefillMs.toFixed(1)} ms/32) decode ${r.decodeTokPerS.toFixed(1).padStart(6)} tok/s (${r.decodeMsMedian.toFixed(2)} ms) export ${r.exportMs.toFixed(2)} ms  gpu ${(r.stats.gpuBytes / 1e6).toFixed(0)} MB  ttfb ${r.load?.timeToFirstBlockMs?.toFixed(0)} ms load ${r.load?.ms?.toFixed(0)} ms  [decode ms runs: ${runs.map((x) => x.decodeMsMedian.toFixed(2)).join(" ")}]`);
      }
    }
  }
});
