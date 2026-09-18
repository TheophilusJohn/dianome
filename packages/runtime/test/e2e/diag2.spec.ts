// Diagnostics for gate 5a (decode vs prefill) and the round16-off gate-4 curve. Writes results/diag2.json.
import { test } from "@playwright/test";
import { mkdirSync, writeFileSync } from "node:fs";
import type { Page } from "@playwright/test";

const results: Record<string, unknown> = {};
let page: Page;
const h = (name: string, ...args: unknown[]): Promise<any> => page.evaluate(([n, a]) => (window.harness as unknown as Record<string, (...x: unknown[]) => unknown>)[n as string]!(...(a as unknown[])), [name, args] as const);

test.beforeAll(async ({ browser }) => { page = await browser.newPage(); page.on("pageerror", (e) => console.log(`[pageerror] ${e.message}`)); await page.goto("/"); await page.waitForFunction(() => Boolean(window.harness)); });
test.afterAll(() => { mkdirSync("results", { recursive: true }); writeFileSync("results/diag2.json", JSON.stringify(results, null, 2)); });

test("gate 5a under each T=1 matmul path; gate 4 with round16 off", async () => {
  test.setTimeout(1_200_000);
  for (const v of [{ forceTiled: true }, { matvec: "seq" }, { matvec: "lanes" }]) {
    const r = await h("gate5a", v);
    results[`gate5a:${JSON.stringify(v)}`] = r;
    const worstOwn = Math.max(...r.map((b: any) => b.vsOwnPrefill.maxAbs));
    const worstFix = Math.max(...r.map((b: any) => b.vsFixture.maxAbs));
    console.log(`  gate5a ${JSON.stringify(v)}: worst vs own prefill ${worstOwn.toExponential(3)}, worst vs fixture ${worstFix.toExponential(3)}; per block own: ${r.map((b: any) => b.vsOwnPrefill.maxAbs.toExponential(1)).join(" ")}`);
  }
  const off = await h("gate4", "fp16", { round16: false });
  results.gate4Round16Off = off;
  for (const b of off) console.log(`  [round16 off] block ${String(b.block).padStart(2)} maxAbs ${b.cmp.maxAbs.toExponential(3)} relRms ${b.cmp.relRms.toExponential(3)}`);
  const offIso = await h("gate4Isolated", { round16: false });
  results.isolatedRound16Off = offIso;
  console.log(`  [round16 off, isolated] worst block ${offIso.reduce((a: any, c: any) => (c.cmp.maxAbs > a.cmp.maxAbs ? c : a)).block} maxAbs ${Math.max(...offIso.map((b: any) => b.cmp.maxAbs)).toExponential(3)}`);
  const unit = await h("gate2Unit");
  console.log(`  seq matvec unit: ${unit.filter((u: any) => u.path === "matvec.seq").map((u: any) => `${u.name} maxRel ${u.cmp.maxRel.toExponential(2)}`).join(" | ")}`);
  const g5b = await h("gate5b", { matvec: "seq" });
  results.gate5bSeq = g5b;
  for (const s of g5b) console.log(`  [seq] step ${String(s.step).padStart(2)} worst boundary ${s.perBoundary.indexOf(s.worst)} maxAbs ${s.worst.maxAbs.toExponential(3)} relRms ${s.worst.relRms.toExponential(3)}  boundary 24 maxAbs ${s.perBoundary[24].maxAbs.toExponential(3)} relRms ${s.perBoundary[24].relRms.toExponential(3)}`);
  const g5bl = await h("gate5b", { matvec: "lanes" });
  results.gate5bLanes = g5bl;
  for (const s of g5bl) console.log(`  [lanes] step ${String(s.step).padStart(2)} worst boundary ${s.perBoundary.indexOf(s.worst)} maxAbs ${s.worst.maxAbs.toExponential(3)} relRms ${s.worst.relRms.toExponential(3)}  boundary 24 maxAbs ${s.perBoundary[24].maxAbs.toExponential(3)} relRms ${s.perBoundary[24].relRms.toExponential(3)}`);
});
