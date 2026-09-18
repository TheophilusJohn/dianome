// Diagnostics (not gates): where the gate-4 error comes from. Writes results/diag.json.
import { test } from "@playwright/test";
import { mkdirSync, writeFileSync } from "node:fs";
import type { Page } from "@playwright/test";

const results: Record<string, unknown> = {};
let page: Page;
const h = (name: string, ...args: unknown[]): Promise<any> => page.evaluate(([n, a]) => (window.harness as unknown as Record<string, (...x: unknown[]) => unknown>)[n as string]!(...(a as unknown[])), [name, args] as const);

test.beforeAll(async ({ browser }) => { page = await browser.newPage(); page.on("pageerror", (e) => console.log(`[pageerror] ${e.message}`)); await page.goto("/"); await page.waitForFunction(() => Boolean(window.harness)); });
test.afterAll(() => { mkdirSync("results", { recursive: true }); writeFileSync("results/diag.json", JSON.stringify(results, null, 2)); });

test("isolated per-block error and attention/silu variants", async () => {
  test.setTimeout(1_200_000);
  const iso = await h("gate4Isolated");
  results.isolated = iso;
  for (const b of iso) console.log(`  isolated block ${String(b.block).padStart(2)} maxAbs ${b.cmp.maxAbs.toExponential(3)} relRms ${b.cmp.relRms.toExponential(3)} (ref ${b.cmp.refAtMax} got ${b.cmp.gotAtMax})`);
  for (const blk of [20, 21, 22, 23]) {
    const st = await h("stagesFor", blk);
    results[`stages_${blk}`] = st;
    for (const s of st) console.log(`  block ${blk} ${s.stage.padEnd(12)} maxAbs ${s.cmp.maxAbs.toExponential(3)} relRms ${s.cmp.relRms.toExponential(2)} (ref ${s.cmp.refAtMax} got ${s.cmp.gotAtMax})`);
  }
  const variants: Record<string, unknown> = {};
  for (const v of [{ attnFlags: 3 }, { attnFlags: 5 }, { attnFlags: 7 }, { siluMode: 1 }, { attnFlags: 3, siluMode: 1 }]) {
    const key = JSON.stringify(v);
    const b = await h("gate4", "fp16", v);
    variants[key] = b;
    const worst = b.reduce((a: any, c: any) => (c.cmp.maxAbs > a.cmp.maxAbs ? c : a));
    console.log(`  variant ${key}: block 0 maxAbs ${b[0].cmp.maxAbs.toExponential(3)} block 23 maxAbs ${b[23].cmp.maxAbs.toExponential(3)} relRms ${b[23].cmp.relRms.toExponential(3)}; worst block ${worst.block} ${worst.cmp.maxAbs.toExponential(3)}`);
    const st0 = await h("stagesFor", 0, v);
    console.log(`     block 0 attn maxAbs ${st0.find((s: any) => s.stage === "attn").cmp.maxAbs.toExponential(3)}  down maxAbs ${st0.find((s: any) => s.stage === "down").cmp.maxAbs.toExponential(3)}`);
    variants[key + ":stages0"] = st0;
  }
  results.variants = variants;
});
