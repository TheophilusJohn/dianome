// /summarize end to end with the 0.5B model against local servers: a mode is chosen (and said), the summary is
// non-empty, a schema-3 telemetry report reaches the Worker. Results go to results/summarize-e2e.json for the notes.
import { test, expect } from "@playwright/test";
import { mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { resolve } from "node:path";

const MODEL = "qwen2.5-0.5b-instruct";
const LOCAL = "?api=http://127.0.0.1:8797&cdn=http://127.0.0.1:8797";
const sample = readFileSync(resolve(import.meta.dirname, "sample.txt"), "utf8");
const telemetry = async () => (await (await fetch("http://127.0.0.1:8797/__telemetry")).json()) as { status: number; body: Record<string, unknown> }[];

test("home, stats and write-up pages render", async ({ page }) => {
  await page.goto("/" + LOCAL);
  await expect(page.locator("h1")).toHaveText("dianome");
  await expect(page.locator("nav a")).toHaveCount(4);
  await page.goto("/stats/" + LOCAL);
  await expect(page.locator("#loads-status")).toContainText(/loaded|failed/, { timeout: 30_000 });
  await expect(page.locator("#sessions-status")).toContainText(/loaded|failed/, { timeout: 30_000 });
  await page.goto("/writeup/" + LOCAL);
  await expect(page.locator("article h1")).toHaveCount(1);
});

test("summarise a document with 0.5B: mode chosen and explained, summary non-empty, telemetry sent", async ({ page }) => {
  page.on("pageerror", (e) => console.log(`[pageerror] ${e.message}`));
  page.on("console", (m) => { if (m.type() === "error") console.log(`[page error] ${m.text()}`); });
  const before = (await telemetry()).length;
  await page.goto(`/summarize/${LOCAL}&model=${MODEL}`);
  await expect(page.locator("#model")).toHaveValue(MODEL);
  await expect(page.locator("#run")).toBeEnabled();
  await page.fill("#doc", sample);
  await page.click("#run");
  const status = page.locator("#status");
  await expect(status).toContainText(/mode: (local|server|split)/, { timeout: 120_000 });
  await expect(status).toContainText("why:");
  await expect(status).toContainText("cached:");
  await expect(status).toContainText("done:", { timeout: 480_000 });
  const text = (await page.locator("#output").innerText()).trim();
  console.log(`  summary (${text.length} chars): ${text.slice(0, 160).replace(/\n/g, " ")}…`);
  expect(text.length).toBeGreaterThan(20);
  const st = await status.innerText();
  expect(st).toMatch(/tok\/s/);
  expect(st).toMatch(/cost: server share \d+ %/);
  expect(st).toMatch(/privacy/);
  expect(st).toMatch(/telemetry: session report sent \(HTTP 202\)/);
  const result = JSON.parse(await page.locator("#result-json").innerText()) as Record<string, unknown>;
  expect(["local", "split", "server"]).toContain(result.mode);
  const tel = (await telemetry()).slice(before);
  const mine = tel.find((t) => t.body.schema === 3 && t.body.model === MODEL);
  expect(mine?.status).toBe(202);
  expect(mine?.body.mode).toBe(result.mode);
  expect(mine?.body.N).toBe(result.N);
  console.log(`  ${String(result.mode)} N=${String(result.N)} ${Number(result.tokPerS).toFixed(1)} tok/s, ${String(result.promptTokens)} prompt tokens, reasons: ${(result.reasons as string[]).join("; ")}`);
  mkdirSync("results", { recursive: true });
  writeFileSync("results/summarize-e2e.json", JSON.stringify({ model: MODEL, status: st, result, telemetry: mine, summaryChars: text.length, updated: new Date().toISOString() }, null, 2));
});

test("compare runs the other feasible modes side by side", async ({ page }) => {
  page.on("pageerror", (e) => console.log(`[pageerror] ${e.message}`));
  await page.goto(`/summarize/${LOCAL}&model=${MODEL}`);
  await expect(page.locator("#run")).toBeEnabled();
  await page.fill("#doc", sample.split("\n\n").slice(0, 2).join("\n\n") || sample);
  await page.click("#run");
  await expect(page.locator("#status")).toContainText("done:", { timeout: 480_000 });
  await page.click("#compare");
  const cards = page.locator("#compare-out .card");
  await expect(cards).toHaveCount(3);
  await expect(cards.nth(0).locator(".meta")).toContainText(/tok\/s|not feasible/, { timeout: 480_000 });
  await expect(cards.nth(1).locator(".meta")).toContainText(/tok\/s|not feasible/, { timeout: 480_000 });
  await expect(cards.nth(2).locator(".meta")).toContainText(/tok\/s|not feasible/, { timeout: 480_000 });
  await expect(page.locator("#compare")).toBeEnabled();
  const out: { title: string; meta: string; chars: string }[] = [];
  for (let i = 0; i < 3; i++) {
    const title = await cards.nth(i).locator("h3").innerText(), meta = await cards.nth(i).locator(".meta").innerText(), text = (await cards.nth(i).locator(".output").innerText()).trim();
    out.push({ title, meta, chars: String(text.length) });
    console.log(`  ${title}: ${meta.slice(0, 120)} (${text.length} chars)`);
    if (!/not feasible/.test(meta)) expect(text.length).toBeGreaterThan(20);
  }
  expect(out.filter((c) => /tok\/s/.test(c.meta)).length).toBeGreaterThanOrEqual(2);
  const prev = JSON.parse(readFileSync("results/summarize-e2e.json", "utf8")) as Record<string, unknown>;
  writeFileSync("results/summarize-e2e.json", JSON.stringify({ ...prev, compare: out }, null, 2));
});
