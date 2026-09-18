// /dashboard renders totals, per-hour bars, modes, the cost estimate and the key list from a stubbed usage response
// (page.route), then a stubbed DELETE revokes a key. No Worker is involved: the shape is the one dashboard-api.ts returns.
import { test, expect } from "@playwright/test";

const LOCAL = "?api=http://127.0.0.1:8797&cdn=http://127.0.0.1:8797";
const KEY = "dk_live_" + "e2e".repeat(10) + "ab";
const now = new Date(Date.UTC(2026, 8, 18, 15, 0));
const hour = (h: number) => new Date(now.getTime() - h * 3_600_000).toISOString();
const bucket = (h: number, key_id: string, sessions: number, client: number, server: number, busy: number, modes: Partial<{ local: number; split: number; server: number }>) =>
  ({ hour: hour(h), key_id, sessions, client_layer_tokens: client, server_layer_tokens: server, new_tokens: client / 24 + server / 25, server_busy_ms: busy, bytes_served: 0, modes: { local: 0, split: 0, server: 0, ...modes } });
const usage = {
  key_id: "k_0123456789abcdef", owner: "k_0123456789abcdef", keys: ["k_0123456789abcdef", "k_fedcba9876543210"], hours: 168, since: hour(167), until: hour(-1),
  buckets: [
    bucket(30, "k_0123456789abcdef", 2, 0, 500, 1200.5, { server: 2 }),
    bucket(2, "k_0123456789abcdef", 3, 4608, 0, 0, { local: 3 }),
    bucket(2, "k_fedcba9876543210", 1, 768, 832, 710.2, { split: 1 }),
    { ...bucket(0, "k_0123456789abcdef", 1, 0, 0, 0, {}), sessions: 0, bytes_served: 323893760 },
  ],
  totals: { sessions: 6, client_layer_tokens: 5376, server_layer_tokens: 1332, new_tokens: 245, server_busy_ms: 1910.7, bytes_served: 323893760, modes: { local: 3, split: 1, server: 2 } },
  last_24h: { sessions: 4, client_layer_tokens: 5376, server_layer_tokens: 832, new_tokens: 225, server_busy_ms: 710.2, bytes_served: 323893760, modes: { local: 3, split: 1, server: 0 } },
  cost_estimate: { estimate: true, basis: "server_busy_ms / 3.6e6 h × usd_per_hour", usd: 0.00026, usd_per_hour: 0.49, gpu: "NVIDIA L4 24GB", source: "RunPod", retrieved: "2026-09-17" },
};
const keys = { owner: "k_0123456789abcdef", keys: [
  { id: "k_0123456789abcdef", owner: "k_0123456789abcdef", created: "2026-09-18T10:00:00.000Z", revoked: null, plan: "free", current: true },
  { id: "k_fedcba9876543210", owner: "k_0123456789abcdef", created: "2026-09-18T11:00:00.000Z", revoked: null, plan: "free", current: false },
] };

test("dashboard: sign in with a key, totals and bars from a stubbed usage response, revoke a sibling key", async ({ page }) => {
  page.on("pageerror", (e) => console.log(`[pageerror] ${e.message}`));
  const seen: { url: string; method: string; auth: string | null }[] = [];
  let revoked = false;
  await page.route("**/v1/me/**", async (route) => {
    const req = route.request();
    seen.push({ url: req.url(), method: req.method(), auth: req.headers()["authorization"] ?? null });
    const u = new URL(req.url());
    if (u.pathname === "/v1/me/usage") return route.fulfill({ json: usage });
    if (u.pathname === "/v1/me/keys" && req.method() === "GET") return route.fulfill({ json: revoked ? { ...keys, keys: [keys.keys[0]!, { ...keys.keys[1]!, revoked: "2026-09-18T12:00:00.000Z" }] } : keys });
    if (u.pathname === "/v1/me/keys/k_fedcba9876543210" && req.method() === "DELETE") { revoked = true; return route.fulfill({ json: { ...keys.keys[1], revoked: "2026-09-18T12:00:00.000Z", current: false } }); }
    return route.fulfill({ status: 404, json: { error: "not_found" } });
  });
  await page.goto("/dashboard/" + LOCAL);
  await expect(page.locator("h1")).toHaveText("Dashboard");
  await expect(page.locator("#app")).toBeHidden();
  await page.fill("#key", "nope");
  await page.click("#signin-btn");
  await expect(page.locator("#signin-status")).toContainText("not a dk_live_ key");
  await page.fill("#key", KEY);
  await page.click("#signin-btn");
  await expect(page.locator("#app")).toBeVisible();
  await expect(page.locator("#status")).toContainText("loaded");
  expect(seen.every((s) => s.auth === `Bearer ${KEY}`)).toBe(true);
  expect(seen.some((s) => s.url.includes("/v1/me/usage?hours=168"))).toBe(true);
  // totals: 24 h and 7 d columns
  const cell = (label: string, i: number) => page.locator("#totals table tr", { hasText: label }).first().locator("td").nth(i);
  await expect(cell("sessions", 1)).toHaveText("4");
  await expect(cell("sessions", 2)).toHaveText("6");
  await expect(cell("client layer-tokens", 1)).toHaveText("5,376");
  await expect(cell("client layer-tokens", 2)).toHaveText("5,376");
  await expect(cell("server layer-tokens", 1)).toHaveText("832");
  await expect(cell("server layer-tokens", 2)).toHaveText("1,332");
  await expect(cell("server busy time", 1)).toHaveText("710 ms");
  await expect(cell("server busy time", 2)).toHaveText("1.9 s");
  await expect(cell("bytes served", 2)).toHaveText("308.9 MiB");
  await expect(cell("server cost estimate", 1)).toHaveText("$0.00010");
  await expect(cell("server cost estimate", 2)).toHaveText("$0.00026");
  // per-hour bars: 3 distinct hours (two keys share one), client and server widths proportional
  const rows = page.locator("#hours .hour");
  await expect(rows).toHaveCount(3);
  await expect(rows.nth(1)).toContainText("5,376 / 832 · 4 sessions");
  const widths = await rows.nth(1).locator(".bar").evaluateAll((els) => els.map((e) => parseFloat((e as HTMLElement).style.width)));
  expect(widths[0]).toBeGreaterThan(widths[1]!);
  expect(Math.abs(widths[0]! / widths[1]! - 5376 / 832)).toBeLessThan(0.05);
  await expect(page.locator("#modes")).toContainText("local");
  await expect(page.locator("#modes tr", { hasText: "local" })).toContainText("50 %");
  await expect(page.locator("#cost")).toContainText("$0.00026");
  await expect(page.locator("#cost")).toContainText("$0.49/h on NVIDIA L4 24GB");
  await expect(page.locator("#usage-note")).toContainText("keys counted: k_0123456789abcdef, k_fedcba9876543210");
  // keys: the current one marked, the sibling revocable
  await expect(page.locator("#keys tr", { hasText: "k_0123456789abcdef" })).toContainText("(this key)");
  await page.locator("#keys button.revoke[data-id='k_fedcba9876543210']").click();
  await expect(page.locator("#keys tr", { hasText: "k_fedcba9876543210" })).toContainText("revoked 2026-09-18 12:00:00");
  expect(seen.some((s) => s.method === "DELETE" && s.url.endsWith("/v1/me/keys/k_fedcba9876543210"))).toBe(true);
  // the key never appears in the page text and survives a reload through sessionStorage
  expect(await page.locator("body").innerText()).not.toContain(KEY);
  await page.reload();
  await expect(page.locator("#app")).toBeVisible();
  await expect(page.locator("#status")).toContainText("loaded");
  await page.click("#signout");
  await expect(page.locator("#app")).toBeHidden();
  expect(await page.evaluate(() => sessionStorage.getItem("dianome:apiKey"))).toBeNull();
});

test("dashboard: a refused key signs out with the API's reason", async ({ page }) => {
  await page.route("**/v1/me/**", (route) => route.fulfill({ status: 403, json: { error: "key_revoked", detail: "key k_0123456789abcdef was revoked" } }));
  await page.goto("/dashboard/" + LOCAL);
  await page.fill("#key", KEY);
  await page.click("#signin-btn");
  await expect(page.locator("#signin-status")).toContainText("403 key_revoked");
  await expect(page.locator("#app")).toBeHidden();
});
