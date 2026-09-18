// /dashboard (Phase 6): paste an API key (kept in sessionStorage only), see usage (GET /v1/me/usage) and keys
// (GET/POST/DELETE /v1/me/keys). Plain DOM like the rest of the site. Every number is what the API returned.
import { $, API, el, fmtBytes, nav, table } from "./shared";

nav("/dashboard/");

interface ModeCounts { local: number; split: number; server: number }
interface Counters { sessions: number; client_layer_tokens: number; server_layer_tokens: number; new_tokens: number; server_busy_ms: number; bytes_served: number; modes: ModeCounts }
interface Bucket extends Counters { hour: string; key_id: string }
interface CostEstimate { estimate: true; basis: string; usd: number | null; usd_per_hour: number | null; gpu: string; source: string; retrieved: string }
export interface Usage { key_id: string; owner: string; keys: string[]; hours: number; since: string; until: string; buckets: Bucket[]; totals: Counters; last_24h: Counters; cost_estimate: CostEstimate }
interface KeyInfo { id: string; owner: string; created: string; revoked: string | null; plan: string; current: boolean }
interface KeyList { owner: string; keys: KeyInfo[] }
interface NewKey { key: string; id: string; owner: string; created: string }

const STORAGE = "dianome:apiKey";
const KEY_RE = /^dk_live_[A-Za-z0-9_-]{32}$/;

function storedKey(): string | null {
  try { return sessionStorage.getItem(STORAGE); } catch { return null; }
}
function storeKey(k: string | null): void {
  try { if (k) sessionStorage.setItem(STORAGE, k); else sessionStorage.removeItem(STORAGE); } catch { /* a private window without storage: the key lives in memory for this page load */ }
}

let key: string | null = storedKey();

async function api<T>(path: string, init: RequestInit = {}): Promise<T> {
  const headers = new Headers(init.headers);
  if (key) headers.set("Authorization", `Bearer ${key}`);
  const res = await fetch(`${API}${path}`, { ...init, headers, cache: "no-store" });
  if (!res.ok) {
    let detail = `${res.status} ${res.statusText}`;
    try { const j = (await res.json()) as { error?: string; detail?: string }; detail = `${res.status} ${j.error ?? ""}${j.detail ? `: ${j.detail}` : ""}`; } catch { /* keep the status line */ }
    throw new Error(detail);
  }
  return (await res.json()) as T;
}

const n = (x: number): string => x.toLocaleString();
const ms = (x: number): string => (x >= 60_000 ? `${(x / 60_000).toFixed(1)} min` : x >= 1000 ? `${(x / 1000).toFixed(1)} s` : `${x.toFixed(0)} ms`);
const usd = (x: number | null): string => (x === null ? "no rate stated" : `$${x.toFixed(x < 0.01 ? 5 : 3)}`);
const hourLabel = (iso: string): string => iso.slice(5, 13).replace("T", " ") + "h";

function costOf(c: Counters, rate: CostEstimate): number | null {
  return rate.usd_per_hour === null ? null : (c.server_busy_ms / 3_600_000) * rate.usd_per_hour;
}

export function renderUsage(u: Usage): void {
  $("usage-note").textContent = `window ${u.hours} h from ${u.since} to ${u.until} · keys counted: ${u.keys.join(", ")} · ${u.buckets.length} hour bucket${u.buckets.length === 1 ? "" : "s"}`;
  const rows: (string | number)[][] = [
    ["sessions", n(u.last_24h.sessions), n(u.totals.sessions)],
    ["client layer-tokens", n(u.last_24h.client_layer_tokens), n(u.totals.client_layer_tokens)],
    ["server layer-tokens", n(u.last_24h.server_layer_tokens), n(u.totals.server_layer_tokens)],
    ["new tokens", n(u.last_24h.new_tokens), n(u.totals.new_tokens)],
    ["server busy time", ms(u.last_24h.server_busy_ms), ms(u.totals.server_busy_ms)],
    ["bytes served (loads reporting this key)", fmtBytes(u.last_24h.bytes_served), fmtBytes(u.totals.bytes_served)],
    ["server cost estimate", usd(costOf(u.last_24h, u.cost_estimate)), usd(u.cost_estimate.usd)],
  ];
  $("totals").replaceChildren(table([{ label: "" }, { label: "last 24 h", num: true }, { label: `last ${u.hours} h`, num: true }], rows, "no usage yet"));

  // per-hour stacked bars (buckets of several keys in one hour are summed)
  const byHour = new Map<string, { client: number; server: number; sessions: number }>();
  for (const b of u.buckets) {
    const h = byHour.get(b.hour) ?? { client: 0, server: 0, sessions: 0 };
    h.client += b.client_layer_tokens; h.server += b.server_layer_tokens; h.sessions += b.sessions;
    byHour.set(b.hour, h);
  }
  const hours = $("hours"); hours.replaceChildren();
  const max = Math.max(1, ...[...byHour.values()].map((h) => h.client + h.server));
  if (byHour.size === 0) hours.append(el("p", "no sessions in this window", "empty"));
  for (const [hour, h] of [...byHour.entries()].sort((a, b) => (a[0] < b[0] ? -1 : 1))) {
    const row = el("div"); row.className = "hour";
    row.append(el("span", hourLabel(hour), "label"));
    const client = el("span", undefined, "bar client"); client.style.width = `${(400 * h.client) / max}px`; client.title = `client ${n(h.client)}`;
    const server = el("span", undefined, "bar server"); server.style.width = `${(400 * h.server) / max}px`; server.title = `server ${n(h.server)}`;
    row.append(client, server, el("span", `${n(h.client)} / ${n(h.server)} · ${h.sessions} session${h.sessions === 1 ? "" : "s"}`, "small"));
    hours.append(row);
  }

  const m = u.totals.modes;
  $("modes").replaceChildren(table([{ label: "mode" }, { label: "sessions", num: true }, { label: "share", num: true }],
    (["local", "split", "server"] as const).map((k) => [k, m[k], u.totals.sessions ? `${((100 * m[k]) / u.totals.sessions).toFixed(0)} %` : "–"]), "no sessions"));

  const c = u.cost_estimate;
  const cost = $("cost"); cost.replaceChildren();
  cost.append(el("p", c.usd_per_hour === null
    ? `No GPU rate is stated (${c.source || "rates.json"} has none), so only the server busy time above is shown.`
    : `Estimate: ${usd(c.usd)} for ${ms(u.totals.server_busy_ms)} of server busy time at $${c.usd_per_hour}/h on ${c.gpu} (rate from ${c.source}, retrieved ${c.retrieved}).`));
  cost.append(el("p", `How it is computed: ${c.basis}`, "small"));
}

export function renderKeys(list: KeyList): void {
  const box = $("keys"); box.replaceChildren();
  if (list.keys.length === 0) { box.append(el("p", "no keys", "empty")); return; }
  const t = el("table");
  const head = el("tr");
  for (const h of ["id", "created", "status", ""]) head.append(el("th", h));
  const thead = el("thead"); thead.append(head); t.append(thead);
  const body = t.appendChild(el("tbody"));
  for (const k of list.keys) {
    const tr = el("tr");
    tr.append(el("td", k.id + (k.current ? " (this key)" : "")), el("td", k.created.replace("T", " ").slice(0, 19)));
    tr.append(el("td", k.revoked ? `revoked ${k.revoked.replace("T", " ").slice(0, 19)}` : "active"));
    const td = el("td");
    if (!k.revoked) {
      const b = el("button", "Revoke"); b.className = "revoke"; b.dataset.id = k.id;
      b.onclick = () => void revoke(k);
      td.append(b);
    }
    tr.append(td);
    body.append(tr);
  }
  box.append(t);
}

function say(text: string, error = false): void {
  const s = $("status"); s.textContent = text; s.className = error ? "err" : "small";
}

async function refresh(): Promise<void> {
  if (!key) return;
  say(`loading ${API}/v1/me/usage …`);
  try {
    const [u, k] = await Promise.all([api<Usage>("/v1/me/usage?hours=168"), api<KeyList>("/v1/me/keys")]);
    $("who").textContent = `key ${u.key_id} · owner ${u.owner}`;
    renderUsage(u);
    renderKeys(k);
    say(`loaded ${new Date().toLocaleTimeString()}`);
  } catch (e) {
    say(`failed: ${(e as Error).message}`, true);
    if (/^40[13]/.test((e as Error).message)) { signOut(); $("signin-status").textContent = `that key was refused: ${(e as Error).message}`; }
  }
}

async function revoke(k: KeyInfo): Promise<void> {
  if (k.current && !confirm(`Revoke ${k.id}, the key this tab is signed in with? Its next session mint will be refused and you will be signed out.`)) return;
  try {
    await api(`/v1/me/keys/${k.id}`, { method: "DELETE" });
    if (k.current) { signOut(); $("signin-status").textContent = `${k.id} revoked; sign in with another key.`; return; }
    await refresh();
  } catch (e) { say(`revoke failed: ${(e as Error).message}`, true); }
}

function showNewKey(nk: NewKey, note: string): void {
  const b = $("new-key");
  b.replaceChildren(el("b", "New key (shown once): "), el("code", nk.key), el("span", ` · id ${nk.id}. ${note}`));
  b.classList.add("on");
}

async function createSibling(): Promise<void> {
  try {
    const nk = await api<NewKey>("/v1/me/keys", { method: "POST" });
    showNewKey(nk, "It shares this owner, so its usage shows here too.");
    await refresh();
  } catch (e) { say(`create failed: ${(e as Error).message}`, true); }
}

async function createFirst(): Promise<void> {
  $("signin-status").textContent = `creating a key at ${API}/v1/keys …`;
  try {
    const nk = await api<NewKey>("/v1/keys", { method: "POST" });
    showNewKey(nk, "Keep it: there are no accounts, the key is your only identity. Signing you in with it now.");
    signIn(nk.key);
  } catch (e) { $("signin-status").textContent = `create failed: ${(e as Error).message}`; }
}

function signIn(k: string): void {
  if (!KEY_RE.test(k)) { $("signin-status").textContent = "that is not a dk_live_ key (dk_live_ plus 32 characters)"; return; }
  key = k; storeKey(k);
  $("signin").hidden = true; $("app").hidden = false;
  $("signin-status").textContent = "";
  void refresh();
}

function signOut(): void {
  key = null; storeKey(null);
  $("app").hidden = true; $("signin").hidden = false;
  ($("key") as HTMLInputElement).value = "";
  $("who").textContent = "";
}

$("signin-btn").onclick = () => signIn(($("key") as HTMLInputElement).value.trim());
($("key") as HTMLInputElement).onkeydown = (e) => { if (e.key === "Enter") signIn(($("key") as HTMLInputElement).value.trim()); };
$("create-btn").onclick = () => void createFirst();
$("signout").onclick = signOut;
$("refresh").onclick = () => void refresh();
$("newkey").onclick = () => void createSibling();

if (key) signIn(key);
