// /stats: the Phase 2 load dashboard (moved from apps/load-dashboard) plus the Phase 7 sessions view from the
// schema-3 data (GET /v1/stats/sessions). Plain DOM, no dependencies.
import { $, API, el, fmtBytes, nav, pct, sec, table } from "./shared";

nav("/stats/");

interface CountryStats { country: string; loads: number; p50_ms: number; p90_ms: number; cache_hit_rate: number }
interface ModelVariantStats { model: string; variant: string; loads: number; p50_ms: number; bytes: number }
interface SourceStats { source: string; loads: number; p50_ms: number }
interface LoadStats { since: string; window_hours: number; computed_at?: string; by_country: CountryStats[]; by_model_variant: ModelVariantStats[]; by_source: SourceStats[]; degraded?: boolean }
interface ModelModeStats { model: string; mode: string; sessions: number; p50_tok_per_s: number; p50_n: number; L: number; p50_server_busy_ms: number; p50_rtt_ms: number; new_tokens: number }
interface PolicyStats { policy: string; mode: string; sessions: number }
interface BrowserSessionStats { browser: string; sessions: number; p50_tok_per_s: number; webgpu_share: number }
interface SessionStats { since: string; window_hours: number; computed_at?: string; by_model_mode: ModelModeStats[]; by_policy: PolicyStats[]; by_browser: BrowserSessionStats[]; degraded?: boolean }

async function fetchJson<T>(path: string, statusEl: HTMLElement): Promise<T | null> {
  const url = `${API}${path}`;
  statusEl.textContent = `loading ${url} …`;
  try {
    const res = await fetch(url, { cache: "no-store" });
    if (!res.ok) throw new Error(`${res.status} ${res.statusText}`);
    const v = (await res.json()) as T;
    statusEl.textContent = `loaded ${url} (${new Date().toLocaleTimeString()})`;
    return v;
  } catch (e) {
    statusEl.className = "err";
    statusEl.textContent = `failed to load ${url}: ${(e as Error).message}`;
    return null;
  }
}
const note = (s: { since: string; window_hours: number; computed_at?: string }) => `since ${s.since} · window ${s.window_hours} h (${(s.window_hours / 24).toFixed(0)} days)` + (s.computed_at ? ` · computed ${s.computed_at}` : "");

export function renderLoads(s: LoadStats): void {
  $("loads-banner").classList.toggle("on", s.degraded === true);
  $("loads-note").textContent = note(s);
  const app = $("loads"); app.replaceChildren();
  const byCountry = [...s.by_country].sort((a, b) => b.loads - a.loads);
  const sec1 = el("section"); sec1.append(el("h3", "By country"));
  sec1.append(table([{ label: "country" }, { label: "loads", num: true }, { label: "p50", num: true }, { label: "p90", num: true }, { label: "cache-hit rate", num: true }],
    byCountry.map((c) => [c.country, c.loads, sec(c.p50_ms), sec(c.p90_ms), pct(c.cache_hit_rate)]), "no loads in this window"));
  if (byCountry.length > 0) {
    const max = byCountry[0]!.loads || 1;
    const bars = el("div", undefined, "bars");
    for (const c of byCountry) {
      const row = el("div"); row.append(el("span", c.country, "label"));
      const bar = el("span", undefined, "bar"); bar.style.width = `${Math.max(1, (400 * c.loads) / max)}px`;
      row.append(bar, el("span", c.loads.toLocaleString())); bars.append(row);
    }
    sec1.append(bars);
  }
  const sec2 = el("section"); sec2.append(el("h3", "By model / variant"));
  sec2.append(table([{ label: "model" }, { label: "variant" }, { label: "loads", num: true }, { label: "p50", num: true }, { label: "bytes", num: true }],
    [...s.by_model_variant].sort((a, b) => b.loads - a.loads).map((m) => [m.model, m.variant, m.loads, sec(m.p50_ms), fmtBytes(m.bytes)]), "no loads in this window"));
  const sec3 = el("section"); sec3.append(el("h3", "By source"));
  sec3.append(table([{ label: "source" }, { label: "loads", num: true }, { label: "p50", num: true }],
    [...s.by_source].sort((a, b) => b.loads - a.loads).map((x) => [x.source, x.loads, sec(x.p50_ms)]), "no loads in this window"));
  app.append(sec1, sec2, sec3);
}

export function renderSessions(s: SessionStats): void {
  $("sessions-banner").classList.toggle("on", s.degraded === true);
  $("sessions-note").textContent = note(s);
  const app = $("sessions"); app.replaceChildren();
  const sec1 = el("section"); sec1.append(el("h3", "By model and mode"));
  sec1.append(table([{ label: "model" }, { label: "mode" }, { label: "sessions", num: true }, { label: "p50 tok/s", num: true }, { label: "p50 N / L", num: true }, { label: "p50 server busy", num: true }, { label: "p50 RTT", num: true }, { label: "new tokens", num: true }],
    [...s.by_model_mode].sort((a, b) => b.sessions - a.sessions).map((m) => [m.model, m.mode, m.sessions, m.p50_tok_per_s.toFixed(1), `${m.p50_n} / ${m.L}`, `${m.p50_server_busy_ms} ms`, `${m.p50_rtt_ms.toFixed(1)} ms`, m.new_tokens]), "no sessions in this window"));
  const sec2 = el("section"); sec2.append(el("h3", "By policy: what the planner chose"));
  sec2.append(table([{ label: "policy" }, { label: "mode chosen" }, { label: "sessions", num: true }],
    [...s.by_policy].sort((a, b) => b.sessions - a.sessions).map((p) => [p.policy, p.mode, p.sessions]), "no sessions in this window"));
  const sec3 = el("section"); sec3.append(el("h3", "By browser"));
  sec3.append(table([{ label: "browser" }, { label: "sessions", num: true }, { label: "p50 tok/s", num: true }, { label: "WebGPU share", num: true }],
    [...s.by_browser].sort((a, b) => b.sessions - a.sessions).map((b) => [b.browser, b.sessions, b.p50_tok_per_s.toFixed(1), pct(b.webgpu_share)]), "no sessions in this window"));
  app.append(sec1, sec2, sec3);
}

void (async () => {
  const [loads, sessions] = await Promise.all([fetchJson<LoadStats>("/v1/stats/loads", $("loads-status")), fetchJson<SessionStats>("/v1/stats/sessions", $("sessions-status"))]);
  if (loads) renderLoads(loads);
  if (sessions) renderSessions(sessions);
})();
