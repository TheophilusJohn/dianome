// Renders GET /v1/stats/loads: tables by country / model+variant / source, one bar per country, a note line
// with the window, and a "degraded" banner when the API says so. Plain DOM, no dependencies.

interface CountryStats { country: string; loads: number; p50_ms: number; p90_ms: number; cache_hit_rate: number }
interface ModelVariantStats { model: string; variant: string; loads: number; p50_ms: number; bytes: number }
interface SourceStats { source: string; loads: number; p50_ms: number }
interface LoadStats {
  since: string; window_hours: number; computed_at?: string;
  by_country: CountryStats[]; by_model_variant: ModelVariantStats[]; by_source: SourceStats[];
  degraded?: boolean;
}

const params = new URLSearchParams(location.search);
const API = (params.get("api") ?? (import.meta.env.VITE_API_BASE as string | undefined) ?? "http://localhost:8787").replace(/\/+$/, "");

const app = document.getElementById("app")!;
const status = document.getElementById("status")!;
const note = document.getElementById("note")!;
const banner = document.getElementById("banner")!;

void load();

async function load(): Promise<void> {
  const url = `${API}/v1/stats/loads`;
  status.textContent = `loading ${url} …`;
  try {
    const res = await fetch(url, { cache: "no-store" });
    if (!res.ok) throw new Error(`${res.status} ${res.statusText}`);
    const stats = (await res.json()) as LoadStats;
    status.textContent = `loaded ${url} (${new Date().toLocaleTimeString()})`;
    render(stats);
  } catch (e) {
    status.className = "err";
    status.textContent = `failed to load ${url}: ${(e as Error).message}`;
  }
}

// ---------------------------------------------------------------- DOM helpers

function el<K extends keyof HTMLElementTagNameMap>(tag: K, text?: string | number, cls?: string): HTMLElementTagNameMap[K] {
  const e = document.createElement(tag);
  if (text !== undefined) e.textContent = String(text);
  if (cls) e.className = cls;
  return e;
}

function table(headers: { label: string; num?: boolean }[], rows: (string | number)[][], emptyText: string): HTMLElement {
  if (rows.length === 0) return el("p", emptyText, "empty");
  const t = el("table");
  const tr = el("tr");
  for (const h of headers) tr.append(el("th", h.label, h.num ? "num" : undefined));
  const thead = el("thead");
  thead.append(tr);
  t.append(thead);
  const body = t.appendChild(el("tbody"));
  for (const r of rows) {
    const row = el("tr");
    r.forEach((cell, i) => row.append(el("td", typeof cell === "number" ? cell.toLocaleString() : cell, headers[i]?.num ? "num" : undefined)));
    body.append(row);
  }
  return t;
}

const fmtBytes = (n: number): string => (n >= 1024 ** 3 ? `${(n / 1024 ** 3).toFixed(2)} GiB` : `${(n / 1024 ** 2).toFixed(1)} MiB`);
const pct = (x: number): string => `${(100 * x).toFixed(0)} %`;
const sec = (ms: number): string => `${(ms / 1000).toFixed(1)} s`;

// ---------------------------------------------------------------- rendering

export function render(s: LoadStats): void {
  banner.classList.toggle("on", s.degraded === true);
  note.textContent = `since ${s.since} · window ${s.window_hours} h (${(s.window_hours / 24).toFixed(0)} days)` + (s.computed_at ? ` · computed ${s.computed_at}` : "");
  app.replaceChildren();

  const byCountry = [...s.by_country].sort((a, b) => b.loads - a.loads);
  const sec1 = el("section");
  sec1.append(el("h2", "By country"));
  sec1.append(table(
    [{ label: "country" }, { label: "loads", num: true }, { label: "p50", num: true }, { label: "p90", num: true }, { label: "cache-hit rate", num: true }],
    byCountry.map((c) => [c.country, c.loads, sec(c.p50_ms), sec(c.p90_ms), pct(c.cache_hit_rate)]),
    "no loads in this window",
  ));
  if (byCountry.length > 0) {
    const max = byCountry[0]!.loads || 1;
    const bars = el("div", undefined, "bars");
    for (const c of byCountry) {
      const row = el("div");
      row.append(el("span", c.country, "label"));
      const bar = el("span", undefined, "bar");
      bar.style.width = `${Math.max(1, (400 * c.loads) / max)}px`;
      row.append(bar, el("span", c.loads.toLocaleString()));
      bars.append(row);
    }
    sec1.append(bars);
  }
  app.append(sec1);

  const sec2 = el("section");
  sec2.append(el("h2", "By model / variant"));
  sec2.append(table(
    [{ label: "model" }, { label: "variant" }, { label: "loads", num: true }, { label: "p50", num: true }, { label: "bytes", num: true }],
    [...s.by_model_variant].sort((a, b) => b.loads - a.loads).map((m) => [m.model, m.variant, m.loads, sec(m.p50_ms), fmtBytes(m.bytes)]),
    "no loads in this window",
  ));
  app.append(sec2);

  const sec3 = el("section");
  sec3.append(el("h2", "By source"));
  sec3.append(table(
    [{ label: "source" }, { label: "loads", num: true }, { label: "p50", num: true }],
    [...s.by_source].sort((a, b) => b.loads - a.loads).map((x) => [x.source, x.loads, sec(x.p50_ms)]),
    "no loads in this window",
  ));
  app.append(sec3);
}
