import type { CrossSiteProgress, CrossSiteResult, LoadSummary, Progress } from "dianome";
import { API, CDN, MODEL, el, fmtBytes, fmtRate, makeDianome } from "./common";

el("site").textContent = location.host;
el("api").textContent = API;
el("cdn").textContent = CDN;
el("model-id").textContent = MODEL;

const status = el("status"), xsite = el("xsite"), bars = el("bars"), stats = el("stats"), mount = el("mount");
const persisted = () => { el("persisted").textContent = `persisted: ${localStorage.getItem("dianome:crosssite") ?? "none"}`; };
persisted();

let d = makeDianome({ telemetry: (el("telemetry") as HTMLInputElement).checked });
el("telemetry").addEventListener("change", () => { d = makeDianome({ telemetry: (el("telemetry") as HTMLInputElement).checked }); });

// ---- cross-site opt-in: called synchronously from the click handler ----
el("enable").addEventListener("click", () => {
  xsite.textContent = "requesting (silent attempt through the hidden frame)…";
  mount.replaceChildren();
  console.log("[dianome demo] enableCrossSiteCache() called");
  d.enableCrossSiteCache().then(showXsite, (e) => showError("enableCrossSiteCache", e));
});
function showError(where: string, e: unknown): void {
  const err = e as { name?: string; message?: string; code?: string };
  console.error(`[dianome demo] ${where} failed`, e);
  xsite.textContent = `error in ${where}: ${err?.name ?? "Error"}${err?.code ? ` (${err.code})` : ""}: ${err?.message ?? String(e)}`;
  persisted();
}
const STATE_TEXT: Record<CrossSiteResult["state"], string> = {
  granted: "Shared cache active: loads now go through the cdn frame and are shared with other sites.",
  denied: "The browser denied storage access for the cdn frame (site permission). Loads use the per-site cache.",
  unsupported: "This browser cannot share the cache across sites (Safari/WebKit, or no Storage Access API). Loads use the per-site cache.",
  "needs-visit": "The browser has no first-party interaction with the cdn origin yet. Open the link once, click Enable there, and come back:",
  "needs-click": "The browser wants the click inside the cdn frame itself. Click the button below:",
};
function showXsite(r: CrossSiteResult): void {
  persisted();
  console.log(`[dianome demo] cross-site result: state=${r.state}${r.path ? ` path=${r.path}` : ""}${r.reason ? ` reason=${r.reason}` : ""}`, r);
  const lines = [`state: ${r.state}`];
  if (r.path) lines.push(`grant path: ${r.path}`);
  if (r.reason) lines.push(`reason: ${r.reason}`);
  lines.push(STATE_TEXT[r.state]);
  xsite.textContent = lines.join("\n");
  if (r.state === "needs-visit" && r.visitUrl) {
    const a = document.createElement("a"); a.href = r.visitUrl; a.textContent = r.visitUrl;
    xsite.appendChild(document.createTextNode("\n")); xsite.appendChild(a);
    return;
  }
  if (r.state === "needs-click" && r.mount) {
    const progressText: Record<CrossSiteProgress, string> = {
      "waiting-click": "frame ready, waiting for your click inside it…",
      clicked: "clicked; the frame is calling requestStorageAccess()…",
      requesting: "requestStorageAccess() pending: if the browser shows a permission prompt (address bar), answer it.",
    };
    r.mount(mount, { onProgress: (stage) => { console.log(`[dianome demo] mount progress: ${stage}`); xsite.textContent = `${lines.join("\n")}\n→ ${progressText[stage]}`; } })
      .then((final) => { mount.replaceChildren(); showXsite(final); }, (e) => { mount.replaceChildren(); showError("mount", e); });
  }
}

// ---- load ----
let ac: AbortController | null = null;
const groupBars = new Map<string, { fill: HTMLElement; label: HTMLElement; bytes: number; done: number; sources: Set<string> }>();

async function load(): Promise<void> {
  el<HTMLButtonElement>("load").disabled = true; el<HTMLButtonElement>("abort").disabled = false;
  bars.replaceChildren(); groupBars.clear();
  ac = new AbortController();
  const t0 = performance.now();
  status.textContent = `fetching manifest… (cache mode: ${await d.cache.mode()})`;
  try {
    const manifest = await d.manifest(MODEL);
    const variant = "variants" in manifest && manifest.variants ? manifest.variants.q4 ?? manifest.variants.fp16 : null;
    if (variant) for (const g of variant.groups) addBar(g.name, g.bytes);
    let last: Progress | null = null;
    const onProgress = (p: Progress) => {
      last = p;
      const b = groupBars.get(p.group);
      if (b) { b.sources.add(p.source); }
      status.textContent = `${fmtBytes(p.bytesDone)} / ${fmtBytes(p.bytesTotal)}  ·  ${p.chunksDone}/${p.chunksTotal} chunks  ·  ${fmtRate(p.bytesPerSecond)}  ·  ${(p.elapsedMs / 1000).toFixed(1)} s  ·  last: ${p.group} from ${p.source}`;
    };
    const gen = d.stream(MODEL, { variant: "q4", signal: ac.signal, onProgress });
    let summary: LoadSummary;
    for (;;) {
      const n = await gen.next();
      if (n.done) { summary = n.value; break; }
      const b = groupBars.get(n.value.name);
      if (b) { b.done = b.bytes; b.fill.style.width = "100%"; b.fill.className = cls(b.sources); b.label.textContent = `${fmtBytes(n.value.bytes)} · ${[...b.sources].join("+")}`; }
    }
    void last;
    const ms = performance.now() - t0;
    const st = await d.cache.status();
    const counts = summary.sources.reduce<Record<string, number>>((m, s) => ((m[s] = (m[s] ?? 0) + 1), m), {});
    status.textContent = [
      `done in ${(ms / 1000).toFixed(2)} s · ${fmtBytes(summary.bytes)} · ${summary.chunks} chunks · ${fmtRate(summary.bytesPerSecond)}`,
      `source: ${summary.source} · cache hits ${summary.cacheHits}/${summary.chunks} · breakdown ${JSON.stringify(counts)}`,
      `cache mode: ${summary.cacheMode}${summary.cacheDisabled ? " (quota fallback during this load)" : ""} · verify ${summary.verifyMs.toFixed(0)} ms · postMessage hop median ${summary.transferMs.toFixed(1)} ms/chunk (${summary.transferSamples.length} samples)`,
      `cache status: ${st.mode}, ${st.chunks} chunks, ${fmtBytes(st.usageBytes)} used${st.quotaBytes ? ` of ${fmtBytes(st.quotaBytes)}` : ""}`,
      `telemetry: ${summary.report ? `sent (HTTP ${summary.telemetryStatus})` : "off"}${summary.report ? `\n${JSON.stringify(summary.report)}` : ""}`,
    ].join("\n");
  } catch (e) {
    status.textContent = `error: ${(e as Error).name}: ${(e as Error).message}`;
  } finally {
    el<HTMLButtonElement>("load").disabled = false; el<HTMLButtonElement>("abort").disabled = true; ac = null;
  }
}
function cls(sources: Set<string>): string {
  if (sources.size > 1) return "mixed";
  const s = [...sources][0];
  return s === "per-site-cache" ? "cache" : s === "cross-site-cache" ? "xsite" : "";
}
function addBar(name: string, bytes: number): void {
  const n = document.createElement("span"); n.textContent = name;
  const bar = document.createElement("div"); bar.className = "bar";
  const fill = document.createElement("i"); bar.appendChild(fill);
  const label = document.createElement("span"); label.textContent = fmtBytes(bytes);
  bars.append(n, bar, label);
  groupBars.set(name, { fill, label, bytes, done: 0, sources: new Set() });
}
el("load").addEventListener("click", () => void load());
el("abort").addEventListener("click", () => ac?.abort());
el("evict").addEventListener("click", async () => { await d.cache.evict(MODEL); status.textContent = `evicted ${MODEL}: ${JSON.stringify(await d.cache.status())}`; });
el("clear").addEventListener("click", async () => { await d.cache.clear(); status.textContent = `cleared: ${JSON.stringify(await d.cache.status())}`; });

// ---- measurements: SHA-256 throughput on an 8 MB buffer, and postMessage transfer cost through the frame ----
el("measure").addEventListener("click", async () => {
  const MB8 = 8 * 1024 * 1024;
  const buf = new Uint8Array(MB8);
  for (let o = 0; o < MB8; o += 65536) crypto.getRandomValues(buf.subarray(o, o + 65536));
  const rounds = 5;
  let t = performance.now();
  for (let i = 0; i < rounds; i++) await crypto.subtle.digest("SHA-256", buf);
  const shaMs = (performance.now() - t) / rounds;
  const lines = [`sha256 of 8 MiB: ${shaMs.toFixed(1)} ms/chunk → ${(MB8 / 1e6 / (shaMs / 1000)).toFixed(0)} MB/s (mean of ${rounds})`];
  // Transfer cost: the cross-site store's per-op timings, if the shared cache is active.
  const st = await d.cache.status();
  if (st.mode === "cross-site") {
    const store = await (d as unknown as { store(): Promise<{ timings?: () => Record<string, { count: number; totalMs: number; maxMs: number; frameMs: number; hopOutMs: number; hopInMs: number }> }> }).store();
    const tm = store?.timings?.();
    if (tm) for (const [op, v] of Object.entries(tm)) lines.push(`frame ${op}: n=${v.count} mean ${(v.totalMs / v.count).toFixed(1)} ms (frame-side ${(v.frameMs / v.count).toFixed(1)} ms) max ${v.maxMs.toFixed(1)} ms`);
    if (tm?.fetch) lines.push(`→ measured frame→parent hop per fetch (mean) ${(tm.fetch.hopOutMs / tm.fetch.count).toFixed(2)} ms; request hop ${(tm.fetch.hopInMs / tm.fetch.count).toFixed(2)} ms`);
  } else {
    lines.push(`cross-site store not active (mode ${st.mode}); enable the shared cache and load once to measure transfer cost`);
  }
  stats.textContent = lines.join("\n");
});
