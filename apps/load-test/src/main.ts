import { buildReport, detectBrowser, loadVariant, postLoadReport, type Progress } from "./load";

const DEFAULT_API = "http://localhost:8787";
const DEFAULT_CDN = "https://cdn.dianome.dev";
const DEFAULT_ID = "qwen2.5-0.5b-instruct";

const $ = <T extends HTMLElement>(id: string) => document.getElementById(id) as T;
const form = $<HTMLFormElement>("form");
const manifestInput = $<HTMLInputElement>("manifest");
const cdnInput = $<HTMLInputElement>("cdn");
const variantSelect = $<HTMLSelectElement>("variant");
const go = $<HTMLButtonElement>("go");
const status = $("status");
const stats = $("stats");
const groupsEl = $("groups");
const result = $<HTMLPreElement>("result");

const params = new URLSearchParams(location.search);
manifestInput.value = params.get("manifest") ?? `${params.get("api") ?? DEFAULT_API}/v1/models/${params.get("id") ?? DEFAULT_ID}/manifest`;
cdnInput.value = params.get("cdn") ?? DEFAULT_CDN;
variantSelect.value = params.get("variant") ?? "q4";

const fmtMiB = (n: number) => `${(n / 1048576).toFixed(1)} MiB`;
const fmtRate = (bps: number) => `${(bps / 1048576).toFixed(1)} MiB/s`;

function stat(label: string, value: string): HTMLElement {
  const d = document.createElement("div");
  const s = document.createElement("span"); s.textContent = label;
  const b = document.createElement("b"); b.textContent = value;
  d.append(s, b);
  return d;
}

function render(p: Progress, state: "running" | "ok" | "bad" = "running"): void {
  stats.replaceChildren(
    stat("received", `${fmtMiB(p.bytesDone)} / ${fmtMiB(p.bytesTotal)}`),
    stat("chunks", `${p.chunksDone} / ${p.chunksTotal}`),
    stat("elapsed", `${(p.elapsedMs / 1000).toFixed(1)} s`),
    stat("throughput", fmtRate(p.bytesPerSecond)),
  );
  groupsEl.replaceChildren(...p.groups.map((g) => {
    const row = document.createElement("div");
    const label = document.createElement("span"); label.className = "label"; label.textContent = g.name;
    const count = document.createElement("span"); count.className = "count"; count.textContent = `${g.done}/${g.total}`;
    const bar = document.createElement("span"); bar.className = `bar ${state === "running" ? "" : state}`;
    const fill = document.createElement("i"); fill.style.width = g.total ? `${(100 * g.done) / g.total}%` : "100%";
    bar.append(fill);
    row.append(label, count, bar);
    return row;
  }));
}

form.addEventListener("submit", (ev) => { ev.preventDefault(); void run(); });
if (params.get("autostart") === "1") void run();

async function run(): Promise<void> {
  go.disabled = true;
  result.textContent = "…";
  const manifestUrl = manifestInput.value.trim();
  const apiBase = new URL(manifestUrl).origin;
  const variant = variantSelect.value as "fp16" | "q8" | "q4";
  status.className = "";
  status.textContent = `loading ${variant} from ${manifestUrl} …`;
  let lastPaint = 0;
  try {
    const r = await loadVariant({
      manifestUrl, variant, chunkBase: cdnInput.value.trim(), concurrency: 6,
      onProgress: (p) => { const now = performance.now(); if (now - lastPaint > 100 || p.chunksDone === p.chunksTotal) { lastPaint = now; render(p); } },
    });
    render({ bytesDone: r.bytes, bytesTotal: r.bytes, chunksDone: r.chunks, chunksTotal: r.chunks, elapsedMs: r.ms, bytesPerSecond: r.bytesPerSecond, groups: [] }, "ok");
    const report = buildReport(r, detectBrowser(navigator.userAgent), "gpu" in navigator);
    status.textContent = `loaded ${r.chunks} chunks (${fmtMiB(r.bytes)}) in ${(r.ms / 1000).toFixed(2)} s = ${fmtRate(r.bytesPerSecond)}; posting report …`;
    let telemetry: string;
    try { telemetry = `HTTP ${await postLoadReport(apiBase, report)}`; } catch (e) { telemetry = `failed: ${(e as Error).message}`; }
    status.textContent = `done: ${r.chunks} chunks, ${fmtMiB(r.bytes)}, ${(r.ms / 1000).toFixed(2)} s, ${fmtRate(r.bytesPerSecond)}; telemetry ${telemetry}`;
    result.textContent = JSON.stringify({ result: r, report, telemetry }, null, 2);
  } catch (e) {
    status.className = "err";
    status.textContent = `failed: ${(e as Error).message}`;
    result.textContent = String((e as Error).stack ?? e);
  } finally {
    go.disabled = false;
  }
}
