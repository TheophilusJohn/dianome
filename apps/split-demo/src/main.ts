// Split demo: planner inputs measured on this device, a slider over N with live estimates (ms/token, server share and
// cost at the stated rate, privacy band), then run() with the selection forced, streaming tokens and showing measured
// vs estimated, the per-step breakdown and the telemetry that was sent.
import { Dianome } from "dianome";
import type { Plan, PlanCandidate, RunOptions, RunResult } from "dianome";

const params = new URLSearchParams(location.search);
const api = params.get("api") ?? undefined, cdn = params.get("cdn") ?? undefined;
const splitUrl = params.get("split"), splitToken = params.get("token");
const MODEL = "qwen2.5-0.5b-instruct";
const VARIANT = "q4" as const;
const d = new Dianome({ ...(api ? { api } : {}), ...(cdn ? { cdn } : {}) });
const split = splitUrl && splitToken ? { url: splitUrl, token: splitToken } : undefined;

const $ = <T extends HTMLElement>(id: string) => document.getElementById(id) as T;
const slider = $<HTMLInputElement>("slider"), nval = $("nval"), localBox = $<HTMLInputElement>("local"), auto = $<HTMLSelectElement>("auto");
const status = $("status"), runBtn = $<HTMLButtonElement>("run");
const f = (n: number | null | undefined, digits = 1): string => (n === null || n === undefined || !Number.isFinite(n) ? "–" : n.toFixed(digits));
const mib = (b: number) => (b / 2 ** 20).toFixed(0);
const kv = (el: HTMLElement, rows: [string, string][]) => { el.innerHTML = rows.map(([k, v]) => `<dt>${k}</dt><dd>${v}</dd>`).join(""); };

let plan: Plan | null = null;
let L = 24;

function baseOpts(): RunOptions {
  return { variant: VARIANT, ...(split ? { split } : {}), policy: { prefer: auto.value === "manual" ? "cost" : (auto.value as "cost" | "latency" | "local" | "server") } };
}

async function refreshInputs(): Promise<void> {
  $("inputs-status").textContent = "measuring…";
  runBtn.disabled = true;
  try {
    plan = await d.planRun(MODEL, { ...baseOpts(), prompt: $<HTMLTextAreaElement>("prompt").value, maxTokens: Number($<HTMLInputElement>("maxTokens").value) });
    L = plan.inputs.model.L;
    slider.max = String(L);
    const i = plan.inputs;
    kv($("dev"), [
      ["WebGPU", i.device.webgpu ? "yes" : "no"], ["maxBufferSize", i.device.maxBufferSize === null ? "–" : `${mib(i.device.maxBufferSize)} MiB`],
      ["GPU budget", `${mib(i.device.gpuBudgetBytes)} MiB (storage quota ${i.device.quotaBytes ? mib(i.device.quotaBytes) + " MiB" : "unknown"})`], ["cached", `${i.device.embedCached ? "embed + " : ""}${i.device.cachedBlocks} blocks`],
      ["variant", `${i.model.variant} (${mib(i.model.blockBytes[0] ?? 0)} MiB/block, embed ${mib(i.model.embedBytes)} MiB)`],
    ]);
    kv($("mb"), [
      ["ms/block, T=1 (decode)", f(i.device.msPerBlockT1, 3)], ["ms/block, T=32 (prefill)", f(i.device.msPerBlockT32, 3)],
      ["lm_head + readback ms", f(i.device.lmHeadMs, 2)], ["export ms", f(i.device.exportMs, 2)],
    ]);
    kv($("net"), [
      ["download bandwidth", i.network.bytesPerSecond === null ? "unknown" : `${(i.network.bytesPerSecond / 1e6).toFixed(0)} MB/s`],
      ["server reachable", i.network.serverReachable ? "yes" : "no"], ["RTT (WS ping, median of 5)", i.network.rttMs === null ? "–" : `${f(i.network.rttMs, 2)} ms`],
      ["split server", split ? split.url : `${d.api}/v1/split/session`],
    ]);
    const s = i.server, r = i.rate;
    kv($("srv"), [
      ["server L / busy fraction 60 s", s ? `${s.L} / ${f(s.busyFraction60s, 3)}` : "–"], ["server ms/block decode / prefill", s ? `${f(s.msPerBlockDecode, 3)} / ${f(s.msPerBlockPrefill, 3)}` : "–"],
      ["server lm_head ms", s ? f(s.lmHeadMs, 2) : "–"], ["server device", s?.device ?? "–"],
      ["rate", r && r.usdPerHour !== null ? `${r.gpu}: $${r.usdPerHour}/h (retrieved ${r.retrieved})` : `none stated${r ? ` (${r.gpu}, ${r.retrieved})` : ""}: shares only`],
      ["cost at N=0", plan.cost0Per1M === null ? "–" : `$${plan.cost0Per1M.toFixed(2)} per 1M tokens`],
    ]);
    $("inputs-status").textContent = `measured ${new Date().toLocaleTimeString()}`;
    if (auto.value !== "manual") { if (plan.mode === "local") { localBox.checked = true; slider.value = String(L); } else { localBox.checked = false; slider.value = String(plan.N); } }
    renderSelection();
    renderCandidates();
    runBtn.disabled = false;
    status.textContent = "ready";
  } catch (e) {
    $("inputs-status").textContent = `error: ${(e as Error).message}`;
    status.textContent = `error: ${(e as Error).message}`;
  }
}

function selected(): PlanCandidate | null {
  if (!plan) return null;
  const N = Number(slider.value);
  if (localBox.checked && N === L) return plan.candidates.find((c) => c.mode === "local") ?? null;
  return plan.candidates.find((c) => c.mode !== "local" && c.N === N) ?? null;
}

function renderSelection(): void {
  const N = Number(slider.value);
  nval.textContent = String(N);
  localBox.disabled = N !== L;
  if (N !== L) localBox.checked = false;
  const c = selected();
  if (!plan || !c) return;
  $("choice").innerHTML = `${c.mode} at N = ${c.N}${c.feasible ? "" : ` <span class="warn">(infeasible: ${c.reasons.join("; ")})</span>`}${auto.value !== "manual" ? ` — policy <b>${auto.value}</b> chose ${plan.mode} N=${plan.N}: ${plan.reasons.join("; ")}` : ""}`;
  $("est-ms").textContent = c.msPerToken === null ? "–" : `${f(c.msPerToken)} ms`;
  const b = c.breakdown;
  $("est-ms-detail").textContent = b ? `client ${f(b.clientMs)} + export ${f(b.exportMs)} + rtt ${f(b.rttMs)} + server ${f(b.serverMs)} + lm_head ${f(b.lmHeadMs)} + sampling ${f(b.samplingMs)}; prefill ≈ ${f(c.prefillMs)} ms` : "no estimate (an input is missing)";
  const r = plan.inputs.rate;
  $("est-cost").textContent = `${(c.serverShare * 100).toFixed(0)}% ${c.costPer1M === null ? "" : ` · $${c.costPer1M.toFixed(2)} / 1M`}`;
  $("est-cost-detail").textContent = r && r.usdPerHour !== null ? `share of the N = 0 server cost; ${r.gpu} at $${r.usdPerHour}/h (${r.source}, retrieved ${r.retrieved}); lm_head floor ${(plan.lmHeadBlockEquivalents).toFixed(1)} block-equivalents` : "no rate stated (rates.json usd_per_hour is null): share only";
  if (c.mode === "local") { $("est-priv").textContent = "nothing sent"; $("est-priv-detail").textContent = "local mode: no hidden state leaves this device"; }
  else if (c.privacy) {
    $("est-priv").textContent = `${(c.privacy.linear500kTop1 * 100).toFixed(1)}% / ${(c.privacy.inversionTop1 * 100).toFixed(1)}%`;
    $("est-priv-detail").textContent = `linear probe (500k) / inversion decoder top-1 at boundary ${c.privacy.boundary}: the fraction of input tokens a server could recover from what it receives (Phase 4, held-out WikiText).${c.N === 0 ? " N = 0 sends the token ids themselves." : ""}`;
  } else { $("est-priv").textContent = "–"; $("est-priv-detail").textContent = "no band for this model"; }
}

function renderCandidates(): void {
  if (!plan) return;
  const sel = selected();
  const tb = document.querySelector("#cands tbody")!;
  tb.innerHTML = plan.candidates.map((c) => `<tr class="cand ${c === sel ? "sel" : ""} ${c.feasible ? "" : "no"}" data-mode="${c.mode}" data-n="${c.N}"><td>${c.mode}</td><td>${c.N}</td><td>${c.feasible ? "yes" : "no"}</td><td>${f(c.msPerToken)}</td><td>${(c.serverShare * 100).toFixed(0)}%</td><td>${c.costPer1M === null ? "–" : c.costPer1M.toFixed(2)}</td><td>${c.privacy ? (c.privacy.linear500kTop1 * 100).toFixed(1) : "–"}</td><td>${c.privacy ? (c.privacy.inversionTop1 * 100).toFixed(1) : "–"}</td><td>${mib(c.gpuBytes)}</td><td>${mib(c.downloadBytes)}</td><td style="text-align:left">${c.reasons.join("; ")}</td></tr>`).join("");
  tb.querySelectorAll<HTMLTableRowElement>("tr.cand").forEach((tr) => { tr.onclick = () => { auto.value = "manual"; const mode = tr.dataset.mode, n = Number(tr.dataset.n); slider.value = String(n); localBox.checked = mode === "local"; renderSelection(); renderCandidates(); }; });
}

slider.oninput = () => { auto.value = "manual"; renderSelection(); renderCandidates(); };
localBox.onchange = () => { auto.value = "manual"; renderSelection(); renderCandidates(); };
auto.onchange = () => { void refreshInputs(); };
$("refresh").onclick = () => { void refreshInputs(); };

runBtn.onclick = async () => {
  const c = selected();
  if (!plan || !c) return;
  runBtn.disabled = true;
  $("output").textContent = "";
  $("telemetry").textContent = "(sending after the run)";
  status.textContent = `loading + running ${c.mode} N=${c.N}…`;
  const t0 = performance.now();
  try {
    const temp = Number($<HTMLInputElement>("temp").value);
    const r: RunResult = await d.run(MODEL, {
      ...baseOpts(),
      messages: [{ role: "user", content: $<HTMLTextAreaElement>("prompt").value }],
      maxTokens: Number($<HTMLInputElement>("maxTokens").value),
      sampling: { temperature: temp },
      force: c.mode === "local" ? { mode: "local" } : { N: c.N },
      onProgress: (p) => { status.textContent = `loading ${p.group}: ${(p.bytesDone / 1e6).toFixed(0)} MB at ${(p.bytesPerSecond / 1e6).toFixed(0)} MB/s`; },
      onToken: (t) => { $("output").textContent += t; },
    });
    const dec = r.timings.slice(1);
    const med = (k: keyof (typeof dec)[number]) => median(dec.map((s) => Number(s[k])));
    const measured = med("totalMs");
    const est = r.plan.estimate.msPerToken;
    kv($("meas"), [
      ["mode / N", `${r.mode} / ${r.N}`], ["tokens", `${r.tokens.length} in ${((performance.now() - t0) / 1000).toFixed(1)} s (${f(r.tokPerS)} tok/s decode)`],
      ["measured ms/token (median)", f(measured, 2)], ["estimated ms/token", f(est, 2)], ["ratio measured / estimated", est ? f(measured / est, 2) : "–"],
      ["prefill ms (step 0)", `${f(r.timings[0]?.totalMs, 1)} (est ${f(r.plan.estimate.prefillMs, 1)})`],
      ["server busy_ms total", f(r.serverBusyMs, 1)], ["server share / cost", `${(r.costEstimate.serverShare * 100).toFixed(0)}%${r.costEstimate.costPer1M === null ? "" : ` · $${r.costEstimate.costPer1M.toFixed(2)}/1M`}`],
      ["load", r.load ? `${(r.load.bytes / 2 ** 20).toFixed(0)} MiB in ${(r.load.ms / 1000).toFixed(1)} s (${r.load.cacheHits} of ${r.load.chunks} chunks cached)` : "runtime reused"],
    ]);
    const parts: [string, number, string][] = [
      ["client", med("clientMs"), "var(--client)"], ["export", med("exportMs"), "var(--export)"],
      ["network", Math.max(0, med("roundTripMs") - med("serverBusyMs")), "var(--net)"], ["server", med("serverBusyMs"), "var(--server)"],
      ["lm_head", med("lmHeadMs"), "var(--head)"], ["sampling", med("sampleMs"), "var(--sample)"],
    ];
    const total = parts.reduce((n, p) => n + p[1], 0) || 1;
    $("bar").innerHTML = parts.map(([, v, col]) => `<span style="width:${((v / total) * 100).toFixed(1)}%;background:${col}"></span>`).join("");
    $("bar-detail").textContent = parts.map(([k, v]) => `${k} ${f(v, 2)}`).join(" · ") + ` ms (sum ${f(total, 2)}; measured step median ${f(measured, 2)})`;
    $("telemetry").textContent = r.telemetry.report ? `${JSON.stringify(r.telemetry.report, null, 2)}\n→ POST ${d.api}/v1/telemetry/load: ${r.telemetry.status ?? "failed"}` : "telemetry off";
    status.textContent = "done";
  } catch (e) {
    status.textContent = `error: ${(e as Error).message}`;
    console.error(e);
  } finally {
    runBtn.disabled = false;
  }
};

function median(a: number[]): number { const s = [...a].sort((x, y) => x - y); return s.length ? s[Math.floor(s.length / 2)]! : 0; }

void refreshInputs();
