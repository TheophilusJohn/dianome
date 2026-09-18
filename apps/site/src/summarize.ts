// The vertical demo (Phase 7 Part B): one document, run() with prefer: "cost", a plain status line, and a compare
// button that runs the same document in the other two modes when the plan says they are feasible.
import { Dianome } from "dianome";
import type { Plan, PlanCandidate, RunOptions, RunResult } from "dianome";
import { $, API, CDN, el, f, nav, params } from "./shared";

nav("/summarize/");

const SYSTEM_PROMPT = "You are a careful summariser. Summarise the document the user provides in at most five plain sentences. Keep names, numbers and dates exactly as written and add nothing that is not in the document.";
const MAX_TOKENS = 256;
const MAX_CTX = 8192; // the Worker mints sessions up to this and the server accepts up to this
const VARIANT = "q4" as const;
const MODELS: { id: string; label: string }[] = [
  { id: "qwen2.5-0.5b-instruct", label: "Qwen2.5 0.5B Instruct (q4)" },
  { id: "qwen2.5-3b-instruct", label: "Qwen2.5 3B Instruct (q4)" },
  { id: "qwen2.5-7b-instruct", label: "Qwen2.5 7B Instruct (q4)" },
];
const DEFAULT_MODEL = "qwen2.5-3b-instruct"; // once it exists in R2; else the first available (0.5B today)

const splitUrl = params.get("split"), splitToken = params.get("token");
const split = splitUrl && splitToken ? { url: splitUrl, token: splitToken } : undefined;
const d = new Dianome({ api: API, cdn: CDN });

const modelSel = $<HTMLSelectElement>("model"), doc = $<HTMLTextAreaElement>("doc"), runBtn = $<HTMLButtonElement>("run"), cmpBtn = $<HTMLButtonElement>("compare");
const status = $("status"), output = $("output"), compareOut = $("compare-out"), resultJson = $("result-json");

// -- status line -------------------------------------------------------------------------------
const lines = new Map<string, HTMLElement>();
function say(key: string, text: string, cls?: string): void {
  let line = lines.get(key);
  if (!line) { line = el("div"); lines.set(key, line); status.append(line); }
  line.textContent = text;
  line.className = cls ?? "";
}
function clearStatus(): void { lines.clear(); status.replaceChildren(); }

// -- models --------------------------------------------------------------------------------------
async function availableModels(): Promise<Set<string> | null> {
  try {
    const r = await fetch(`${API}/v1/models`, { cache: "no-store" });
    if (!r.ok) return null;
    const j = (await r.json()) as { models?: string[] };
    return new Set(j.models ?? []);
  } catch { return null; }
}
async function fillModels(): Promise<void> {
  const have = await availableModels();
  modelSel.replaceChildren(...MODELS.map((m) => {
    const o = el("option", m.label + (have && !have.has(m.id) ? " — not in the store yet" : ""));
    o.value = m.id; o.disabled = have !== null && !have.has(m.id);
    return o;
  }));
  const wanted = params.get("model");
  const pick = [wanted, DEFAULT_MODEL, ...MODELS.map((m) => m.id)].find((id) => id && (have === null || have.has(id)));
  if (pick) modelSel.value = pick;
  $("models-status").textContent = have === null ? `could not list models at ${API}` : `${have.size} model${have.size === 1 ? "" : "s"} in the store`;
  runBtn.disabled = !modelSel.value;
}

// -- context length: sized to the document, corrected from run()'s own count if the estimate was short ---------------
const roundUp = (n: number, m: number) => Math.ceil(n / m) * m;
function ctxFor(text: string): number {
  const est = Math.ceil(text.length / 3); // conservative for English (~4 chars/token)
  return Math.min(MAX_CTX, Math.max(1024, roundUp(est + MAX_TOKENS + 32, 256)));
}
const TOO_LONG = /prompt \((\d+) tokens\) \+ maxTokens \((\d+)\) exceeds maxCtx (\d+)/;

function baseOpts(text: string, maxCtx: number): RunOptions {
  return {
    messages: [{ role: "system", content: SYSTEM_PROMPT }, { role: "user", content: text }],
    variant: VARIANT, policy: { prefer: "cost" }, maxTokens: MAX_TOKENS, maxCtx, sampling: { temperature: 0 },
    ...(split ? { split } : {}),
  };
}

/** run() with the context sized to the document; one retry with the exact count when the estimate was short. */
async function runOnce(model: string, text: string, extra: Partial<RunOptions>): Promise<RunResult> {
  let maxCtx = ctxFor(text);
  for (let attempt = 0; ; attempt++) {
    try {
      return await d.run(model, { ...baseOpts(text, maxCtx), ...extra });
    } catch (e) {
      const m = TOO_LONG.exec((e as Error).message ?? "");
      if (!m || attempt > 0) throw e;
      const need = roundUp(Number(m[1]) + Number(m[2]) + 32, 256);
      if (need > MAX_CTX) throw new Error(`document is ${m[1]} tokens; the limit is ${MAX_CTX - MAX_TOKENS} (plus ${MAX_TOKENS} new tokens)`);
      maxCtx = need;
      say("ctx", `document is ${m[1]} tokens: context raised to ${maxCtx}`);
    }
  }
}

// -- describing a plan / result in plain lines ------------------------------------------------------
function describePlan(plan: Plan): void {
  const L = plan.inputs.model.L, dev = plan.inputs.device;
  const where = plan.mode === "local" ? `local: all ${L} blocks, lm_head and sampling in this browser` : plan.mode === "server" ? "server: N = 0, the token ids go to the server" : `split at N = ${plan.N} of ${L}: blocks 0–${plan.N - 1} here, the hidden state at boundary ${plan.N} goes to the server`;
  say("mode", `mode: ${where}`);
  say("why", `why: ${plan.reasons.join("; ")}`);
  const cached = dev.cachedBlocks > 0 || dev.embedCached ? `${dev.embedCached ? "embedding" : ""}${dev.embedCached && dev.cachedBlocks > 0 ? " and " : ""}${dev.cachedBlocks > 0 ? `${dev.cachedBlocks} of ${L} blocks` : ""} already cached in this browser` : "nothing cached in this browser: the blocks the plan needs stream from the CDN";
  say("cached", `cached: ${cached}`);
  if (plan.estimate.msPerToken !== null) say("est", `estimate: ${f(plan.estimate.msPerToken)} ms/token, server share ${(plan.estimate.serverShare * 100).toFixed(0)} %`);
}

function privacyLine(r: { mode: string; N: number; privacy: PlanCandidate["privacy"] }): string {
  if (r.mode === "local") return "privacy: nothing left this device (local mode)";
  if (r.N === 0) return "privacy: the server received the token ids themselves (N = 0), so it saw the whole document";
  if (!r.privacy) return `privacy band at boundary ${r.N}: not measured for this model yet`;
  return `privacy band at boundary ${r.N}: a server could recover ${(r.privacy.linear500kTop1 * 100).toFixed(1)} % of the input tokens from the hidden state it received (linear probe with a 500k-token training set on held-out WikiText, Phase 4)`;
}

function costLine(r: RunResult): string {
  const { serverShare, costPer1M, rate } = r.costEstimate;
  const cost = costPer1M === null || !rate || rate.usdPerHour === null ? "no rate stated (bench/rates.json has none), so share only" : `$${costPer1M.toFixed(3)} per 1M tokens at $${rate.usdPerHour}/h on ${rate.gpu}`;
  return `cost: server share ${(serverShare * 100).toFixed(0)} % of an all-server run; ${cost}`;
}

function describeResult(r: RunResult): void {
  say("done", `done: ${r.tokens.length} tokens at ${f(r.tokPerS)} tok/s (server busy ${f(r.serverBusyMs, 0)} ms over the session)`);
  say("cost", costLine(r));
  say("privacy", privacyLine(r));
  say("telemetry", r.telemetry.report ? `telemetry: session report sent (HTTP ${r.telemetry.status ?? "?"}), no prompt content` : "telemetry: not sent");
  if (r.load) say("load", `loaded: ${(r.load.bytes / 1e6).toFixed(0)} MB in ${(r.load.ms / 1000).toFixed(1)} s (${r.load.cacheHits} of ${r.load.chunks} chunks from cache)`);
}

// -- run ---------------------------------------------------------------------------------------------
let last: { model: string; text: string; result: RunResult } | null = null;

async function summarise(): Promise<void> {
  const text = doc.value.trim(), model = modelSel.value;
  if (!text || !model) return;
  runBtn.disabled = cmpBtn.disabled = true;
  clearStatus(); output.textContent = ""; compareOut.replaceChildren(); resultJson.textContent = "";
  say("plan", `planning: measuring this device and asking ${API} for the server's load…`);
  try {
    const r = await runOnce(model, text, {
      onPlan: (plan) => { lines.get("plan")?.remove(); lines.delete("plan"); describePlan(plan); },
      onProgress: (p) => say("load", `loading ${p.group}: ${(p.bytesDone / 1e6).toFixed(0)} MB at ${(p.bytesPerSecond / 1e6).toFixed(0)} MB/s`),
      onToken: (t) => { output.textContent += t; },
    });
    describeResult(r);
    last = { model, text, result: r };
    resultJson.textContent = JSON.stringify({ model, mode: r.mode, N: r.N, tokens: r.tokens.length, tokPerS: r.tokPerS, serverShare: r.costEstimate.serverShare, costPer1M: r.costEstimate.costPer1M, privacy: r.privacy, telemetryStatus: r.telemetry.status, reasons: r.plan.reasons, promptTokens: r.promptTokens });
    cmpBtn.disabled = false;
  } catch (e) {
    say("error", `error: ${(e as Error).message}`, "warn");
  } finally {
    runBtn.disabled = false;
  }
}

/** The other two modes, when the plan's candidates say they are feasible; side by side with the first run. */
async function compare(): Promise<void> {
  if (!last) return;
  const { model, text, result } = last;
  runBtn.disabled = cmpBtn.disabled = true;
  const cands = result.plan.candidates;
  const server = cands.find((c) => c.mode === "server") ?? null;
  const local = cands.find((c) => c.mode === "local") ?? null;
  const splits = cands.filter((c) => c.mode === "split" && c.feasible);
  const splitC = splits.length ? splits[splits.length - 1]! : cands.filter((c) => c.mode === "split").at(-1) ?? null;
  const cols: { title: string; cand: PlanCandidate | null; force: RunOptions["force"] }[] = [
    { title: "server (N = 0)", cand: server, force: { N: 0 } },
    { title: splitC ? `split (N = ${splitC.N})` : "split", cand: splitC, force: splitC ? { N: splitC.N } : null },
    { title: "local", cand: local, force: { mode: "local" } },
  ];
  compareOut.replaceChildren();
  const cards = cols.map((col) => {
    const card = el("div", undefined, "card panel");
    const body = el("div", "", "output"), meta = el("div", "queued", "meta");
    card.append(el("h3", col.title), body, meta);
    compareOut.append(card);
    return { body, meta };
  });
  for (const [i, col] of cols.entries()) {
    const { body, meta } = cards[i]!;
    const same = (col.cand && col.cand.mode === result.mode && col.cand.N === result.N) || (col.cand?.mode === "local" && result.mode === "local");
    if (same) { body.textContent = result.text; meta.textContent = `${f(result.tokPerS)} tok/s · ${cardMeta(result)} (the run above)`; continue; }
    if (!col.cand || !col.cand.feasible || !col.force) { body.textContent = ""; meta.textContent = `not feasible here: ${col.cand?.reasons.join("; ") || "no such candidate"}`; continue; }
    meta.textContent = "running…";
    try {
      const r = await runOnce(model, text, { force: col.force, onToken: (t) => { body.textContent += t; }, onProgress: (p) => { meta.textContent = `loading ${p.group}: ${(p.bytesDone / 1e6).toFixed(0)} MB`; } });
      meta.textContent = `${f(r.tokPerS)} tok/s · ${cardMeta(r)}`;
    } catch (e) { meta.textContent = `error: ${(e as Error).message}`; meta.classList.add("warn"); }
  }
  runBtn.disabled = false; cmpBtn.disabled = false;
}
function cardMeta(r: RunResult): string {
  const cost = r.costEstimate.costPer1M === null ? "" : ` · $${r.costEstimate.costPer1M.toFixed(3)}/1M`;
  return `server share ${(r.costEstimate.serverShare * 100).toFixed(0)} %${cost} · ${privacyLine(r).replace(/^privacy( band)?( at boundary \d+)?: /, "")}`;
}

// -- wiring ------------------------------------------------------------------------------------------
runBtn.onclick = () => { void summarise(); };
cmpBtn.onclick = () => { void compare(); };
doc.addEventListener("dragover", (e) => { e.preventDefault(); doc.classList.add("drop"); });
doc.addEventListener("dragleave", () => doc.classList.remove("drop"));
doc.addEventListener("drop", async (e) => {
  e.preventDefault(); doc.classList.remove("drop");
  const file = e.dataTransfer?.files[0];
  if (file) doc.value = await file.text();
});
modelSel.onchange = () => { cmpBtn.disabled = true; };
void fillModels();
