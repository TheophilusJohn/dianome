// Bench page: loads a model through the SDK into the runtime for a given N, runs prefill and decode, reports
// time to first block, total load time, prefill tok/s, decode tok/s (per T=1 kernel), hidden-state export
// time, GPU buffer bytes and adapter info. `?api=&cdn=` override the endpoints (local: serve-store.mjs on 8788).
import { Dianome, type ModelManifest, type VariantName } from "dianome";
import { Runtime, acquireDevice, loadTokenizer, type AcquiredDevice, type Tokenizer } from "dianome-runtime";

const params = new URLSearchParams(location.search);
const api = params.get("api") ?? undefined;
const cdn = params.get("cdn") ?? undefined;
const form = document.getElementById("f") as HTMLFormElement;
const status = document.getElementById("status")!;
const log = document.getElementById("log")!;
const tbody = document.querySelector("#out tbody")!;
const btnLoad = document.getElementById("load") as HTMLButtonElement;
const btnRun = document.getElementById("run") as HTMLButtonElement;
const btnBoth = document.getElementById("both") as HTMLButtonElement;

const field = (n: string): string => (form.elements.namedItem(n) as HTMLInputElement | HTMLSelectElement | HTMLTextAreaElement).value;
const say = (s: string) => { status.textContent = s; };
const note = (s: string) => { log.textContent += s + "\n"; };
const row = (k: string, v: string) => { const tr = document.createElement("tr"); tr.innerHTML = `<th>${k}</th><td>${v}</td>`; tbody.append(tr); };
const fmt = (n: number, d = 1) => n.toFixed(d);

let gpu: AcquiredDevice | null = null;
let manifest: ModelManifest | null = null;
let tokenizer: Tokenizer | null = null;
let loaded: { rt: Runtime; variant: VariantName; N: number; matvec: string; round16: boolean; bytes: number; loadMs: number } | null = null;

async function makeRuntime(matvec: string, round16: boolean): Promise<Runtime> {
  const d = new Dianome({ ...(api ? { api } : {}), ...(cdn ? { cdn } : {}), telemetry: false });
  gpu ??= await acquireDevice();
  const model = field("model"), variant = field("variant") as VariantName, N = Number(field("N")), maxCtx = Number(field("maxCtx"));
  manifest ??= (await d.manifest(model)) as ModelManifest;
  tokenizer ??= await loadTokenizer(d.cdn, manifest);
  const rt = await Runtime.create(gpu.device, manifest, variant, N, maxCtx, { round16, ...(matvec === "tiled" ? { forceTiled: true } : matvec === "auto" ? {} : { matvec: matvec as "seq" | "lanes" }) });
  matvec = matvec === "auto" ? rt.ops.matvecMode() : matvec;
  let bytes = 0;
  const t0 = performance.now();
  await rt.loadFrom(d.stream(model, { variant, onProgress: (p) => { bytes = p.bytesDone; say(`loading ${variant} N=${N}: ${(p.bytesDone / 1e6).toFixed(0)} MB, ${p.group}, ${(p.bytesPerSecond / 1e6).toFixed(0)} MB/s`); } }));
  const st = rt.stats();
  loaded = { rt, variant, N, matvec, round16, bytes, loadMs: performance.now() - t0 };
  say(`loaded ${variant} N=${N} (${matvec}, round16 ${round16 ? "on" : "off"}): ${(bytes / 1e6).toFixed(0)} MB in ${fmt(loaded.loadMs, 0)} ms, first block after ${fmt(st.timeToFirstBlockMs ?? 0, 0)} ms, GPU ${(st.gpuBytes / 1e6).toFixed(0)} MB`);
  return rt;
}

async function measure(rt: Runtime, label: string): Promise<{ prefillTokPerS: number; decodeTokPerS: number; decodeMs: number; exportMs: number; T: number }> {
  const ids = tokenizer!.encode(field("prompt"));
  const steps = Number(field("steps"));
  rt.reset();
  await rt.prefill(ids); // warm-up (pipelines, bind groups)
  rt.reset();
  await rt.prefill(ids);
  const prefillMs = rt.stats().lastPrefillMs;
  const times: number[] = [];
  let pos = ids.length;
  for (let i = 0; i < steps; i++) { await rt.decode(ids[i % ids.length]!, pos++); times.push(rt.stats().lastDecodeMs); }
  times.sort((a, b) => a - b);
  const decodeMs = times[Math.floor(times.length / 2)]!;
  const hidden = await rt.exportHidden();
  const exportMs = rt.stats().lastExportMs;
  note(`${label}: T=${ids.length} prefill ${fmt(prefillMs)} ms (${fmt((ids.length / prefillMs) * 1000, 0)} tok/s), decode median ${fmt(decodeMs, 2)} ms (${fmt(1000 / decodeMs)} tok/s) over ${steps} steps, export ${hidden.length * 2} bytes in ${fmt(exportMs, 2)} ms`);
  return { prefillTokPerS: (ids.length / prefillMs) * 1000, decodeTokPerS: 1000 / decodeMs, decodeMs, exportMs, T: ids.length };
}

btnLoad.onclick = async () => {
  btnLoad.disabled = true; btnRun.disabled = true; btnBoth.disabled = true;
  try {
    loaded?.rt.destroy();
    loaded = null;
    tbody.innerHTML = "";
    const rt = await makeRuntime(field("matvec"), field("round16") === "1");
    const st = rt.stats();
    const info = gpu!.info;
    row("adapter", `${info.vendor} ${info.architecture} ${info.device} ${info.description}`.trim() + ` (shader-f16 ${info.shaderF16 ? "yes" : "no"}, subgroups ${info.subgroups ? "yes" : "no"}, maxBufferSize ${(info.limits.maxBufferSize / 2 ** 20).toFixed(0)} MiB)`);
    row("model / variant / N", `${field("model")} / ${loaded!.variant} / ${loaded!.N}`);
    row("time to first block", `${fmt(st.timeToFirstBlockMs ?? 0, 0)} ms`);
    row("total load time", `${fmt(loaded!.loadMs, 0)} ms for ${(loaded!.bytes / 2 ** 20).toFixed(1)} MiB`);
    row("GPU buffer bytes", `${(st.gpuBytes / 2 ** 20).toFixed(1)} MiB (weights ${(st.weightBytes / 2 ** 20).toFixed(1)}, KV ${(st.kvBytes / 2 ** 20).toFixed(1)}, workspace ${(st.workspaceBytes / 2 ** 20).toFixed(1)})`);
    btnRun.disabled = false; btnBoth.disabled = false;
  } catch (e) { say(`error: ${(e as Error).message}`); note(String((e as Error).stack ?? e)); }
  btnLoad.disabled = false;
};

btnRun.onclick = async () => {
  if (!loaded) return;
  btnRun.disabled = true; btnBoth.disabled = true;
  try {
    const m = await measure(loaded.rt, `${loaded.variant} N=${loaded.N} ${loaded.matvec}`);
    row(`prefill tok/s (${loaded.matvec})`, `${fmt(m.prefillTokPerS, 0)} (T=${m.T})`);
    row(`decode tok/s (${loaded.matvec})`, `${fmt(m.decodeTokPerS)} (median ${fmt(m.decodeMs, 2)} ms/token)`);
    row("hidden-state export", `${fmt(m.exportMs, 2)} ms`);
    say("done");
  } catch (e) { say(`error: ${(e as Error).message}`); note(String((e as Error).stack ?? e)); }
  btnRun.disabled = false; btnBoth.disabled = false;
};

btnBoth.onclick = async () => {
  if (!loaded) return;
  btnRun.disabled = true; btnBoth.disabled = true;
  try {
    for (const matvec of ["seq", "lanes", "tiled"]) {
      let rt = loaded.rt;
      if (loaded.matvec !== matvec) { loaded.rt.destroy(); rt = await makeRuntime(matvec, loaded.round16); }
      const m = await measure(rt, `${loaded.variant} N=${loaded.N} ${matvec}`);
      row(`decode tok/s (${matvec})`, `${fmt(m.decodeTokPerS)} (median ${fmt(m.decodeMs, 2)} ms/token)`);
    }
    say("done");
  } catch (e) { say(`error: ${(e as Error).message}`); note(String((e as Error).stack ?? e)); }
  btnRun.disabled = false; btnBoth.disabled = false;
};

note(`api ${api ?? "(default)"} cdn ${cdn ?? "(default)"}; WebGPU ${navigator.gpu ? "available" : "NOT available in this browser"}`);
