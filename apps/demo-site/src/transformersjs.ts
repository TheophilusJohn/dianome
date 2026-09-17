import { env, pipeline } from "@huggingface/transformers";
import { dianomeCache } from "dianome/transformersjs";
import { CDN, el, fmtBytes, makeDianome, requestCounter } from "./common";

const ARTIFACT = new URLSearchParams(location.search).get("artifact") ?? "qwen2.5-0.5b-instruct-onnx";
const HF_MODEL = "onnx-community/Qwen2.5-0.5B-Instruct";
el("artifact").textContent = ARTIFACT;
const log = el("log");
const say = (s: string) => { log.textContent += (log.textContent === "idle" ? "" : "\n") + s; if (log.textContent.startsWith("idle")) log.textContent = s; };

el("run").addEventListener("click", async () => {
  el<HTMLButtonElement>("run").disabled = true;
  log.textContent = "";
  const d = makeDianome({ telemetry: false });
  const served: string[] = [];
  env.useCustomCache = true;
  env.customCache = dianomeCache(d, ARTIFACT, { onFile: (f, source) => { served.push(f); say(`served ${f} from Dianome (${source})`); } });
  env.allowLocalModels = false;
  (env.backends.onnx as { wasm?: { proxy?: boolean; numThreads?: number } }).wasm ??= {};
  const chunkCount = requestCounter(`${CDN}/chunks/`);
  const hfCount = requestCounter("https://huggingface.co/");
  const t0 = performance.now();
  try {
    const pipe = await pipeline("text-generation", HF_MODEL, { dtype: "q4", device: "wasm", progress_callback: (p: { status: string; file?: string; progress?: number }) => { if (p.status === "progress" && p.file && (p.progress ?? 0) >= 100) say(`transformers.js: ${p.file} 100%`); } });
    say(`pipeline ready in ${((performance.now() - t0) / 1000).toFixed(1)} s`);
    const out = await pipe("The capital of France is", { max_new_tokens: 8, do_sample: false });
    say(`generated: ${JSON.stringify(out)}`);
  } catch (e) {
    say(`error: ${(e as Error).message}`);
  }
  const st = await d.cache.status();
  say(`chunk requests from this page: ${chunkCount()} · huggingface.co requests: ${hfCount()} · files served by adapter: ${served.length}`);
  say(`dianome cache: ${st.mode}, ${st.chunks} chunks, ${fmtBytes(st.usageBytes)}`);
  say(`second run should show: chunk requests 0 (all from ${st.mode} cache), huggingface.co requests 0 for model files`);
  el<HTMLButtonElement>("run").disabled = false;
});
el("clear").addEventListener("click", async () => {
  await makeDianome().cache.clear();
  await caches.delete("transformers-cache");
  log.textContent = "cleared dianome-v1 and transformers-cache";
});
