import { CreateMLCEngine, hasModelInCache, prebuiltAppConfig } from "@mlc-ai/web-llm";
import { prewarmWebLLM } from "dianome/webllm";
import { CDN, el, fmtBytes, makeDianome, requestCounter } from "./common";

const ARTIFACT = new URLSearchParams(location.search).get("artifact") ?? "qwen2.5-0.5b-instruct-mlc";
const MODEL_ID = "Qwen2.5-0.5B-Instruct-q4f16_1-MLC";
// The model URL is only a cache key (WebLLM turns it into `${MODEL_URL}resolve/main/<file>`): every file under
// it is pre-warmed, so WebLLM never fetches from it.
const MODEL_URL = `${CDN}/mlc/${ARTIFACT}/`;
el("artifact").textContent = ARTIFACT;
const log = el("log");
const say = (s: string) => { log.textContent = log.textContent === "idle" ? s : `${log.textContent}\n${s}`; };

const prebuilt = prebuiltAppConfig.model_list.find((m) => m.model_id === MODEL_ID);
const appConfig = { ...prebuiltAppConfig, model_list: prebuilt ? [{ ...prebuilt, model: MODEL_URL }] : [] };

el("prewarm").addEventListener("click", async () => {
  log.textContent = "";
  const d = makeDianome({ telemetry: false });
  const chunkCount = requestCounter(`${CDN}/chunks/`);
  const t0 = performance.now();
  const r = await prewarmWebLLM(d, ARTIFACT, { modelUrl: MODEL_URL, onFile: (f, b) => say(`cached ${f} (${fmtBytes(b)}) under ${MODEL_URL}${f}`) });
  say(`pre-warm done in ${((performance.now() - t0) / 1000).toFixed(1)} s: ${r.files.length} files streamed (${fmtBytes(r.bytes)}, source ${r.source}), ${r.skipped.length} already cached · chunk requests ${chunkCount()}`);
  say(`webllm hasModelInCache: ${await hasModelInCache(MODEL_ID, appConfig)}`);
});

el("run").addEventListener("click", async () => {
  el<HTMLButtonElement>("run").disabled = true;
  const shardCount = requestCounter(`${MODEL_URL}resolve/main/`);
  const t0 = performance.now();
  try {
    if (!prebuilt) throw new Error(`${MODEL_ID} not in WebLLM's prebuilt list`);
    const engine = await CreateMLCEngine(MODEL_ID, { appConfig, initProgressCallback: (p) => { log.textContent = log.textContent.replace(/\nwebllm: .*$/, "") + `\nwebllm: ${p.text}`; } });
    say(`engine ready in ${((performance.now() - t0) / 1000).toFixed(1)} s`);
    const reply = await engine.chat.completions.create({ messages: [{ role: "user", content: "Say hello in five words." }], max_tokens: 24 });
    say(`reply: ${reply.choices[0]?.message.content ?? ""}`);
  } catch (e) {
    say(`error: ${(e as Error).message}`);
  }
  say(`requests to the model URL from this page: ${shardCount()} (expected 0 after pre-warm; wasm lib comes from WebLLM's CDN)`);
  el<HTMLButtonElement>("run").disabled = false;
});

el("clear").addEventListener("click", async () => {
  await makeDianome().cache.clear();
  for (const c of ["webllm/model", "webllm/config"]) await caches.delete(c);
  log.textContent = "cleared dianome-v1, webllm/model, webllm/config";
});
