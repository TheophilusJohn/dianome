import { describe, expect, it } from "vitest";
import { dianomeCache, keyPath, resolveFile } from "../src/adapters/transformersjs";
import { cacheTargets, isWebLLMFile, normalizeModelUrl, prewarmWebLLM } from "../src/adapters/webllm";
import { Dianome } from "../src/index";
import { synthFiles } from "./fixtures/synth";
import { FakeCacheStorage } from "./helpers/fakeCaches";
import { FakeFetch } from "./helpers/fakeFetch";

const API = "https://api.test", CDN = "https://cdn.test";
const ONNX_FILES = ["config.json", "generation_config.json", "onnx/model.onnx", "onnx/model_q4.onnx", "onnx/model_q4f16.onnx", "tokenizer.json", "tokenizer_config.json"];

describe("dianome/transformersjs path mapping", () => {
  it("maps Transformers.js cache keys (local path and remote URL) to artifact files on path boundaries", () => {
    expect(keyPath("https://huggingface.co/onnx-community/Qwen2.5-0.5B-Instruct/resolve/main/onnx/model_q4.onnx")).toBe("/onnx-community/Qwen2.5-0.5B-Instruct/resolve/main/onnx/model_q4.onnx");
    expect(keyPath("/models/onnx-community/Qwen2.5-0.5B-Instruct/config.json")).toBe("/models/onnx-community/Qwen2.5-0.5B-Instruct/config.json");
    expect(keyPath(new Request("https://x.test/a%20b/tokenizer.json"))).toBe("/a b/tokenizer.json");
    expect(resolveFile("/models/m/onnx/model_q4.onnx", ONNX_FILES)).toBe("onnx/model_q4.onnx");
    expect(resolveFile("/m/resolve/main/onnx/model_q4f16.onnx", ONNX_FILES)).toBe("onnx/model_q4f16.onnx");
    expect(resolveFile("/m/generation_config.json", ONNX_FILES)).toBe("generation_config.json"); // not config.json
    expect(resolveFile("/m/xconfig.json", ONNX_FILES)).toBeNull();
    expect(resolveFile("/m/onnx/model_q4.onnx_data", ONNX_FILES)).toBeNull();
    expect(resolveFile("config.json", ONNX_FILES)).toBe("config.json");
  });

  it("match() serves artifact files from Dianome with Content-Length, and other keys from the fallback cache; put() only stores foreign keys", async () => {
    const files = await synthFiles([{ name: "config.json", length: 40, seed: 1 }, { name: "onnx/model_q4.onnx", length: 9000, seed: 2 }], { runtime: "transformersjs", chunkSize: 4096, id: "onnx-art" });
    const ff = new FakeFetch(API, CDN, files.chunks, new Map([[files.manifest.id, files.manifest]]));
    const d = new Dianome({ api: API, cdn: CDN, fetch: ff.fetch, cache: "none", telemetry: false });
    const fb = new FakeCacheStorage();
    const served: string[] = [];
    const cache = dianomeCache(d, "onnx-art", { onFile: (f) => served.push(f) });
    (globalThis as { caches?: unknown }).caches = fb.asCacheStorage();
    try {
      const res = await cache.match("https://huggingface.co/onnx-community/Qwen2.5-0.5B-Instruct/resolve/main/onnx/model_q4.onnx");
      expect(res?.status).toBe(200);
      expect(res?.headers.get("content-length")).toBe("9000");
      expect(new Uint8Array(await res!.arrayBuffer())).toEqual(files.raw.get("onnx/model_q4.onnx"));
      expect(served).toEqual(["onnx/model_q4.onnx"]);
      expect(ff.chunkRequests()).toBe(3);
      expect(await cache.match("/models/onnx-community/Qwen2.5-0.5B-Instruct/tokenizer.json")).toBeUndefined();
      await cache.put("/models/x/tokenizer.json", new Response("{}"));
      expect(await (await cache.match("/models/x/tokenizer.json"))!.text()).toBe("{}");
      await cache.put("/models/x/onnx/model_q4.onnx", new Response("should not be stored"));
      expect(fb.caches.get("transformers-cache")!.entries.size).toBe(1);
      expect(await cache.files()).toEqual(["config.json", "onnx/model_q4.onnx"]);
    } finally {
      delete (globalThis as { caches?: unknown }).caches;
    }
  });
});

describe("dianome/webllm", () => {
  it("computes WebLLM's cache keys", () => {
    // WebLLM's cleanModelUrl(): "https://huggingface.co/USER/MODEL" -> ".../MODEL/resolve/main/"
    expect(normalizeModelUrl("https://cdn.test/mlc/q")).toBe("https://cdn.test/mlc/q/resolve/main/");
    expect(normalizeModelUrl("https://cdn.test/mlc/q/")).toBe("https://cdn.test/mlc/q/resolve/main/");
    expect(normalizeModelUrl("https://huggingface.co/mlc-ai/X-MLC/resolve/abc123/")).toBe("https://huggingface.co/mlc-ai/X-MLC/resolve/abc123/");
    expect(cacheTargets("params_shard_0.bin", "https://cdn.test/mlc/q")).toEqual([{ cache: "webllm/model", url: "https://cdn.test/mlc/q/resolve/main/params_shard_0.bin" }]);
    expect(cacheTargets("mlc-chat-config.json", "https://cdn.test/mlc/q/")).toEqual([{ cache: "webllm/config", url: "https://cdn.test/mlc/q/resolve/main/mlc-chat-config.json" }, { cache: "webllm/model", url: "https://cdn.test/mlc/q/resolve/main/mlc-chat-config.json" }]);
    expect(isWebLLMFile("README.md")).toBe(false);
    expect(isWebLLMFile(".gitattributes")).toBe(false);
    expect(isWebLLMFile("ndarray-cache.json")).toBe(true);
  });

  it("pre-warms every artifact file under the exact URLs WebLLM requests, and skips files already cached", async () => {
    const files = await synthFiles([
      { name: "mlc-chat-config.json", length: 30, seed: 1 }, { name: "ndarray-cache.json", length: 50, seed: 2 }, { name: "tensor-cache.json", length: 50, seed: 2 },
      { name: "params_shard_0.bin", length: 10_000, seed: 3 }, { name: "params_shard_1.bin", length: 300, seed: 4 }, { name: "tokenizer.json", length: 20, seed: 5 }, { name: "README.md", length: 5, seed: 6 },
    ], { runtime: "webllm", chunkSize: 4096, id: "mlc-art" });
    const ff = new FakeFetch(API, CDN, files.chunks, new Map([[files.manifest.id, files.manifest]]));
    const d = new Dianome({ api: API, cdn: CDN, fetch: ff.fetch, cache: "none", telemetry: false });
    const cs = new FakeCacheStorage();
    const modelUrl = "https://cdn.test/mlc/qwen";
    const r = await prewarmWebLLM(d, "mlc-art", { modelUrl, caches: cs.asCacheStorage() });
    expect(r.files).toEqual(["mlc-chat-config.json", "ndarray-cache.json", "tensor-cache.json", "params_shard_0.bin", "params_shard_1.bin", "tokenizer.json"]);
    expect(r.skipped).toEqual([]);
    expect(r.bytes).toBe(30 + 50 + 50 + 10_000 + 300 + 20);
    const model = cs.caches.get("webllm/model")!, config = cs.caches.get("webllm/config")!;
    const base = `${modelUrl}/resolve/main/`;
    expect([...model.entries.keys()].sort()).toEqual(["mlc-chat-config.json", "ndarray-cache.json", "params_shard_0.bin", "params_shard_1.bin", "tensor-cache.json", "tokenizer.json"].map((f) => `${base}${f}`).sort());
    expect([...config.entries.keys()]).toEqual([`${base}mlc-chat-config.json`]);
    const shard = await (await cs.open("webllm/model")).match(new Request(`${base}params_shard_0.bin`));
    expect(new Uint8Array(await shard!.arrayBuffer())).toEqual(files.raw.get("params_shard_0.bin"));
    expect(shard!.headers.get("content-length")).toBe("10000");
    // Second run: nothing to stream, no chunk requests.
    const before = ff.chunkRequests();
    const r2 = await prewarmWebLLM(d, "mlc-art", { modelUrl, caches: cs.asCacheStorage() });
    expect(r2.files).toEqual([]);
    expect(r2.skipped).toHaveLength(6);
    expect(ff.chunkRequests()).toBe(before);
  });
});
