// One-time device microbench (Phase 5b): ms per block at T = 1 and T = 32, client lm_head ms and export ms for a
// variant on this device, using synthetic weights of the variant's exact kinds and shapes (no download needed,
// so the planner can run before anything is fetched). Median of `runs`. Callers cache the result (the SDK keys it
// by adapter + model + variant in localStorage).

import type { ModelManifest, VariantName } from "dianome";
import { forwardBlock, type BlockWeights, type ModelConfig, type Workspace } from "./block";
import { configFromManifest } from "./index";
import { createStorage, uploadWeight, type WeightBuffer } from "./gpu/buffers";
import { GpuOps, type GpuTensor } from "./gpu/ops";
import { createKv, type GpuKv } from "./kv";
import { LmHead, type HeadMatvec } from "./lmhead";
import { q4Bytes, q4Encode, q8Bytes, q8Encode } from "./quant";
import { readback } from "./gpu/buffers";

export interface Microbench {
  model: string;
  variant: string;
  msPerBlockT1: number;
  msPerBlockT32: number;
  lmHeadMs: number;
  exportMs: number;
  runs: number;
  T: number;
  reps: number;
  matvec: string;
  headMatvec: string;
  measuredAt: string;
  /** Per-run values (ms for one block / one head / one export). */
  samples: { t1: number[]; t32: number[]; head: number[]; export: number[] };
}

export interface MicrobenchOptions {
  runs?: number;
  T?: number;
  /** Block repetitions per timed submit (default 8): amortises the fixed submit + completion overhead that a real
   * forward pays once for N blocks, so ms/block is what one more block costs inside a pass. */
  reps?: number;
  maxCtx?: number;
  round16?: boolean;
  matvec?: "seq" | "lanes";
  headMatvec?: HeadMatvec | "auto";
  /** Skip the lm_head measurement (saves the vocab-sized upload). */
  skipHead?: boolean;
}

const now = (): number => performance.now();
const median = (a: number[]): number => { const s = [...a].sort((x, y) => x - y); return s[Math.floor(s.length / 2)] ?? 0; };

function rng(seed: number): () => number {
  let s = seed >>> 0;
  return () => { s = (s + 0x6d2b79f5) >>> 0; let t = s; t = Math.imul(t ^ (t >>> 15), t | 1); t ^= t + Math.imul(t ^ (t >>> 7), t | 61); return ((t ^ (t >>> 14)) >>> 0) / 4294967296; };
}

type Kind = "fp16" | "q8" | "q4";

/** A random weight of `kind` and `shape`, uploaded like an entry would be. */
export function syntheticWeight(device: GPUDevice, name: string, kind: Kind, shape: number[], scale: number, seed: number, tally?: { bytes: number }): WeightBuffer {
  const n = shape.reduce((a, b) => a * b, 1);
  const r = rng(seed);
  if (kind === "fp16") {
    const u16 = new Uint16Array(n);
    // random fp16 bit patterns in a sane range: sign + exponent around 2^-5..2^0
    for (let i = 0; i < n; i++) { const e = 10 + Math.floor(r() * 5); u16[i] = ((r() < 0.5 ? 0x8000 : 0) | (e << 10) | Math.floor(r() * 1024)) >>> 0; }
    return uploadWeight(device, { name, shape, storage: { kind: "fp16" }, bytes: new Uint8Array(u16.buffer) }, tally);
  }
  const [out, inn] = [shape[0]!, shape[1] ?? 1];
  const w = new Float32Array(n);
  for (let i = 0; i < n; i++) w[i] = (r() * 2 - 1) * scale;
  if (kind === "q8") { const q = q8Encode(w, out, inn); const { bytes, parts } = q8Bytes(q); return uploadWeight(device, { name, shape, storage: { kind: "q8", parts }, bytes }, tally); }
  const q = q4Encode(w, out, inn); const { bytes, parts } = q4Bytes(q);
  return uploadWeight(device, { name, shape, storage: { kind: "q4", group_size: 128, parts }, bytes }, tally);
}

const ROLES: Record<keyof BlockWeights<unknown>, string> = {
  inputNorm: "input_layernorm.weight", q: "attn.q_proj.weight", qBias: "attn.q_proj.bias", k: "attn.k_proj.weight", kBias: "attn.k_proj.bias",
  v: "attn.v_proj.weight", vBias: "attn.v_proj.bias", o: "attn.o_proj.weight", postNorm: "post_attention_layernorm.weight",
  gate: "mlp.gate_proj.weight", up: "mlp.up_proj.weight", down: "mlp.down_proj.weight",
};

/** Block 0's entries of `variant` reproduced with random contents (same kinds and shapes). */
export function syntheticBlockWeights(device: GPUDevice, manifest: ModelManifest, variant: VariantName, tally?: { bytes: number }): BlockWeights<WeightBuffer> {
  const g = manifest.variants[variant]?.groups.find((x) => x.name === "layer.0");
  if (!g) throw new Error(`manifest has no layer.0 for ${variant}`);
  const w = {} as BlockWeights<WeightBuffer>;
  let seed = 1;
  for (const key of Object.keys(ROLES) as (keyof BlockWeights<unknown>)[]) {
    const e = g.entries.find((x) => x.role === ROLES[key]);
    if (!e) throw new Error(`layer.0 has no ${ROLES[key]}`);
    const kind = e.storage.kind as Kind;
    w[key] = syntheticWeight(device, `mb.${key}`, kind, e.shape, e.shape.length === 1 ? 1 : 0.05, seed++, tally);
  }
  return w;
}

export async function microbench(device: GPUDevice, manifest: ModelManifest, variant: VariantName, opts: MicrobenchOptions = {}): Promise<Microbench> {
  const runs = opts.runs ?? 5, T = opts.T ?? 32, reps = Math.max(1, opts.reps ?? 8), maxCtx = opts.maxCtx ?? Math.max(64, T + runs + 2);
  const cfg: ModelConfig = configFromManifest(manifest);
  const round16 = opts.round16 ?? true;
  const ops = new GpuOps(device, cfg, { maxCtx, round16, ...(opts.matvec ? { matvec: opts.matvec } : {}) });
  const tally = { bytes: 0 };
  const w = syntheticBlockWeights(device, manifest, variant, tally);
  const { d, heads, kvHeads, headDim, inter } = cfg;
  const mk = (label: string, cols: number): GpuTensor => ops.tensor(label, maxCtx, cols);
  const ws: Workspace<GpuTensor> = {
    x: mk("mb.x", d), x2: mk("mb.x2", d), h: mk("mb.h", d), q: mk("mb.q", heads * headDim), k: mk("mb.k", kvHeads * headDim), v: mk("mb.v", kvHeads * headDim),
    attn: mk("mb.attn", heads * headDim), o: mk("mb.o", d), gate: mk("mb.gate", inter), up: mk("mb.up", inter), act: mk("mb.act", inter), mlp: mk("mb.mlp", d),
  };
  const kv: GpuKv = createKv(device, cfg, maxCtx, "mb.kv");
  const x = new Float32Array(T * d);
  const r = rng(99);
  for (let i = 0; i < x.length; i++) x[i] = (r() * 2 - 1);
  ops.write(ws.x, x);
  const sync = () => device.queue.onSubmittedWorkDone();

  const timed = async (f: () => void): Promise<number> => { const t0 = now(); ops.run(f); await sync(); return now() - t0; };
  // `reps` blocks per submit (the same weights and KV rows each time; only the timing matters), divided by reps.
  const block = (Tr: number, pos0: number) => () => { for (let r = 0; r < reps; r++) forwardBlock(ops, w, ws, kv, Tr, pos0); };
  // warm-up (pipelines, bind groups)
  await timed(block(T, 0));
  await timed(block(1, T));
  const t32: number[] = [];
  for (let i = 0; i < runs; i++) t32.push((await timed(block(T, 0))) / reps);
  const t1: number[] = [];
  for (let i = 0; i < runs; i++) t1.push((await timed(block(1, T + i))) / reps);

  // export: pack one row to fp16 + read it back
  const exp: number[] = [];
  const exportBuf = createStorage(device, "mb.export", d * 2);
  for (let i = 0; i < runs + 1; i++) {
    const t0 = now();
    ops.run(() => ops.packF16(ws.x, 1, exportBuf));
    await readback(device, exportBuf, d * 2);
    if (i > 0) exp.push(now() - t0);
  }
  exportBuf.destroy();

  // lm_head: final norm + [vocab, d] matvec of the lm_head entry's kind + readback of the logits
  const head: number[] = [];
  let headMatvec = "skipped";
  if (!opts.skipHead) {
    const lg = manifest.variants[variant]!.groups.find((g) => g.name === "lm_head");
    const le = lg?.entries.find((e) => e.role === "lm_head.weight");
    const fg = manifest.variants[variant]!.groups.find((g) => g.name === "final_norm");
    const fe = fg?.entries.find((e) => e.role === "norm.weight");
    if (le && fe) {
      const hw = syntheticWeight(device, "mb.lm_head", le.storage.kind as Kind, le.shape, 0.05, 7, tally);
      const nw = syntheticWeight(device, "mb.final_norm", "fp16", fe.shape, 1, 8, tally);
      const lm = new LmHead(ops, nw, hw, opts.headMatvec ?? "auto");
      headMatvec = opts.headMatvec && opts.headMatvec !== "auto" ? opts.headMatvec : ops.matvecMode();
      for (let i = 0; i < runs + 1; i++) {
        const t0 = now();
        ops.run(() => lm.record(ws.x, 1));
        await lm.read();
        if (i > 0) head.push(now() - t0);
      }
      lm.destroy();
    }
  }
  for (const b of Object.values(w)) (b as WeightBuffer).buffer.destroy();
  for (const t of Object.values(ws)) (t as GpuTensor).buffer.destroy();
  kv.k.destroy(); kv.v.destroy(); ops.destroy();
  return {
    model: manifest.id, variant, msPerBlockT1: median(t1), msPerBlockT32: median(t32), lmHeadMs: median(head), exportMs: median(exp),
    runs, T, reps, matvec: ops.matvecMode(), headMatvec, measuredAt: new Date().toISOString(), samples: { t1, t32, head, export: exp },
  };
}
