// Browser-side harness for the Playwright gate suite. Everything runs in Chrome against the local static server
// (fixtures under /fixtures, the chunk store with the api/cdn URL shapes on the same origin). Each gate function
// returns plain numbers; the spec asserts the tolerances.

import { Dianome, type ModelManifest, type VariantName } from "dianome";
import { CpuOps, cpuKv, type CpuWeight } from "../../../src/cpu";
import { f16ToF32, f32ToF16, roundF16 } from "../../../src/f16";
import { GpuOps, type GpuTensor } from "../../../src/gpu/ops";
import { uploadWeight, type WeightBuffer } from "../../../src/gpu/buffers";
import { acquireDevice, type AcquiredDevice } from "../../../src/gpu/device";
import { createKv } from "../../../src/kv";
import { parseNpy } from "../../../src/npy";
import { SplitClient } from "../../../src/protocol";
import { q4Bytes, q4Decode, q4Encode, q8Bytes, q8Decode, q8Encode } from "../../../src/quant";
import { ropeTable, type RopeTable } from "../../../src/rope";
import { Runtime, configFromManifest, type RuntimeOptions } from "../../../src/index";
import { SplitSession } from "../../../src/session";
import { argmax, topk } from "../../../src/lmhead";
import { microbench as runMicrobench, type MicrobenchOptions } from "../../../src/microbench";
import { forwardBlock, STAGES, type ModelConfig, type Stage } from "../../../src/block";

const MODEL = "qwen2.5-0.5b-instruct";
const origin = location.origin;

// -- metrics ------------------------------------------------------------------------------

export interface Cmp { maxAbs: number; relRms: number; maxRel: number; n: number; argmax: number; refAtMax: number; gotAtMax: number }

function cmp(got: ArrayLike<number>, ref: ArrayLike<number>): Cmp {
  if (got.length !== ref.length) throw new Error(`length mismatch ${got.length} vs ${ref.length}`);
  let maxAbs = 0, argmax = -1, se = 0, sr = 0, maxRef = 0;
  for (let i = 0; i < got.length; i++) {
    const d = Math.abs(got[i]! - ref[i]!);
    if (d > maxAbs) { maxAbs = d; argmax = i; }
    se += d * d; sr += ref[i]! * ref[i]!;
    maxRef = Math.max(maxRef, Math.abs(ref[i]!));
  }
  return { maxAbs, relRms: sr > 0 ? Math.sqrt(se / sr) : Math.sqrt(se), maxRel: maxRef > 0 ? maxAbs / maxRef : maxAbs, n: got.length, argmax, refAtMax: argmax >= 0 ? ref[argmax]! : 0, gotAtMax: argmax >= 0 ? got[argmax]! : 0 };
}

// -- fixtures ----------------------------------------------------------------------------

const npyCache = new Map<string, Promise<{ shape: number[]; data: Float32Array }>>();
function npy(name: string): Promise<{ shape: number[]; data: Float32Array }> {
  let p = npyCache.get(name);
  if (!p) {
    p = fetch(`/fixtures/${name}`).then(async (r) => { if (!r.ok) throw new Error(`fixture ${name}: ${r.status}`); const n = parseNpy(await r.arrayBuffer()); return { shape: n.shape, data: n.data as Float32Array }; });
    npyCache.set(name, p);
  }
  return p;
}
async function json<T>(path: string): Promise<T> { const r = await fetch(path); if (!r.ok) throw new Error(`${path}: ${r.status}`); return (await r.json()) as T; }

interface Prompt { text: string; token_ids: number[]; T: number }
interface Greedy { steps: number; prompt_T: number; tokens: number[]; positions: number[] }

// -- device / model -------------------------------------------------------------------------

let acquired: AcquiredDevice | null = null;
async function gpu(): Promise<AcquiredDevice> { return (acquired ??= await acquireDevice()); }

let manifestP: Promise<ModelManifest> | null = null;
function manifest(): Promise<ModelManifest> {
  return (manifestP ??= new Dianome({ api: origin, cdn: origin, cache: "none", telemetry: false }).manifest(MODEL) as Promise<ModelManifest>);
}

/** Fixture cos/sin for the first 32 positions, computed for the rest (decode steps go to position 47). */
async function fixtureRope(cfg: ModelConfig, maxCtx: number): Promise<{ rope: RopeTable; cmpCos: Cmp; cmpSin: Cmp }> {
  const [c, s] = await Promise.all([npy("rope_cos.npy"), npy("rope_sin.npy")]);
  const computed = ropeTable(cfg.ropeTheta, cfg.headDim, maxCtx, true);
  const cmpCos = cmp(computed.cos.subarray(0, c.data.length), c.data);
  const cmpSin = cmp(computed.sin.subarray(0, s.data.length), s.data);
  const cos = computed.cos.slice(), sin = computed.sin.slice();
  cos.set(c.data, 0); sin.set(s.data, 0);
  return { rope: { cos, sin, positions: maxCtx, headDim: cfg.headDim }, cmpCos, cmpSin };
}

const runtimes = new Map<string, Promise<Runtime>>();
interface LoadInfo { ms: number; timeToFirstBlockMs: number | null; bytes: number; gpuBytes: number }
const loadInfo = new Map<string, LoadInfo>();

async function runtime(variant: VariantName, N: number, opts: RuntimeOptions & { maxCtx?: number; useFixtureRope?: boolean } = {}): Promise<Runtime> {
  const key = `${variant}:${N}:${JSON.stringify(opts)}`;
  let p = runtimes.get(key);
  if (!p) {
    p = (async () => {
      const { device } = await gpu();
      const m = await manifest();
      const maxCtx = opts.maxCtx ?? 256;
      const cfg = configFromManifest(m);
      const ropeOpt = opts.useFixtureRope ? { rope: (await fixtureRope(cfg, maxCtx)).rope } : {};
      const { maxCtx: _m, useFixtureRope: _u, ...rest } = opts;
      const rt = await Runtime.create(device, m, variant, N, maxCtx, { ...rest, ...ropeOpt });
      const d = new Dianome({ api: origin, cdn: origin, cache: "none", telemetry: false });
      const t0 = performance.now();
      let bytes = 0;
      await rt.loadFrom(d.stream(MODEL, { variant, onProgress: (pr) => { bytes = pr.bytesDone; } }));
      loadInfo.set(key, { ms: performance.now() - t0, timeToFirstBlockMs: rt.stats().timeToFirstBlockMs, bytes, gpuBytes: rt.stats().gpuBytes });
      return rt;
    })();
    runtimes.set(key, p);
  }
  return p;
}

// -- random tensors ------------------------------------------------------------------------

function rng(seed: number): () => number {
  let s = seed >>> 0;
  return () => { s = (s + 0x6d2b79f5) >>> 0; let t = s; t = Math.imul(t ^ (t >>> 15), t | 1); t ^= t + Math.imul(t ^ (t >>> 7), t | 61); return ((t ^ (t >>> 14)) >>> 0) / 4294967296; };
}
function randF16(n: number, scale: number, seed: number): Float32Array {
  const r = rng(seed);
  const out = new Float32Array(n);
  for (let i = 0; i < n; i++) out[i] = roundF16((r() * 2 - 1) * scale);
  return out;
}

function f16Weight(device: GPUDevice, name: string, shape: number[], values: Float32Array): WeightBuffer {
  const u16 = new Uint16Array(values.length);
  for (let i = 0; i < values.length; i++) u16[i] = f32ToF16(values[i]!);
  return uploadWeight(device, { name, shape, storage: { kind: "fp16" }, bytes: new Uint8Array(u16.buffer) });
}

// -- gate 2: unit-level ops ------------------------------------------------------------------

export interface UnitResult { name: string; T: number; kind: string; path: string; cmp: Cmp }

async function unitMatmul(kind: "fp16" | "q8" | "q4", T: number, inn: number, out: number, bias: boolean, seed: number, forceTiled = false, matvec: "lanes" | "seq" = "lanes"): Promise<UnitResult> {
  const { device } = await gpu();
  const cfg: ModelConfig = { d: inn, heads: 1, kvHeads: 1, headDim: 64, inter: out, eps: 1e-6, ropeTheta: 1e6, layers: 1 };
  const ops = new GpuOps(device, cfg, { maxCtx: 64, round16: false, forceTiled, matvec });
  const x = randF16(T * inn, 1, seed), w = randF16(out * inn, 0.05, seed + 1), b = randF16(out, 0.5, seed + 2);
  let wb: WeightBuffer, wref: Float32Array;
  if (kind === "fp16") { wb = f16Weight(device, "w", [out, inn], w); wref = w; }
  else if (kind === "q8") { const q = q8Encode(w, out, inn); const { bytes, parts } = q8Bytes(q); wb = uploadWeight(device, { name: "w", shape: [out, inn], storage: { kind: "q8", parts }, bytes }); wref = q8Decode(q); }
  else { const q = q4Encode(w, out, inn); const { bytes, parts } = q4Bytes(q); wb = uploadWeight(device, { name: "w", shape: [out, inn], storage: { kind: "q4", group_size: 128, parts }, bytes }); wref = q4Decode(q); }
  const bb = bias ? f16Weight(device, "b", [out], b) : null;
  const xt = ops.tensor("x", T, inn), yt = ops.tensor("y", T, out);
  ops.write(xt, x);
  ops.run(() => ops.matmul(xt, wb, bb, yt, T));
  const got = await ops.read(yt, T);
  const cpu = new CpuOps(cfg, { round16: false, maxCtx: 64 });
  const ref = new Float32Array(T * out);
  cpu.matmul(x, { data: wref, shape: [out, inn] }, bb ? { data: b, shape: [out] } : null, ref, T);
  const path = T === 1 && !forceTiled ? `matvec.${matvec}` : "tiled";
  wb.buffer.destroy(); bb?.buffer.destroy(); xt.buffer.destroy(); yt.buffer.destroy(); ops.destroy();
  return { name: `matmul ${kind} [${T},${inn}]x[${out},${inn}]^T${bias ? "+b" : ""}`, T, kind, path, cmp: cmp(got, ref) };
}

async function unitRmsnorm(T: number, d: number, seed: number): Promise<UnitResult> {
  const { device } = await gpu();
  const cfg: ModelConfig = { d, heads: 1, kvHeads: 1, headDim: 64, inter: d, eps: 1e-6, ropeTheta: 1e6, layers: 1 };
  const ops = new GpuOps(device, cfg, { maxCtx: 64, round16: false });
  const x = randF16(T * d, 3, seed), w = randF16(d, 2, seed + 1);
  const wb = f16Weight(device, "w", [d], w);
  const xt = ops.tensor("x", T, d), yt = ops.tensor("y", T, d);
  ops.write(xt, x);
  ops.run(() => ops.rmsnorm(xt, wb, yt, T));
  const got = await ops.read(yt, T);
  const cpu = new CpuOps(cfg, { round16: false, maxCtx: 64 });
  const ref = new Float32Array(T * d);
  cpu.rmsnorm(x, { data: w, shape: [d] }, ref, T);
  wb.buffer.destroy(); xt.buffer.destroy(); yt.buffer.destroy(); ops.destroy();
  return { name: `rmsnorm [${T},${d}]`, T, kind: "fp16", path: "rmsnorm", cmp: cmp(got, ref) };
}

/** Random-weight synthetic block on GPU vs the CPU reference (rope/attention/silu/add coverage at unit level). */
async function unitBlock(T: number, pos0: number, seed: number, round16: boolean): Promise<UnitResult[]> {
  const { device } = await gpu();
  const cfg: ModelConfig = { d: 128, heads: 4, kvHeads: 2, headDim: 32, inter: 256, eps: 1e-6, ropeTheta: 1e4, layers: 1 };
  const maxCtx = 64;
  const ops = new GpuOps(device, cfg, { maxCtx, round16 });
  const cpu = new CpuOps(cfg, { round16, maxCtx, rope: ops.rope });
  const mk = (n: string, shape: number[], scale: number, s: number) => { const v = randF16(shape.reduce((a, b) => a * b, 1), scale, s); return { gpu: f16Weight(device, n, shape, v), cpu: { data: v, shape } as CpuWeight }; };
  const W = {
    inputNorm: mk("in", [128], 1, seed), q: mk("q", [128, 128], 0.1, seed + 1), qBias: mk("qb", [128], 0.1, seed + 2), k: mk("k", [64, 128], 0.1, seed + 3), kBias: mk("kb", [64], 0.1, seed + 4),
    v: mk("v", [64, 128], 0.1, seed + 5), vBias: mk("vb", [64], 0.1, seed + 6), o: mk("o", [128, 128], 0.1, seed + 7), postNorm: mk("pn", [128], 1, seed + 8),
    gate: mk("g", [256, 128], 0.1, seed + 9), up: mk("u", [256, 128], 0.1, seed + 10), down: mk("d", [128, 256], 0.1, seed + 11),
  };
  const gw = Object.fromEntries(Object.entries(W).map(([k, v]) => [k, v.gpu])) as Record<keyof typeof W, WeightBuffer>;
  const cw = Object.fromEntries(Object.entries(W).map(([k, v]) => [k, v.cpu])) as Record<keyof typeof W, CpuWeight>;
  const t = (n: string, cols: number) => ops.tensor(n, maxCtx, cols);
  const gws = { x: t("x", 128), x2: t("x2", 128), h: t("h", 128), q: t("q", 128), k: t("k", 64), v: t("v", 64), attn: t("attn", 128), o: t("o", 128), gate: t("gate", 256), up: t("up", 256), act: t("act", 256), mlp: t("mlp", 128) };
  const c = (cols: number) => new Float32Array(maxCtx * cols);
  const cws = { x: c(128), x2: c(128), h: c(128), q: c(128), k: c(64), v: c(64), attn: c(128), o: c(128), gate: c(256), up: c(256), act: c(256), mlp: c(128) };
  const gkv = createKv(device, cfg, maxCtx, "kv"), ckv = cpuKv(cfg, maxCtx);
  const results: UnitResult[] = [];
  // Prefill pos0 rows first (so the cache has history), then the T rows under test.
  const pre = randF16(pos0 * 128, 1, seed + 20);
  if (pos0 > 0) {
    ops.write(gws.x, pre); ops.run(() => forwardBlock(ops, gw, gws, gkv, pos0, 0));
    cws.x.set(pre); forwardBlock(cpu, cw, cws, ckv, pos0, 0);
  }
  const x = randF16(T * 128, 1, seed + 21);
  for (const stage of STAGES) {
    ops.write(gws.x, x);
    ops.run(() => forwardBlock(ops, gw, gws, gkv, T, pos0, stage));
    cws.x.set(x);
    forwardBlock(cpu, cw, cws, ckv, T, pos0, stage);
    const key = stageBuf(stage);
    const got = await ops.read(gws[key], T);
    const ref = cws[key].subarray(0, T * gws[key].cols);
    results.push({ name: `block/${stage}`, T, kind: "fp16", path: `T=${T},pos0=${pos0},round16=${round16}`, cmp: cmp(got, ref) });
  }
  const kg = await ops.readKv(gkv.k, gkv.cols, pos0 + T), vg = await ops.readKv(gkv.v, gkv.cols, pos0 + T);
  results.push({ name: "block/kcache", T, kind: "fp16", path: "", cmp: cmp(kg, ckv.k.subarray(0, (pos0 + T) * 64)) });
  results.push({ name: "block/vcache", T, kind: "fp16", path: "", cmp: cmp(vg, ckv.v.subarray(0, (pos0 + T) * 64)) });
  for (const w of Object.values(gw)) w.buffer.destroy();
  for (const b of Object.values(gws)) b.buffer.destroy();
  gkv.k.destroy(); gkv.v.destroy(); ops.destroy();
  return results;
}

function stageBuf(stage: Stage): keyof typeof STAGE_BUF { return STAGE_BUF_MAP[stage]; }
const STAGE_BUF = { x: 0, x2: 0, h: 0, q: 0, k: 0, v: 0, attn: 0, o: 0, gate: 0, up: 0, act: 0, mlp: 0 };
const STAGE_BUF_MAP: Record<Stage, keyof typeof STAGE_BUF> = { input_norm: "h", q_proj: "q", k_proj: "k", v_proj: "v", rope: "q", attn: "attn", o_proj: "o", residual1: "x2", post_norm: "h", gate: "gate", up: "up", silu_mul: "act", down: "mlp", residual2: "x" };

export async function gate2Unit(): Promise<UnitResult[]> {
  const out: UnitResult[] = [];
  out.push(await unitMatmul("fp16", 32, 896, 896, true, 1));
  out.push(await unitMatmul("fp16", 32, 896, 128, true, 2));
  out.push(await unitMatmul("fp16", 32, 896, 4864, false, 3));
  out.push(await unitMatmul("fp16", 32, 4864, 896, false, 4));
  out.push(await unitMatmul("fp16", 7, 896, 896, true, 5));
  out.push(await unitMatmul("fp16", 1, 896, 896, true, 6));
  out.push(await unitMatmul("fp16", 1, 4864, 896, false, 7));
  out.push(await unitMatmul("fp16", 1, 896, 896, true, 6, true));
  out.push(await unitMatmul("fp16", 1, 896, 896, true, 6, false, "seq"));
  out.push(await unitMatmul("fp16", 1, 4864, 896, false, 7, false, "seq"));
  out.push(await unitRmsnorm(32, 896, 8));
  out.push(await unitRmsnorm(1, 896, 9));
  out.push(...(await unitBlock(5, 0, 10, false)));
  out.push(...(await unitBlock(1, 5, 11, false)));
  out.push(...(await unitBlock(3, 2, 12, true)));
  return out;
}

export async function gate7Unit(): Promise<UnitResult[]> {
  const out: UnitResult[] = [];
  for (const kind of ["q8", "q4"] as const) {
    out.push(await unitMatmul(kind, 32, 896, 896, true, 21));
    out.push(await unitMatmul(kind, 32, 896, 4864, false, 22));
    out.push(await unitMatmul(kind, 32, 4864, 896, false, 23));
    out.push(await unitMatmul(kind, 1, 896, 896, true, 24));
    out.push(await unitMatmul(kind, 1, 4864, 896, false, 25));
    out.push(await unitMatmul(kind, 1, 896, 4864, false, 26));
    out.push(await unitMatmul(kind, 1, 4864, 896, false, 25, false, "seq"));
  }
  return out;
}

export async function gate2Embed(): Promise<{ fp16: Cmp; q8: Cmp; tokens: number }> {
  const p = await json<Prompt>("/fixtures/prompt.json");
  const ref = await npy("embed.npy");
  const rt16 = await runtime("fp16", 1);
  const rtq4 = await runtime("q4", 1);
  const q8ref = await npy("q8/embed.npy");
  return { fp16: cmp(rt16.embed(p.token_ids), ref.data), q8: cmp(rtq4.embed(p.token_ids), q8ref.data), tokens: p.T };
}

export async function ropeCheck(): Promise<{ cos: Cmp; sin: Cmp }> {
  const m = await manifest();
  const r = await fixtureRope(configFromManifest(m), 64);
  return { cos: r.cmpCos, sin: r.cmpSin };
}

// -- gate 3: block 0 stage by stage --------------------------------------------------------

const INTRA: [Stage, string][] = [
  ["input_norm", "post_input_norm"], ["q_proj", "q_proj"], ["k_proj", "k_proj"], ["v_proj", "v_proj"], ["rope", "q_rope"],
  ["attn", "attn_out"], ["o_proj", "o_proj"], ["residual1", "post_o_proj_residual"], ["post_norm", "post_post_attn_norm"],
  ["down", "mlp_out"], ["residual2", "block_out"],
];

export interface StageResult { stage: string; fixture: string; cmp: Cmp }

export async function gate3(opts: { attnFlags?: number; round16?: boolean } = {}): Promise<StageResult[]> {
  const p = await json<Prompt>("/fixtures/prompt.json");
  const emb = await npy("embed.npy");
  const rt = await runtime("fp16", 1, { useFixtureRope: true, round16: opts.round16 ?? true, ...(opts.attnFlags !== undefined ? { attnFlags: opts.attnFlags } : {}) });
  const out: StageResult[] = [];
  for (const [stage, file] of INTRA) {
    rt.reset();
    await rt.debugForward(emb.data, p.T, 0, 0, stage);
    const got = await rt.debugRead(stage, p.T);
    const ref = await npy(`intra/${file}.npy`);
    out.push({ stage, fixture: file, cmp: cmp(got, ref.data) });
    if (stage === "rope") {
      const kr = await npy("intra/k_rope.npy");
      out.push({ stage: "rope(k cache)", fixture: "k_rope", cmp: cmp(await rt.debugReadKv(0, "k", p.T), kr.data) });
      const vr = await npy("intra/v_proj.npy");
      out.push({ stage: "rope(v cache)", fixture: "v_proj", cmp: cmp(await rt.debugReadKv(0, "v", p.T), vr.data) });
    }
  }
  return out;
}

/** Stage-by-stage for any block that has an intra dump, fed the fixture input of that block (isolated). */
export async function stagesFor(block: number, opts: { attnFlags?: number; round16?: boolean; siluMode?: number } = {}): Promise<StageResult[]> {
  const p = await json<Prompt>("/fixtures/prompt.json");
  const sub = block === 0 ? "intra" : `intra_${block}`;
  const input = await npy(`${sub}/block_in.npy`);
  const rt = await runtime("fp16", 24, { useFixtureRope: true, round16: opts.round16 ?? true, ...(opts.attnFlags !== undefined ? { attnFlags: opts.attnFlags } : {}), ...(opts.siluMode !== undefined ? { siluMode: opts.siluMode } : {}) });
  const out: StageResult[] = [];
  for (const [stage, file] of INTRA) {
    rt.reset();
    await rt.debugForward(input.data, p.T, 0, block, stage, block);
    const got = await rt.debugRead(stage, p.T);
    const ref = await npy(`${sub}/${file}.npy`);
    out.push({ stage, fixture: file, cmp: cmp(got, ref.data) });
  }
  return out;
}

/** Each block alone: fed the fixture output of the previous block, compared with its own fixture output. */
export async function gate4Isolated(opts: { attnFlags?: number; round16?: boolean; siluMode?: number } = {}): Promise<BlockResult[]> {
  const p = await json<Prompt>("/fixtures/prompt.json");
  const rt = await runtime("fp16", 24, { useFixtureRope: true, round16: opts.round16 ?? true, ...(opts.attnFlags !== undefined ? { attnFlags: opts.attnFlags } : {}), ...(opts.siluMode !== undefined ? { siluMode: opts.siluMode } : {}) });
  const out: BlockResult[] = [];
  for (let i = 0; i < 24; i++) {
    const input = await npy(i === 0 ? "embed.npy" : `block_${String(i - 1).padStart(2, "0")}.npy`);
    rt.reset();
    await rt.debugForward(input.data, p.T, 0, i, undefined, i);
    const ref = await npy(`block_${String(i).padStart(2, "0")}.npy`);
    out.push({ block: i, cmp: cmp(await rt.debugRead("residual2", p.T), ref.data) });
  }
  return out;
}

// -- gate 4: all blocks, prefill ---------------------------------------------------------------

export interface BlockResult { block: number; cmp: Cmp }

export async function gate4(variant: VariantName = "fp16", opts: { attnFlags?: number; round16?: boolean; siluMode?: number } = {}): Promise<BlockResult[]> {
  const p = await json<Prompt>("/fixtures/prompt.json");
  const prefix = variant === "fp16" ? "" : `${variant}/`;
  const emb = await npy(`${prefix}embed.npy`);
  const rt = await runtime(variant, 24, { useFixtureRope: true, round16: opts.round16 ?? true, ...(opts.attnFlags !== undefined ? { attnFlags: opts.attnFlags } : {}), ...(opts.siluMode !== undefined ? { siluMode: opts.siluMode } : {}) });
  rt.enableTrace();
  rt.reset();
  await rt.debugForward(emb.data, p.T, 0, 23);
  const out: BlockResult[] = [];
  for (let i = 0; i < 24; i++) {
    const ref = await npy(`${prefix}block_${String(i).padStart(2, "0")}.npy`);
    out.push({ block: i, cmp: cmp(await rt.traceRead(i, p.T), ref.data) });
  }
  return out;
}

// -- gate 5: decode with the KV cache -----------------------------------------------------------

export interface Gate5a { block: number; vsOwnPrefill: Cmp; vsFixture: Cmp }

export async function gate5a(opts: RuntimeOptions = {}): Promise<Gate5a[]> {
  const p = await json<Prompt>("/fixtures/prompt.json");
  const emb = await npy("embed.npy");
  const rt = await runtime("fp16", 24, { useFixtureRope: true, ...opts });
  rt.enableTrace();
  const d = rt.cfg.d;
  // Full prefill path, all 32 rows.
  rt.reset();
  await rt.debugForward(emb.data, p.T, 0, 23);
  const full: Float32Array[] = [];
  for (let i = 0; i < 24; i++) full.push((await rt.traceRead(i, p.T)).slice(31 * d, 32 * d));
  // 31 then 1.
  rt.reset();
  await rt.debugForward(emb.data.subarray(0, 31 * d), 31, 0, 23);
  await rt.debugForward(emb.data.subarray(31 * d, 32 * d), 1, 31, 23);
  const out: Gate5a[] = [];
  for (let i = 0; i < 24; i++) {
    const got = await rt.traceRead(i, 1);
    const ref = await npy(`block_${String(i).padStart(2, "0")}.npy`);
    out.push({ block: i, vsOwnPrefill: cmp(got, full[i]!), vsFixture: cmp(got, ref.data.subarray(31 * d, 32 * d)) });
  }
  return out;
}

export interface Gate5b { step: number; token: number; position: number; perBoundary: Cmp[]; worst: Cmp }

export async function gate5b(opts: RuntimeOptions = {}): Promise<Gate5b[]> {
  const p = await json<Prompt>("/fixtures/prompt.json");
  const g = await json<Greedy>("/fixtures/greedy.json");
  const steps = await npy("decode_steps.npy"); // [16, 25, d]
  const rt = await runtime("fp16", 24, { useFixtureRope: true, ...opts });
  rt.enableTrace();
  const d = rt.cfg.d, L = 24;
  rt.reset();
  await rt.prefill(p.token_ids);
  const out: Gate5b[] = [];
  for (let s = 0; s < g.steps; s++) {
    const tok = g.tokens[s]!, pos = g.positions[s]!;
    await rt.decode(tok, pos);
    const per: Cmp[] = [];
    const embRow = rt.embed([tok]);
    per.push(cmp(embRow, steps.data.subarray((s * (L + 1)) * d, (s * (L + 1) + 1) * d)));
    for (let i = 0; i < L; i++) per.push(cmp(await rt.traceRead(i, 1), steps.data.subarray((s * (L + 1) + i + 1) * d, (s * (L + 1) + i + 2) * d)));
    const worst = per.reduce((a, b) => (b.maxAbs > a.maxAbs ? b : a));
    out.push({ step: s, token: tok, position: pos, perBoundary: per, worst });
  }
  return out;
}

// -- gate 6 / 7: end to end with the server ------------------------------------------------------

export interface Gate6 {
  N: number; variant: string; tokens: number[]; expected: number[]; margins: number[]; runnersUp: number[];
  /** Steps where the server's token differs from the reference (the reference token is fed at every step: teacher forcing). */
  mismatches: { step: number; got: number; expected: number; margin: number; runnerUp: number }[];
  /** Mismatches whose reference top-2 gap is at or above the tie tolerance. */
  disagreements: number[];
  tieTolerance: number; serverBusyMs: number[]; exportMs: number[];
}

/** One fp16 logit ulp at |logit| in [8, 16); a reference top-2 gap of at most this is a tie the fixture cannot decide. */
export const TIE_TOLERANCE = 1 / 128;

export async function gate6(N: number, variant: VariantName, wsUrl: string, token: string, steps = 16, opts: RuntimeOptions = {}): Promise<Gate6> {
  const p = await json<Prompt>("/fixtures/prompt.json");
  let expected: number[], margins: number[], runnersUp: number[];
  if (variant === "fp16") { const g = await json<Greedy & { margins: number[]; runners_up: number[] }>("/fixtures/greedy.json"); expected = g.tokens; margins = g.margins; runnersUp = g.runners_up; }
  else { const g = await json<{ N: Record<string, number[]>; margins: Record<string, number[]>; runners_up: Record<string, number[]> }>(`/fixtures/${variant}/greedy.json`); expected = g.N[String(N)]!; margins = g.margins[String(N)]!; runnersUp = g.runners_up[String(N)]!; }
  const rt = await runtime(variant, N, { maxCtx: 128, ...opts });
  rt.reset();
  const client = new SplitClient(wsUrl, token);
  await client.connect();
  const tokens: number[] = [];
  const busy: number[] = [], exportMs: number[] = [];
  try {
    await client.open(MODEL, N, 128);
    await rt.prefill(p.token_ids);
    let hidden = await rt.exportHidden();
    exportMs.push(rt.stats().lastExportMs);
    let tok = await client.prefill(hidden, p.T, 0);
    tokens.push(tok.id); busy.push(tok.busy_ms);
    let pos = p.T;
    for (let s = 1; s < steps; s++) {
      await rt.decode(expected[s - 1]!, pos);   // teacher forcing: the reference token, whatever the server said
      hidden = await rt.exportHidden();
      exportMs.push(rt.stats().lastExportMs);
      tok = await client.decode(hidden, pos);
      tokens.push(tok.id); busy.push(tok.busy_ms);
      pos++;
    }
  } finally {
    client.close();
  }
  const mismatches = tokens.map((t, i) => ({ step: i, got: t, expected: expected[i]!, margin: margins[i]!, runnerUp: runnersUp[i]! })).filter((m) => m.got !== m.expected);
  const disagreements = mismatches.filter((m) => m.margin > TIE_TOLERANCE || m.got !== m.runnerUp).map((m) => m.step);
  return { N, variant, tokens, expected: expected.slice(0, steps), margins: margins.slice(0, steps), runnersUp: runnersUp.slice(0, steps), mismatches, disagreements, tieTolerance: TIE_TOLERANCE, serverBusyMs: busy, exportMs };
}

// -- gate 8: lm_head vs logits.npy; gate 9: local greedy (Phase 5b) --------------------------------

export interface Gate8 { variant: string; N: number; cmp: Cmp; argmaxMismatches: { row: number; got: number; ref: number }[]; lastRowTop2: { id: number; logit: number }[]; lmHeadMs: number[]; headMatvec: string }

/** Every row's logits from the runtime's final norm + lm_head vs `logits.npy` (fp16 reference). */
export async function gate8LmHead(variant: VariantName = "fp16", opts: RuntimeOptions = {}): Promise<Gate8> {
  const p = await json<Prompt>("/fixtures/prompt.json");
  const ref = await npy("logits.npy");
  const rt = await runtime(variant, 24, { maxCtx: 128, lmHead: true, ...opts });
  rt.reset();
  await rt.prefill(p.token_ids);
  const got = await rt.debugLogitsAll();
  const V = rt.cfg.d === ref.shape[1] ? ref.shape[1]! : ref.shape[1]!;
  const mism: Gate8["argmaxMismatches"] = [];
  for (let r = 0; r < p.T; r++) {
    const g = argmax(got.subarray(r * V, (r + 1) * V)), e = argmax(ref.data.subarray(r * V, (r + 1) * V));
    if (g !== e) mism.push({ row: r, got: g, ref: e });
  }
  // the production path (last row only, matvec) timed 5x
  const ms: number[] = [];
  let last: Float32Array = new Float32Array(0);
  for (let i = 0; i < 5; i++) { last = await rt.logits(); ms.push(rt.stats().lastLmHeadMs); }
  return { variant, N: 24, cmp: cmp(got, ref.data), argmaxMismatches: mism, lastRowTop2: topk(last, 2), lmHeadMs: ms, headMatvec: opts.lmHeadMatvec ?? rt.ops.matvecMode() };
}

export interface Gate9 extends Omit<Gate6, "serverBusyMs" | "exportMs"> { steps: { clientMs: number; lmHeadMs: number; sampleMs: number; totalMs: number }[] }

/** Free-running greedy generation in local mode (no server) vs the full-model greedy tokens. */
export async function gate9Local(variant: VariantName = "fp16", steps = 16, opts: RuntimeOptions = {}): Promise<Gate9> {
  const p = await json<Prompt>("/fixtures/prompt.json");
  let expected: number[], margins: number[], runnersUp: number[];
  if (variant === "fp16") { const g = await json<Greedy & { margins: number[]; runners_up: number[] }>("/fixtures/greedy.json"); expected = g.tokens; margins = g.margins; runnersUp = g.runners_up; }
  else { const g = await json<{ N: Record<string, number[]>; margins: Record<string, number[]>; runners_up: Record<string, number[]> }>(`/fixtures/${variant}/greedy.json`); expected = g.N.local!; margins = g.margins.local!; runnersUp = g.runners_up.local!; }
  const rt = await runtime(variant, 24, { maxCtx: 128, lmHead: true, ...opts });
  const session = await SplitSession.open({ mode: "local", N: 24, model: MODEL, maxCtx: 128, runtime: rt, sampling: { temperature: 0 }, eosIds: [] });
  const tokens: number[] = [];
  const st: Gate9["steps"] = [];
  for await (const t of session.generate(p.token_ids, steps)) { tokens.push(t.id); st.push({ clientMs: t.timing.clientMs, lmHeadMs: t.timing.lmHeadMs, sampleMs: t.timing.sampleMs, totalMs: t.timing.totalMs }); }
  session.close();
  // Free-running: after the first divergence the sequences differ anyway; report the first mismatch and the rest.
  const mismatches = tokens.map((t, i) => ({ step: i, got: t, expected: expected[i]!, margin: margins[i]!, runnerUp: runnersUp[i]! })).filter((m) => m.got !== m.expected);
  const first = mismatches[0];
  const disagreements = first && (first.margin > TIE_TOLERANCE || first.got !== first.runnerUp) ? [first.step] : [];
  return { N: 24, variant, tokens, expected: expected.slice(0, steps), margins: margins.slice(0, steps), runnersUp: runnersUp.slice(0, steps), mismatches, disagreements, tieTolerance: TIE_TOLERANCE, steps: st };
}

export async function microbench(variant: VariantName, opts: MicrobenchOptions = {}) {
  const { device } = await gpu();
  return runMicrobench(device, await manifest(), variant, opts);
}

// -- info / bench helpers -------------------------------------------------------------------------

export async function info(): Promise<unknown> { return (await gpu()).info; }
export function loads(): Record<string, LoadInfo> { return Object.fromEntries(loadInfo); }

export async function bench(variant: VariantName, N: number, decodeSteps = 32, matvec: "seq" | "lanes" | "tiled" = "seq"): Promise<{ prefillTokPerS: number; decodeTokPerS: number; prefillMs: number; decodeMsMedian: number; exportMs: number; stats: ReturnType<Runtime["stats"]>; load: LoadInfo | undefined }> {
  const p = await json<Prompt>("/fixtures/prompt.json");
  const o: RuntimeOptions & { maxCtx: number } = matvec === "tiled" ? { maxCtx: 256, forceTiled: true } : { maxCtx: 256, matvec };
  const key = `${variant}:${N}:${JSON.stringify(o)}`;
  const rt = await runtime(variant, N, o);
  rt.reset();
  await rt.prefill(p.token_ids); // warm-up
  rt.reset();
  await rt.prefill(p.token_ids);
  const prefillMs = rt.stats().lastPrefillMs;
  const times: number[] = [];
  let pos = p.T;
  for (let i = 0; i < decodeSteps; i++) { await rt.decode(p.token_ids[i % p.T]!, pos++); times.push(rt.stats().lastDecodeMs); }
  times.sort((a, b) => a - b);
  await rt.exportHidden();
  const med = times[Math.floor(times.length / 2)]!;
  return { prefillTokPerS: (p.T / prefillMs) * 1000, decodeTokPerS: 1000 / med, prefillMs, decodeMsMedian: med, exportMs: rt.stats().lastExportMs, stats: rt.stats(), load: loadInfo.get(key) };
}

declare global { interface Window { harness: typeof harness } }
const harness = { info, loads, ropeCheck, gate2Unit, gate2Embed, gate3, gate4, gate4Isolated, stagesFor, gate5a, gate5b, gate6, gate7Unit, gate8LmHead, gate9Local, microbench, bench, f16ToF32 };
window.harness = harness;
