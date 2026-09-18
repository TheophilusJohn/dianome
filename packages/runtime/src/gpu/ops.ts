// Typed GPU ops implementing BlockOps over a command encoder. Every op records one dispatch into the current
// compute pass; uniform params live in one small buffer per call site (written with queue.writeBuffer before
// the submit) and bind groups are cached per call site, so a steady-state forward pass allocates nothing.

import type { BlockOps, ModelConfig } from "../block";
import { f16ToF32 } from "../f16";
import type { GpuKv } from "../kv";
import { ropeTable, type RopeTable } from "../rope";
import { createStorage, createUniform, partBinding, readback, type WeightBuffer } from "./buffers";
import addSrc from "./kernels/add.wgsl";
import attentionSrc from "./kernels/attention.wgsl";
import f32ToF16Src from "./kernels/f32_to_f16.wgsl";
import matmulHeadSrc from "./kernels/matmul_head.wgsl";
import matmulTiledSrc from "./kernels/matmul_tiled.wgsl";
import matvecSrc from "./kernels/matvec.wgsl";
import matvecSeqSrc from "./kernels/matvec_seq.wgsl";
import rmsnormSrc from "./kernels/rmsnorm.wgsl";
import ropeKvSrc from "./kernels/rope_kv.wgsl";
import siluMulSrc from "./kernels/silu_mul.wgsl";
import weightsF16Src from "./kernels/weights_f16.wgsl";
import weightsQ4Src from "./kernels/weights_q4.wgsl";
import weightsQ8Src from "./kernels/weights_q8.wgsl";

export interface GpuTensor { buffer: GPUBuffer; cols: number; rows: number; label: string }

export interface GpuOpsOptions {
  maxCtx: number;
  /** Emulate an fp16 PyTorch model: round to fp16 wherever it materialises an fp16 tensor. */
  round16: boolean;
  /** Attention rounding flags (bit0 output, bit1 probabilities, bit2 scores); default = round16 ? 1 : 0. */
  attnFlags?: number;
  /** Use a given cos/sin table (the fixtures) instead of computing one. */
  rope?: RopeTable;
  /** Force the tiled kernel even for T = 1 (benchmarking the two decode paths). */
  forceTiled?: boolean;
  /** silu emulation: 0 = round(x*sigmoid(x)), 1 = round(x * round(sigmoid(x))) (diagnostics). */
  siluMode?: number;
  /**
   * T = 1 path. "seq": one thread per output row summing in the tiled kernel's order, so decode reproduces
   * prefill bit for bit under fp16 emulation. "lanes": 16 lanes per row + reduction (about 2x faster per token,
   * different summation order → one-ulp flips vs prefill). Default: "seq" when round16 is on, "lanes" when off.
   */
  matvec?: "lanes" | "seq";
}

export interface OpCounts { dispatches: number; bindGroupsCreated: number; uniformsCreated: number }

const TILE_M = 32, TILE_N = 64, TILE_K = 32;

let nextId = 1;
const ids = new WeakMap<object, number>();
function idOf(o: object): number { let i = ids.get(o); if (!i) { i = nextId++; ids.set(o, i); } return i; }

interface Site { uniform: GPUBuffer; bindGroup: GPUBindGroup }

export class GpuOps implements BlockOps<WeightBuffer, GpuTensor, GpuKv> {
  readonly rope: RopeTable;
  readonly ropeCos: GPUBuffer;
  readonly ropeSin: GPUBuffer;
  readonly counts: OpCounts = { dispatches: 0, bindGroupsCreated: 0, uniformsCreated: 0 };
  readonly round16: boolean;
  readonly attnFlags: number;
  readonly forceTiled: boolean;
  private readonly pipelines = new Map<string, GPUComputePipeline>();
  private readonly sites = new Map<string, Site>();
  private readonly dummyBias: GPUBuffer;
  private encoder: GPUCommandEncoder | null = null;
  private pass: GPUComputePassEncoder | null = null;
  private readonly u32 = new Uint32Array(16);
  private readonly f32 = new Float32Array(this.u32.buffer);
  readonly tally = { bytes: 0 };

  constructor(readonly device: GPUDevice, readonly cfg: ModelConfig, readonly opts: GpuOpsOptions) {
    this.round16 = opts.round16;
    this.attnFlags = opts.attnFlags ?? (opts.round16 ? 1 : 0);
    this.forceTiled = opts.forceTiled ?? false;
    if (cfg.headDim % 4 !== 0 || cfg.headDim > 256) throw new Error(`head_dim ${cfg.headDim} unsupported`);
    const wgMem = device.limits.maxComputeWorkgroupStorageSize;
    const need = opts.maxCtx * 4 + cfg.headDim * 8;
    if (need > wgMem) throw new Error(`maxCtx ${opts.maxCtx} needs ${need} bytes of workgroup memory for attention, device allows ${wgMem}`);
    this.rope = opts.rope ?? ropeTable(cfg.ropeTheta, cfg.headDim, opts.maxCtx, opts.round16);
    if (this.rope.positions < opts.maxCtx) throw new Error("rope table shorter than maxCtx");
    this.ropeCos = createStorage(device, "rope.cos", this.rope.cos.byteLength, this.tally);
    this.ropeSin = createStorage(device, "rope.sin", this.rope.sin.byteLength, this.tally);
    device.queue.writeBuffer(this.ropeCos, 0, this.rope.cos as Float32Array<ArrayBuffer>);
    device.queue.writeBuffer(this.ropeSin, 0, this.rope.sin as Float32Array<ArrayBuffer>);
    this.dummyBias = createStorage(device, "bias.none", 16);
  }

  // -- pipelines ---------------------------------------------------------------------

  private pipeline(key: string, source: () => string): GPUComputePipeline {
    let p = this.pipelines.get(key);
    if (!p) {
      const module = this.device.createShaderModule({ label: key, code: source() });
      p = this.device.createComputePipeline({ label: key, layout: "auto", compute: { module, entryPoint: "main" } });
      this.pipelines.set(key, p);
    }
    return p;
  }

  private matmulPipeline(kind: WeightBuffer["kind"], vec: boolean): GPUComputePipeline {
    const w = kind === "fp16" ? weightsF16Src : kind === "q8" ? weightsQ8Src : weightsQ4Src;
    const mode = this.matvecMode();
    const main = vec ? (mode === "seq" ? matvecSeqSrc : matvecSrc) : matmulTiledSrc;
    return this.pipeline(`matmul.${kind}.${vec ? "vec." + mode : "tiled"}`, () => matmulHeadSrc + "\n" + w + "\n" + main);
  }

  /** T = 1 kernel in use: explicit choice, else "seq" under fp16 emulation (decode == prefill bit for bit), "lanes" otherwise. */
  matvecMode(): "lanes" | "seq" {
    return this.opts.matvec ?? (this.round16 ? "seq" : "lanes");
  }

  private attentionPipeline(): GPUComputePipeline {
    return this.pipeline("attention", () => attentionSrc.replace("__MAX_KEYS__", String(this.opts.maxCtx)).replace("__HD__", String(this.cfg.headDim)));
  }

  // -- recording ---------------------------------------------------------------------

  begin(label = "forward"): void {
    if (this.pass) throw new Error("GpuOps: pass already open");
    this.encoder = this.device.createCommandEncoder({ label });
    this.pass = this.encoder.beginComputePass({ label });
  }

  /** Ends the pass and returns the command buffer (the caller submits, possibly with more work). */
  end(): GPUCommandBuffer {
    if (!this.pass || !this.encoder) throw new Error("GpuOps: no open pass");
    this.pass.end();
    const cb = this.encoder.finish();
    this.pass = null;
    this.encoder = null;
    return cb;
  }

  /** Copies bytes between buffers mid-recording (ends the pass, copies, reopens the pass). */
  copy(src: GPUBuffer, srcOffset: number, dst: GPUBuffer, dstOffset: number, bytes: number): void {
    if (!this.pass || !this.encoder) throw new Error("GpuOps: no open pass");
    const label = this.encoder.label;
    this.pass.end();
    this.encoder.copyBufferToBuffer(src, srcOffset, dst, dstOffset, bytes);
    this.pass = this.encoder.beginComputePass({ label });
  }

  /** Records and submits `f` as one command buffer. */
  run(f: () => void, label?: string): void {
    this.begin(label);
    try { f(); } catch (e) { this.pass?.end(); this.pass = null; this.encoder = null; throw e; }
    this.device.queue.submit([this.end()]);
  }

  private dispatch(pipeline: GPUComputePipeline, s: Site, words: number, x: number, y = 1): void {
    if (!this.pass) throw new Error("GpuOps: op recorded outside begin()/end()");
    this.device.queue.writeBuffer(s.uniform, 0, this.u32, 0, Math.max(4, words));
    this.pass.setPipeline(pipeline);
    this.pass.setBindGroup(0, s.bindGroup);
    this.pass.dispatchWorkgroups(x, y);
    this.counts.dispatches++;
  }

  // -- ops -----------------------------------------------------------------------------

  rmsnorm(x: GpuTensor, w: WeightBuffer, out: GpuTensor, T: number): void {
    const cols = w.shape[0]!;
    if (x.cols !== cols || out.cols !== cols) throw new Error(`rmsnorm: cols mismatch (${x.cols}, ${cols}, ${out.cols})`);
    const p = this.pipeline("rmsnorm", () => rmsnormSrc);
    const s = this.siteU(`rmsnorm:${idOf(x.buffer)}:${idOf(w.buffer)}:${idOf(out.buffer)}`, p, [{ buffer: this.dummyBias }, { buffer: x.buffer }, partBinding(w, "weights"), { buffer: out.buffer }], 16);
    this.u32[0] = T; this.u32[1] = cols; this.f32[2] = this.cfg.eps; this.u32[3] = this.round16 ? 1 : 0;
    this.dispatch(p, s, 4, T, 1);
  }

  matmul(x: GpuTensor, w: WeightBuffer, bias: WeightBuffer | null, out: GpuTensor, T: number): void {
    const [o, i] = [w.shape[0]!, w.shape[1]!];
    if (x.cols !== i || out.cols !== o) throw new Error(`matmul ${w.name}: shapes x[${x.cols}] W[${o},${i}] out[${out.cols}]`);
    if (i % TILE_K !== 0 || o % TILE_N !== 0) throw new Error(`matmul ${w.name}: in must be a multiple of ${TILE_K} and out of ${TILE_N}`);
    const vec = T === 1 && !this.forceTiled;
    const p = this.matmulPipeline(w.kind, vec);
    const entries: (GPUBindingResource | null)[] = [{ buffer: this.dummyBias }, { buffer: x.buffer }, { buffer: out.buffer }, partBinding(w, "weights"), bias ? partBinding(bias, "weights") : { buffer: this.dummyBias }];
    if (w.kind !== "fp16") entries.push(partBinding(w, "scales"));
    if (w.kind === "q4") entries.push(partBinding(w, "zeros"));
    const s = this.siteU(`matmul.${vec ? "v" : "t"}:${idOf(x.buffer)}:${idOf(w.buffer)}:${bias ? idOf(bias.buffer) : 0}:${idOf(out.buffer)}`, p, entries as GPUBindingResource[], 16);
    this.u32[0] = T; this.u32[1] = i; this.u32[2] = o; this.u32[3] = (bias ? 1 : 0) | (this.round16 ? 2 : 0);
    if (vec) this.dispatch(p, s, 4, this.matvecMode() === "seq" ? Math.ceil(o / 64) : Math.ceil(o / 4), 1);
    else this.dispatch(p, s, 4, o / TILE_N, Math.ceil(T / TILE_M));
  }

  ropeKv(q: GpuTensor, k: GpuTensor, v: GpuTensor, T: number, pos0: number, kv: GpuKv): void {
    const { heads, kvHeads, headDim: hd } = this.cfg;
    const quads = hd / 4;
    const qSlots = heads * quads, kSlots = kvHeads * quads, vSlots = (kvHeads * hd) / 2;
    const perT = qSlots + kSlots + vSlots;
    const p = this.pipeline("rope_kv", () => ropeKvSrc);
    const s = this.siteU(`rope:${idOf(q.buffer)}:${idOf(k.buffer)}:${idOf(v.buffer)}:${idOf(kv.k)}`, p,
      [{ buffer: this.dummyBias }, { buffer: q.buffer }, { buffer: k.buffer }, { buffer: v.buffer }, { buffer: this.ropeCos }, { buffer: this.ropeSin }, { buffer: kv.k }, { buffer: kv.v }], 48);
    this.u32.set([T, pos0, heads, kvHeads, hd, this.round16 ? 1 : 0, qSlots, kSlots, vSlots, perT, kv.cols, 0]);
    this.dispatch(p, s, 12, Math.ceil((T * perT) / 64), 1);
  }

  attention(q: GpuTensor, kv: GpuKv, out: GpuTensor, T: number, pos0: number): void {
    const { heads, kvHeads, headDim: hd } = this.cfg;
    if (pos0 + T > this.opts.maxCtx) throw new Error(`attention: ${pos0 + T} > maxCtx ${this.opts.maxCtx}`);
    const p = this.attentionPipeline();
    const s = this.siteU(`attn:${idOf(q.buffer)}:${idOf(kv.k)}:${idOf(out.buffer)}`, p, [{ buffer: this.dummyBias }, { buffer: q.buffer }, { buffer: kv.k }, { buffer: kv.v }, { buffer: out.buffer }], 32);
    this.u32.set([T, pos0, heads, kvHeads, hd, kv.cols, this.attnFlags, 0]);
    this.dispatch(p, s, 8, T, heads);
  }

  siluMul(gate: GpuTensor, up: GpuTensor, out: GpuTensor, T: number): void {
    const n = T * gate.cols;
    const p = this.pipeline("silu_mul", () => siluMulSrc);
    const s = this.siteU(`silu:${idOf(gate.buffer)}:${idOf(up.buffer)}:${idOf(out.buffer)}`, p, [{ buffer: this.dummyBias }, { buffer: gate.buffer }, { buffer: up.buffer }, { buffer: out.buffer }], 16);
    this.u32.set([n, (this.round16 ? 1 : 0) | ((this.opts.siluMode ?? 0) << 1), 0, 0]);
    this.dispatch(p, s, 4, Math.ceil(n / 256), 1);
  }

  add(a: GpuTensor, b: GpuTensor, out: GpuTensor, T: number): void {
    const n = T * a.cols;
    const p = this.pipeline("add", () => addSrc);
    const s = this.siteU(`add:${idOf(a.buffer)}:${idOf(b.buffer)}:${idOf(out.buffer)}`, p, [{ buffer: this.dummyBias }, { buffer: a.buffer }, { buffer: b.buffer }, { buffer: out.buffer }], 16);
    this.u32.set([n, this.round16 ? 1 : 0, 0, 0]);
    this.dispatch(p, s, 4, Math.ceil(n / 256), 1);
  }

  /** Packs rows 0..T-1 of `x` into fp16 pairs in `dst` (a u32 buffer of at least T*cols/2 words). */
  packF16(x: GpuTensor, T: number, dst: GPUBuffer): void {
    const pairs = (T * x.cols) / 2;
    const p = this.pipeline("f32_to_f16", () => f32ToF16Src);
    const s = this.siteU(`pack:${idOf(x.buffer)}:${idOf(dst)}`, p, [{ buffer: this.dummyBias }, { buffer: x.buffer }, { buffer: dst }], 16);
    this.u32.set([pairs, 0, 0, 0]);
    this.dispatch(p, s, 4, Math.ceil(pairs / 256), 1);
  }

  /** A call site's uniform buffer is binding 0; `entries[0]` is a placeholder replaced by it. */
  private siteU(key: string, pipeline: GPUComputePipeline, entries: GPUBindingResource[], uniformBytes: number): Site {
    let s = this.sites.get(key);
    if (!s) {
      const uniform = createUniform(this.device, key, uniformBytes);
      const resources = entries.slice();
      resources[0] = { buffer: uniform };
      const bindGroup = this.device.createBindGroup({ label: key, layout: pipeline.getBindGroupLayout(0), entries: resources.map((resource, binding) => ({ binding, resource })) });
      s = { uniform, bindGroup };
      this.sites.set(key, s);
      this.counts.bindGroupsCreated++;
      this.counts.uniformsCreated++;
    }
    return s;
  }

  // -- helpers -------------------------------------------------------------------------

  tensor(label: string, rows: number, cols: number): GpuTensor {
    return { buffer: createStorage(this.device, label, rows * cols * 4, this.tally), rows, cols, label };
  }

  /** Writes f32 rows into a tensor (row 0 first). */
  write(t: GpuTensor, data: Float32Array, rowOffset = 0): void {
    this.device.queue.writeBuffer(t.buffer, rowOffset * t.cols * 4, data as Float32Array<ArrayBuffer>);
  }

  async read(t: GpuTensor, rows: number): Promise<Float32Array> {
    return new Float32Array(await readback(this.device, t.buffer, rows * t.cols * 4));
  }

  /** Reads fp16 rows of a KV buffer as f32. */
  async readKv(buf: GPUBuffer, cols: number, rows: number): Promise<Float32Array> {
    const u16 = new Uint16Array(await readback(this.device, buf, rows * cols * 2));
    const out = new Float32Array(u16.length);
    for (let i = 0; i < u16.length; i++) out[i] = f16ToF32(u16[i]!);
    return out;
  }

  destroy(): void {
    for (const s of this.sites.values()) s.uniform.destroy();
    this.sites.clear();
    this.ropeCos.destroy();
    this.ropeSin.destroy();
    this.dummyBias.destroy();
  }
}
