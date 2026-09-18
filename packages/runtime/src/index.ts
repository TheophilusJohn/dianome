// dianome runtime: embedding lookup + Qwen2 transformer blocks 0..N-1 on WebGPU with a KV cache, exporting the
// fp16 hidden state that is the input to block N (the Phase 4 boundary). With `lmHead: true` (Phase 5b, N = L)
// it also runs the final norm + lm_head on the last row and returns f32 logits for the CPU sampler: local mode.
//
//   const rt = await Runtime.create(device, manifest, "q4", N, maxCtx);
//   await rt.loadFrom(dianome.stream(id, { variant: "q4" }));     // uploads groups as they arrive; stops after layer N-1
//   await rt.prefill(tokens);  const hidden = await rt.exportHidden();   // fp16 [T, d_model]
//   await rt.decode(token, position);  ...
//   const rt = await Runtime.create(device, manifest, "q4", L, maxCtx, { lmHead: true });   // local mode
//   await rt.prefill(tokens);  const logits = await rt.logits();   // f32 [vocab] for the last position

import type { LoadedEntry, LoadedGroup, ModelManifest, VariantName } from "dianome";
import { forwardBlock, type BlockWeights, type ModelConfig, type Stage, type Workspace, stageOutput } from "./block";
import { f16ToF32 } from "./f16";
import { createStorage, readback, uploadWeight, type WeightBuffer } from "./gpu/buffers";
import { GpuOps, type GpuOpsOptions, type GpuTensor } from "./gpu/ops";
import { createKv, KvState, type GpuKv } from "./kv";
import { LmHead, type HeadMatvec } from "./lmhead";

export { acquireDevice, type AcquiredDevice, type GpuInfo } from "./gpu/device";
export { Tokenizer, tokenizerFromBytes, type TokenizerJson } from "./tokenizer";
export { GpuOps, type GpuTensor, type GpuOpsOptions } from "./gpu/ops";
export { uploadWeight, uploadF32AsF16, readback, type WeightBuffer } from "./gpu/buffers";
export { forwardBlock, STAGES, type BlockWeights, type BlockOps, type ModelConfig, type Stage, type Workspace } from "./block";
export { CpuOps, cpuKv, cpuEmbed, type CpuWeight, type CpuKv } from "./cpu";
export { ropeTable, type RopeTable } from "./rope";
export { f16ToF32, f32ToF16, roundF16, f16ArrayToF32, f32ArrayToF16, f16BytesToF32 } from "./f16";
export { parseNpy, npyF16Bits, type Npy } from "./npy";
export { q8Encode, q8Decode, q4Encode, q4Decode, q8Bytes, q4Bytes, Q4_GROUP } from "./quant";
export { createKv, KvState, type GpuKv } from "./kv";
export { loadTokenizer, fetchStoreFile } from "./loadTokenizer";
export { SplitClient, encodeFrame, decodeFrame, type Message, type TokenMessage } from "./protocol";
export { LmHead, Sampler, Prng, argmax, topk, type SamplingOptions, type HeadMatvec } from "./lmhead";
export { renderChat, QWEN_DEFAULT_SYSTEM, IM_START, IM_END, type ChatMessage, type ChatRole, type ChatTemplateOptions } from "./chat";
export { SplitSession, type SessionMode, type SessionOptions, type StepTiming, type GeneratedToken } from "./session";
export { plan, feasibleN, serverShare, DEFAULT_GPU_BUDGET, type PlanInput, type Plan, type PlanCandidate, type PlanModel, type PlanDevice, type PlanNetwork, type PlanServer, type PlanPolicy, type PlanPrompt, type PrivacyBand, type PrivacyRow } from "./planner";
export { microbench, syntheticBlockWeights, type Microbench, type MicrobenchOptions } from "./microbench";

export interface RuntimeOptions {
  /** Emulate fp16 PyTorch rounding points (default true; the fixtures were produced by an fp16 model). */
  round16?: boolean;
  attnFlags?: number;
  rope?: GpuOpsOptions["rope"];
  forceTiled?: boolean;
  siluMode?: number;
  /** T = 1 kernel; default "seq" when round16 is on, "lanes" when off (see README). */
  matvec?: "lanes" | "seq";
  /** Local mode (Phase 5b): also load final_norm + lm_head (tied → the embed bytes) and expose logits(). Needs N = L. */
  lmHead?: boolean;
  /** Kernel for the lm_head matvec; default "auto" = the runtime's T = 1 rule. */
  lmHeadMatvec?: HeadMatvec | "auto";
}

export interface RuntimeStats {
  N: number;
  maxCtx: number;
  variant: string;
  position: number;
  blocksLoaded: number;
  gpuBytes: number;
  weightBytes: number;
  kvBytes: number;
  workspaceBytes: number;
  dispatches: number;
  /** Wall-clock ms of the last prefill / decode / export (submit to completion). */
  lastPrefillMs: number;
  lastPrefillTokens: number;
  lastDecodeMs: number;
  lastExportMs: number;
  /** Wall-clock ms of the last logits() (final norm + lm_head + readback); 0 without lmHead. */
  lastLmHeadMs: number;
  lmHead: boolean;
  timeToFirstBlockMs: number | null;
  loadMs: number | null;
}

export function configFromManifest(manifest: ModelManifest): ModelConfig {
  const c = manifest.config;
  return {
    d: c.hidden_size, heads: c.num_attention_heads, kvHeads: c.num_key_value_heads,
    headDim: c.hidden_size / c.num_attention_heads, inter: c.intermediate_size,
    eps: c.rms_norm_eps, ropeTheta: c.rope_theta, layers: c.num_hidden_layers,
  };
}

const ROLES: Record<keyof BlockWeights<unknown>, string> = {
  inputNorm: "input_layernorm.weight",
  q: "attn.q_proj.weight", qBias: "attn.q_proj.bias",
  k: "attn.k_proj.weight", kBias: "attn.k_proj.bias",
  v: "attn.v_proj.weight", vBias: "attn.v_proj.bias",
  o: "attn.o_proj.weight",
  postNorm: "post_attention_layernorm.weight",
  gate: "mlp.gate_proj.weight", up: "mlp.up_proj.weight", down: "mlp.down_proj.weight",
};

interface EmbedTable { kind: "fp16" | "q8"; bytes: Uint8Array; scales?: Uint8Array; d: number; vocab: number }

const now = (): number => performance.now();

export class Runtime {
  readonly cfg: ModelConfig;
  readonly ops: GpuOps;
  readonly ws: Workspace<GpuTensor>;
  readonly kvs: GpuKv[] = [];
  readonly kvState: KvState;
  readonly blocks: (BlockWeights<WeightBuffer> | null)[];
  private embedTable: EmbedTable | null = null;
  readonly lmHead: boolean;
  private readonly lmHeadMatvec: HeadMatvec | "auto";
  private lmHeadTied = false;
  private finalNorm: WeightBuffer | null = null;
  private lmHeadW: WeightBuffer | null = null;
  private head: LmHead | null = null;
  private readonly exportBuf: GPUBuffer;
  private readonly weightTally = { bytes: 0 };
  private readonly kvTally = { bytes: 0 };
  private lastT = 0;
  /** When set, every block's output rows are copied here ([N, maxCtx, d] f32) during forward (gates 4/5). */
  private traceBuf: GPUBuffer | null = null;
  private embedRows: Float32Array = new Float32Array(0);
  private readonly t = { prefillMs: 0, prefillTokens: 0, decodeMs: 0, exportMs: 0, lmHeadMs: 0, firstBlock: null as number | null, load: null as number | null };
  private loadStart: number | null = null;

  private constructor(readonly device: GPUDevice, readonly manifest: ModelManifest, readonly variant: VariantName, readonly N: number, readonly maxCtx: number, opts: RuntimeOptions) {
    this.cfg = configFromManifest(manifest);
    if (!(N >= 1 && N <= this.cfg.layers)) throw new Error(`N must be in 1..${this.cfg.layers}, got ${N}`);
    if (!manifest.variants[variant]) throw new Error(`variant ${variant} not in manifest ${manifest.id}`);
    this.lmHead = opts.lmHead ?? false;
    this.lmHeadMatvec = opts.lmHeadMatvec ?? "auto";
    if (this.lmHead && N !== this.cfg.layers) throw new Error(`lmHead needs N = L (${this.cfg.layers}), got ${N}`);
    if (this.lmHead) {
      const lg = manifest.variants[variant]!.groups.find((g) => g.name === "lm_head");
      const le = lg?.entries.find((e) => e.role === "lm_head.weight");
      if (!le) throw new Error("manifest has no lm_head.weight entry");
      this.lmHeadTied = le.tied === true;
    }
    const ropeOpt = opts.rope !== undefined ? { rope: opts.rope } : {};
    const attnOpt = opts.attnFlags !== undefined ? { attnFlags: opts.attnFlags } : {};
    const tiledOpt = opts.forceTiled !== undefined ? { forceTiled: opts.forceTiled } : {};
    const siluOpt = opts.siluMode !== undefined ? { siluMode: opts.siluMode } : {};
    const mvOpt = opts.matvec !== undefined ? { matvec: opts.matvec } : {};
    this.ops = new GpuOps(device, this.cfg, { maxCtx, round16: opts.round16 ?? true, ...ropeOpt, ...attnOpt, ...tiledOpt, ...siluOpt, ...mvOpt });
    const { d, heads, kvHeads, headDim, inter } = this.cfg;
    const mk = (label: string, cols: number) => this.ops.tensor(label, maxCtx, cols);
    this.ws = {
      x: mk("ws.x", d), x2: mk("ws.x2", d), h: mk("ws.h", d),
      q: mk("ws.q", heads * headDim), k: mk("ws.k", kvHeads * headDim), v: mk("ws.v", kvHeads * headDim),
      attn: mk("ws.attn", heads * headDim), o: mk("ws.o", d),
      gate: mk("ws.gate", inter), up: mk("ws.up", inter), act: mk("ws.act", inter), mlp: mk("ws.mlp", d),
    };
    this.exportBuf = createStorage(device, "export.f16", maxCtx * d * 2, this.ops.tally);
    for (let i = 0; i < N; i++) this.kvs.push(createKv(device, this.cfg, maxCtx, `kv.${i}`, this.kvTally));
    this.kvState = new KvState(maxCtx);
    this.blocks = new Array(N).fill(null);
  }

  static async create(device: GPUDevice, manifest: ModelManifest, variant: VariantName, N: number, maxCtx: number, opts: RuntimeOptions = {}): Promise<Runtime> {
    return new Runtime(device, manifest, variant, N, maxCtx, opts);
  }

  // -- loading -------------------------------------------------------------------------

  /** Uploads a group from the SDK stream. Returns true once every needed group (embed + N layers) is in. */
  addGroup(group: LoadedGroup): boolean {
    if (this.loadStart === null) this.loadStart = now();
    if (group.name === "embed") {
      const e = [...group.entries.values()].find((x) => x.role === "embed_tokens.weight");
      if (!e) throw new Error("embed group has no embed_tokens.weight");
      this.setEmbed(e);
      // Tied lm_head: the same bytes go to the GPU once more as the head's weight (the manifest's lm_head group is empty).
      if (this.lmHead && this.lmHeadTied && !this.lmHeadW) this.lmHeadW = uploadWeight(this.device, { ...e, name: "lm_head.weight" }, this.weightTally);
    } else if (group.name === "final_norm") {
      if (this.lmHead && !this.finalNorm) {
        const e = [...group.entries.values()].find((x) => x.role === "norm.weight");
        if (!e) throw new Error("final_norm group has no norm.weight");
        this.finalNorm = uploadWeight(this.device, e, this.weightTally);
      }
    } else if (group.name === "lm_head") {
      if (this.lmHead && !this.lmHeadW) {
        const e = [...group.entries.values()].find((x) => x.role === "lm_head.weight");
        if (!e) throw new Error("lm_head group has no lm_head.weight");
        this.lmHeadW = uploadWeight(this.device, e, this.weightTally);
      }
    } else {
      const m = /^layer\.(\d+)$/.exec(group.name);
      if (m) {
        const i = Number(m[1]);
        if (i < this.N) {
          this.blocks[i] = this.uploadBlock(group);
          if (i === 0 && this.t.firstBlock === null) this.t.firstBlock = now() - this.loadStart;
        }
      }
    }
    const done = this.ready;
    if (done && this.t.load === null) this.t.load = now() - this.loadStart;
    return done;
  }

  get ready(): boolean {
    return this.embedTable !== null && this.blocks.every((b) => b !== null) && (!this.lmHead || (this.finalNorm !== null && this.lmHeadW !== null));
  }

  private getHead(): LmHead {
    if (!this.lmHead) throw new Error("runtime created without lmHead: true");
    if (!this.finalNorm || !this.lmHeadW) throw new Error("final_norm / lm_head not loaded");
    return (this.head ??= new LmHead(this.ops, this.finalNorm, this.lmHeadW, this.lmHeadMatvec));
  }

  /** Consumes a stream (`dianome.stream(id, { variant })`) until embed and layers 0..N-1 have arrived. */
  async loadFrom(stream: AsyncIterable<LoadedGroup>): Promise<void> {
    for await (const g of stream) if (this.addGroup(g)) break;
    if (!this.ready) throw new Error("stream ended before every needed group arrived");
  }

  setEmbed(e: LoadedEntry): void {
    const [vocab, d] = [e.shape[0]!, e.shape[1]!];
    if (d !== this.cfg.d) throw new Error(`embed width ${d} != d_model ${this.cfg.d}`);
    if (e.storage.kind === "fp16") this.embedTable = { kind: "fp16", bytes: e.bytes, d, vocab };
    else if (e.storage.kind === "q8" && e.parts) this.embedTable = { kind: "q8", bytes: e.parts.weights, scales: e.parts.scales, d, vocab };
    else throw new Error(`embed storage ${e.storage.kind} unsupported`);
  }

  uploadBlock(group: LoadedGroup): BlockWeights<WeightBuffer> {
    const byRole = new Map<string, LoadedEntry>();
    for (const e of group.entries.values()) byRole.set(e.role, e);
    const get = (role: string): WeightBuffer => {
      const e = byRole.get(role);
      if (!e) throw new Error(`${group.name}: missing ${role}`);
      return uploadWeight(this.device, e, this.weightTally);
    };
    const w = {} as BlockWeights<WeightBuffer>;
    for (const key of Object.keys(ROLES) as (keyof BlockWeights<unknown>)[]) w[key] = get(ROLES[key]);
    return w;
  }

  /** Embedding rows for `tokens` as f32 (CPU gather from the entry bytes: fp16 or q8 dequantised). */
  embed(tokens: ArrayLike<number>): Float32Array {
    const e = this.embedTable;
    if (!e) throw new Error("embed group not loaded");
    const T = tokens.length, d = e.d;
    const out = new Float32Array(T * d);
    const dv = new DataView(e.bytes.buffer, e.bytes.byteOffset, e.bytes.byteLength);
    for (let t = 0; t < T; t++) {
      const id = tokens[t]!;
      if (!(id >= 0 && id < e.vocab)) throw new Error(`token id ${id} out of range`);
      if (e.kind === "fp16") {
        for (let i = 0; i < d; i++) out[t * d + i] = f16ToF32(dv.getUint16((id * d + i) * 2, true));
      } else {
        const sdv = new DataView(e.scales!.buffer, e.scales!.byteOffset, e.scales!.byteLength);
        const s = f16ToF32(sdv.getUint16(id * 2, true));
        for (let i = 0; i < d; i++) out[t * d + i] = Math.fround(dv.getInt8(id * d + i) * s);
      }
    }
    return out;
  }

  // -- inference -----------------------------------------------------------------------

  reset(): void { this.kvState.reset(); this.lastT = 0; }

  private assertReady(): void { if (!this.ready) throw new Error("runtime not loaded (embed + layers 0..N-1)"); }

  /** Runs blocks 0..N-1 on the f32 rows already in ws.x (positions pos0..pos0+T-1). */
  private forward(T: number, pos0: number, untilBlock?: number, untilStage?: Stage, fromBlock = 0): void {
    this.kvState.advance(pos0, T);
    this.lastT = T;
    this.ops.run(() => {
      const last = untilBlock ?? this.N - 1;
      for (let i = fromBlock; i <= last; i++) {
        const stage = forwardBlock(this.ops, this.blocks[i]!, this.ws, this.kvs[i]!, T, pos0, i === last ? untilStage : undefined);
        if (this.traceBuf && stage === "residual2") this.ops.copy(this.ws.x.buffer, 0, this.traceBuf, i * this.maxCtx * this.cfg.d * 4, T * this.cfg.d * 4);
      }
    }, `forward T=${T} pos0=${pos0}`);
  }

  /** Prefill `tokens` at positions position..position+T-1 (position must continue the cache; 0 after reset). */
  async prefill(tokens: ArrayLike<number>, position = this.kvState.length): Promise<void> {
    this.assertReady();
    const T = tokens.length;
    if (T < 1) throw new Error("prefill needs at least one token");
    const t0 = now();
    this.embedRows = this.embed(tokens);
    this.ops.write(this.ws.x, this.embedRows);
    this.forward(T, position);
    await this.device.queue.onSubmittedWorkDone();
    this.t.prefillMs = now() - t0;
    this.t.prefillTokens = T;
  }

  /** One token at `position`. */
  async decode(token: number, position = this.kvState.length): Promise<void> {
    this.assertReady();
    const t0 = now();
    this.embedRows = this.embed([token]);
    this.ops.write(this.ws.x, this.embedRows);
    this.forward(1, position);
    await this.device.queue.onSubmittedWorkDone();
    this.t.decodeMs = now() - t0;
  }

  /** Local mode: final norm + lm_head on the last row of the last forward → f32 logits [vocab]. */
  async logits(): Promise<Float32Array> {
    const T = this.lastT;
    if (T === 0) throw new Error("nothing to score: run prefill or decode first");
    const head = this.getHead();
    const t0 = now();
    this.ops.run(() => head.record(this.ws.x, T), "lm_head");
    const out = await head.read();
    this.t.lmHeadMs = now() - t0;
    return out;
  }

  /** Test hook (lm_head gate): logits for every row of the last forward, [T, vocab] f32. */
  debugLogitsAll(): Promise<Float32Array> {
    const T = this.lastT;
    if (T === 0) throw new Error("nothing to score: run prefill or decode first");
    return this.getHead().logitsAll(this.ws.x, T);
  }

  /** The last forward's output rows (the input to block N) as fp16 bits, row-major [T, d_model]. */
  async exportHidden(): Promise<Uint16Array> {
    const T = this.lastT;
    if (T === 0) throw new Error("nothing to export: run prefill or decode first");
    const t0 = now();
    this.ops.run(() => this.ops.packF16(this.ws.x, T, this.exportBuf), "export");
    const buf = await readback(this.device, this.exportBuf, T * this.cfg.d * 2);
    this.t.exportMs = now() - t0;
    return new Uint16Array(buf);
  }

  stats(): RuntimeStats {
    const workspaceBytes = this.ops.tally.bytes + (this.head?.tally.bytes ?? 0);
    return {
      N: this.N, maxCtx: this.maxCtx, variant: this.variant, position: this.kvState.length,
      blocksLoaded: this.blocks.filter(Boolean).length,
      gpuBytes: workspaceBytes + this.weightTally.bytes + this.kvTally.bytes,
      weightBytes: this.weightTally.bytes, kvBytes: this.kvTally.bytes, workspaceBytes,
      dispatches: this.ops.counts.dispatches,
      lastPrefillMs: this.t.prefillMs, lastPrefillTokens: this.t.prefillTokens, lastDecodeMs: this.t.decodeMs, lastExportMs: this.t.exportMs,
      lastLmHeadMs: this.t.lmHeadMs, lmHead: this.lmHead,
      timeToFirstBlockMs: this.t.firstBlock, loadMs: this.t.load,
    };
  }

  // -- test hooks (gates 2–5) ------------------------------------------------------------

  /** Writes f32 rows into ws.x and runs blocks 0..untilBlock, stopping after `stage` in the last one. */
  async debugForward(rows: Float32Array, T: number, pos0: number, untilBlock: number, stage?: Stage, fromBlock = 0): Promise<void> {
    this.ops.write(this.ws.x, rows);
    this.forward(T, pos0, untilBlock, stage, fromBlock);
    await this.device.queue.onSubmittedWorkDone();
  }

  /** Reads the workspace buffer holding `stage`'s result (rows 0..T-1). */
  debugRead(stage: Stage, T: number): Promise<Float32Array> {
    return this.ops.read(this.ws[stageOutput(stage)], T);
  }

  /** Enables per-block output tracing (allocates N * maxCtx * d f32). */
  enableTrace(): void {
    this.traceBuf ??= createStorage(this.device, "trace", this.N * this.maxCtx * this.cfg.d * 4);
  }

  /** Rows 0..T-1 of block i's output from the last traced forward. */
  async traceRead(block: number, T: number): Promise<Float32Array> {
    if (!this.traceBuf) throw new Error("trace not enabled");
    return new Float32Array(await readback(this.device, this.traceBuf, T * this.cfg.d * 4, block * this.maxCtx * this.cfg.d * 4));
  }

  /** Reads block i's K or V cache rows 0..rows-1 as f32. */
  debugReadKv(block: number, which: "k" | "v", rows: number): Promise<Float32Array> {
    const kv = this.kvs[block]!;
    return this.ops.readKv(which === "k" ? kv.k : kv.v, kv.cols, rows);
  }

  destroy(): void {
    for (const b of this.blocks) if (b) for (const w of Object.values(b)) (w as WeightBuffer).buffer.destroy();
    for (const t of Object.values(this.ws)) (t as GpuTensor).buffer.destroy();
    for (const kv of this.kvs) { kv.k.destroy(); kv.v.destroy(); }
    this.exportBuf.destroy();
    this.traceBuf?.destroy();
    if (this.head) this.head.destroy();
    else { this.finalNorm?.buffer.destroy(); this.lmHeadW?.buffer.destroy(); }
    this.ops.destroy();
  }
}
