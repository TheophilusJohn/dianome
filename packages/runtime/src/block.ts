// One Qwen2 transformer block as a fixed sequence of ops, generic over the backend (GPU ops record into a
// command encoder; the CPU reference computes immediately). The order is the model's:
//   x → rmsnorm(input_layernorm) → q,k,v = matmul+bias → RoPE(q,k), cache k,v → attention (GQA, causal)
//     → o_proj → x += attn → rmsnorm(post_attention_layernorm) → gate,up = matmul → silu(gate)*up → down_proj → x += mlp
// `until` stops after a named stage so a test can read the intermediate (gate 3).

export interface ModelConfig {
  d: number;          // hidden_size
  heads: number;      // num_attention_heads
  kvHeads: number;    // num_key_value_heads
  headDim: number;
  inter: number;      // intermediate_size
  eps: number;        // rms_norm_eps
  ropeTheta: number;
  layers: number;
}

export interface BlockWeights<W> {
  inputNorm: W;
  q: W; qBias: W;
  k: W; kBias: W;
  v: W; vBias: W;
  o: W;
  postNorm: W;
  gate: W; up: W; down: W;
}

/** Activation buffers shared by every block; each holds `maxCtx` rows. */
export interface Workspace<A> {
  x: A; x2: A; h: A;
  q: A; k: A; v: A;
  attn: A; o: A;
  gate: A; up: A; act: A; mlp: A;
}

export interface BlockOps<W, A, K> {
  rmsnorm(x: A, w: W, out: A, T: number): void;
  /** out[T, out] = x[T, in] · W[out, in]^T (+ bias). */
  matmul(x: A, w: W, bias: W | null, out: A, T: number): void;
  /** RoPE on q (in place) and k; k and v go into the cache at rows pos0..pos0+T-1. */
  ropeKv(q: A, k: A, v: A, T: number, pos0: number, kv: K): void;
  /** Causal attention of the T query rows (positions pos0..) over the cache rows 0..pos0+T-1. */
  attention(q: A, kv: K, out: A, T: number, pos0: number): void;
  siluMul(gate: A, up: A, out: A, T: number): void;
  add(a: A, b: A, out: A, T: number): void;
}

export const STAGES = [
  "input_norm", "q_proj", "k_proj", "v_proj", "rope", "attn", "o_proj", "residual1",
  "post_norm", "gate", "up", "silu_mul", "down", "residual2",
] as const;
export type Stage = (typeof STAGES)[number];

/** Runs one block on ws.x (rows 0..T-1) leaving the output in ws.x. Returns the last stage executed. */
export function forwardBlock<W, A, K>(ops: BlockOps<W, A, K>, w: BlockWeights<W>, ws: Workspace<A>, kv: K, T: number, pos0: number, until?: Stage): Stage {
  let last: Stage = "input_norm";
  const step = (s: Stage, f: () => void): boolean => { f(); last = s; return s === until; };
  if (step("input_norm", () => ops.rmsnorm(ws.x, w.inputNorm, ws.h, T))) return last;
  if (step("q_proj", () => ops.matmul(ws.h, w.q, w.qBias, ws.q, T))) return last;
  if (step("k_proj", () => ops.matmul(ws.h, w.k, w.kBias, ws.k, T))) return last;
  if (step("v_proj", () => ops.matmul(ws.h, w.v, w.vBias, ws.v, T))) return last;
  if (step("rope", () => ops.ropeKv(ws.q, ws.k, ws.v, T, pos0, kv))) return last;
  if (step("attn", () => ops.attention(ws.q, kv, ws.attn, T, pos0))) return last;
  if (step("o_proj", () => ops.matmul(ws.attn, w.o, null, ws.o, T))) return last;
  if (step("residual1", () => ops.add(ws.x, ws.o, ws.x2, T))) return last;
  if (step("post_norm", () => ops.rmsnorm(ws.x2, w.postNorm, ws.h, T))) return last;
  if (step("gate", () => ops.matmul(ws.h, w.gate, null, ws.gate, T))) return last;
  if (step("up", () => ops.matmul(ws.h, w.up, null, ws.up, T))) return last;
  if (step("silu_mul", () => ops.siluMul(ws.gate, ws.up, ws.act, T))) return last;
  if (step("down", () => ops.matmul(ws.act, w.down, null, ws.mlp, T))) return last;
  step("residual2", () => ops.add(ws.x2, ws.mlp, ws.x, T));
  return last;
}

/** Which workspace buffer holds a stage's result (for reading intermediates back). */
export function stageOutput(stage: Stage): keyof Workspace<unknown> {
  switch (stage) {
    case "input_norm": case "post_norm": return "h";
    case "q_proj": case "rope": return "q";
    case "k_proj": return "k";
    case "v_proj": return "v";
    case "attn": return "attn";
    case "o_proj": return "o";
    case "residual1": return "x2";
    case "gate": return "gate";
    case "up": return "up";
    case "silu_mul": return "act";
    case "down": return "mlp";
    case "residual2": return "x";
  }
}
