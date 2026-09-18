// KV cache: per block, fp16 K and V rows for `maxCtx` positions, allocated once at create(). Row p holds the
// key/value of absolute position p (kv_heads * head_dim fp16 values packed as u32 pairs); rope_kv writes rows,
// attention reads rows 0..pos0+T-1. `length` is the number of valid rows, advanced by the runtime.

import type { ModelConfig } from "./block";
import { createStorage } from "./gpu/buffers";

export interface GpuKv { k: GPUBuffer; v: GPUBuffer; cols: number; rows: number; bytes: number }

export function kvBytes(cfg: ModelConfig, maxCtx: number): number {
  return 2 * maxCtx * cfg.kvHeads * cfg.headDim * 2;
}

export function createKv(device: GPUDevice, cfg: ModelConfig, maxCtx: number, label: string, tally?: { bytes: number }): GpuKv {
  const cols = cfg.kvHeads * cfg.headDim;
  const bytes = maxCtx * cols * 2;
  return { k: createStorage(device, `${label}.k`, bytes, tally), v: createStorage(device, `${label}.v`, bytes, tally), cols, rows: maxCtx, bytes: bytes * 2 };
}

/** Row index for (position) and word index for (position, column) inside the packed cache. */
export function kvWord(kv: GpuKv, position: number, col: number): number {
  return (position * kv.cols + col) >> 1;
}

export class KvState {
  length = 0;
  constructor(readonly maxCtx: number) {}
  /** Reserve T rows at `position`; positions must be contiguous with what is cached. */
  advance(position: number, T: number): void {
    if (position !== this.length) throw new Error(`KV cache: positions must be contiguous (have ${this.length}, got ${position})`);
    if (position + T > this.maxCtx) throw new Error(`KV cache: ${position + T} exceeds maxCtx ${this.maxCtx}`);
    this.length = position + T;
  }
  reset(): void { this.length = 0; }
}
