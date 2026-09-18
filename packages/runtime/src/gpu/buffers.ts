// Weight upload from SDK entries. One GPUBuffer per entry holding the entry's bytes exactly as ingest laid
// them out (weights, then scales, then zeros, each 256-byte aligned), so the parts are bound as offsets into
// the same buffer (minStorageBufferOffsetAlignment is 256). fp16 data is consumed as u32 pairs (element 2i in
// the low half, matching unpack2x16float), int8 as 4 per u32, q4 nibbles as 8 per u32. Never larger than 1 GiB.

import type { LoadedEntry } from "dianome";
import { MAX_BUFFER } from "./device";

export type WeightKind = "fp16" | "q8" | "q4";

export interface WeightBuffer {
  name: string;
  kind: WeightKind;
  shape: number[];
  buffer: GPUBuffer;
  bytes: number;
  /** Byte offset/length of each part inside `buffer` (fp16: weights only). */
  parts: { weights: { offset: number; length: number }; scales?: { offset: number; length: number }; zeros?: { offset: number; length: number } };
  groupSize?: number;
}

export interface WeightSource {
  name: string;
  shape: number[];
  storage: LoadedEntry["storage"];
  bytes: Uint8Array;
  parts?: LoadedEntry["parts"];
}

function pad4(n: number): number { return (n + 3) & ~3; }

/** Uploads an entry (or anything shaped like one). The source bytes may alias a chunk buffer; they are copied. */
export function uploadWeight(device: GPUDevice, src: WeightSource, tally?: { bytes: number }): WeightBuffer {
  const st = src.storage;
  if (st.kind === "raw") throw new Error(`${src.name}: raw entries are not weights`);
  const size = pad4(src.bytes.byteLength);
  if (size > MAX_BUFFER) throw new Error(`${src.name}: ${size} bytes exceeds the 1 GiB buffer limit`);
  const buffer = device.createBuffer({ label: src.name, size, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST, mappedAtCreation: true });
  new Uint8Array(buffer.getMappedRange()).set(src.bytes);
  buffer.unmap();
  if (tally) tally.bytes += size;
  const base = { name: src.name, kind: st.kind as WeightKind, shape: src.shape, buffer, bytes: size };
  if (st.kind === "fp16") return { ...base, parts: { weights: { offset: 0, length: src.bytes.byteLength } } };
  if (st.kind === "q8") return { ...base, parts: { weights: { ...st.parts.weights }, scales: { ...st.parts.scales } } };
  return { ...base, groupSize: st.group_size, parts: { weights: { ...st.parts.weights }, scales: { ...st.parts.scales }, zeros: { ...st.parts.zeros } } };
}

/** A binding for one part (the length is rounded up to 4 bytes, which stays inside the padded buffer). */
export function partBinding(w: WeightBuffer, part: "weights" | "scales" | "zeros"): GPUBufferBinding {
  const p = w.parts[part];
  if (!p) throw new Error(`${w.name}: no ${part} part`);
  return { buffer: w.buffer, offset: p.offset, size: pad4(p.length) };
}

/** Uploads fp16-representable f32 values as an fp16 weight (tests and synthetic models). */
export function uploadF32AsF16(device: GPUDevice, name: string, shape: number[], values: Float32Array, f32ToF16: (v: number) => number, tally?: { bytes: number }): WeightBuffer {
  const u16 = new Uint16Array(values.length);
  for (let i = 0; i < values.length; i++) u16[i] = f32ToF16(values[i]!);
  const bytes = new Uint8Array(u16.buffer);
  return uploadWeight(device, { name, shape, storage: { kind: "fp16" }, bytes }, tally);
}

export function createStorage(device: GPUDevice, label: string, bytes: number, tally?: { bytes: number }): GPUBuffer {
  const size = pad4(Math.max(bytes, 4));
  if (size > MAX_BUFFER) throw new Error(`${label}: ${size} bytes exceeds the 1 GiB buffer limit`);
  if (tally) tally.bytes += size;
  return device.createBuffer({ label, size, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST });
}

export function createUniform(device: GPUDevice, label: string, bytes: number): GPUBuffer {
  return device.createBuffer({ label, size: Math.max(16, pad4(bytes)), usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
}

/** Reads `bytes` from a storage buffer (one staging buffer per call; tests and the export path). */
export async function readback(device: GPUDevice, src: GPUBuffer, bytes: number, srcOffset = 0): Promise<ArrayBuffer> {
  const size = pad4(bytes);
  const staging = device.createBuffer({ size, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
  const enc = device.createCommandEncoder();
  enc.copyBufferToBuffer(src, srcOffset, staging, 0, size);
  device.queue.submit([enc.finish()]);
  await staging.mapAsync(GPUMapMode.READ);
  const out = staging.getMappedRange().slice(0, bytes);
  staging.unmap();
  staging.destroy();
  return out;
}
