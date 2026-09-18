// Minimal .npy reader (numpy v1/v2 header, C order) for the fixtures: '<f2' -> Float32Array (converted), '<f4',
// '<i4', '<i8' (as Float64Array of the values), '|u1', '|i1'. Used by the tests and the bench page, not by the runtime.

import { f16ToF32 } from "./f16";

export interface Npy { shape: number[]; dtype: string; data: Float32Array | Int32Array | Uint8Array | Int8Array | Float64Array }

export function parseNpy(buf: ArrayBuffer): Npy {
  const u8 = new Uint8Array(buf);
  if (u8[0] !== 0x93 || String.fromCharCode(...u8.subarray(1, 6)) !== "NUMPY") throw new Error("not an .npy file");
  const major = u8[6]!;
  const dv = new DataView(buf);
  const headerLen = major === 1 ? dv.getUint16(8, true) : dv.getUint32(8, true);
  const start = (major === 1 ? 10 : 12);
  const header = new TextDecoder("latin1").decode(u8.subarray(start, start + headerLen));
  const descr = /'descr':\s*'([^']+)'/.exec(header)?.[1];
  const fortran = /'fortran_order':\s*(True|False)/.exec(header)?.[1];
  const shapeStr = /'shape':\s*\(([^)]*)\)/.exec(header)?.[1];
  if (!descr || fortran === undefined || shapeStr === undefined) throw new Error(`bad .npy header: ${header}`);
  if (fortran === "True") throw new Error("fortran_order .npy not supported");
  const shape = shapeStr.split(",").map((s) => s.trim()).filter(Boolean).map(Number);
  const count = shape.reduce((a, b) => a * b, 1);
  const off = start + headerLen;
  let data: Npy["data"];
  switch (descr) {
    case "<f2": {
      const out = new Float32Array(count);
      for (let i = 0; i < count; i++) out[i] = f16ToF32(dv.getUint16(off + i * 2, true));
      data = out; break;
    }
    case "<f4": { const out = new Float32Array(count); for (let i = 0; i < count; i++) out[i] = dv.getFloat32(off + i * 4, true); data = out; break; }
    case "<i4": { const out = new Int32Array(count); for (let i = 0; i < count; i++) out[i] = dv.getInt32(off + i * 4, true); data = out; break; }
    case "<i8": { const out = new Float64Array(count); for (let i = 0; i < count; i++) out[i] = Number(dv.getBigInt64(off + i * 8, true)); data = out; break; }
    case "|u1": data = u8.slice(off, off + count); break;
    case "|i1": data = new Int8Array(buf.slice(off, off + count)); break;
    default: throw new Error(`unsupported .npy dtype ${descr}`);
  }
  return { shape, dtype: descr, data };
}

/** Raw fp16 payload of a '<f2' .npy as Uint16Array (no conversion). */
export function npyF16Bits(buf: ArrayBuffer): { shape: number[]; bits: Uint16Array } {
  const u8 = new Uint8Array(buf);
  const major = u8[6]!;
  const dv = new DataView(buf);
  const headerLen = major === 1 ? dv.getUint16(8, true) : dv.getUint32(8, true);
  const start = (major === 1 ? 10 : 12);
  const header = new TextDecoder("latin1").decode(u8.subarray(start, start + headerLen));
  if (!/'descr':\s*'<f2'/.test(header)) throw new Error("not an fp16 .npy");
  const shape = (/'shape':\s*\(([^)]*)\)/.exec(header)?.[1] ?? "").split(",").map((s) => s.trim()).filter(Boolean).map(Number);
  const count = shape.reduce((a, b) => a * b, 1);
  const off = start + headerLen;
  const bits = new Uint16Array(count);
  for (let i = 0; i < count; i++) bits[i] = dv.getUint16(off + i * 2, true);
  return { shape, bits };
}
