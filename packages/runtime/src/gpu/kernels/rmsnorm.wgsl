// RMSNorm per row, one workgroup of 256 threads per row: y = x * rsqrt(mean(x^2) + eps) * w.
// Qwen2RMSNorm computes the normalisation in f32, casts to fp16, then multiplies by the fp16 weight; with
// round16 both casts are emulated so the result matches an fp16 PyTorch model.

struct Params {
  rows: u32,
  cols: u32,
  eps: f32,
  round16: u32,
}

@group(0) @binding(0) var<uniform> P: Params;
@group(0) @binding(1) var<storage, read> x: array<f32>;
@group(0) @binding(2) var<storage, read> w: array<u32>;   // fp16 pairs [cols/2]
@group(0) @binding(3) var<storage, read_write> y: array<f32>;

var<workgroup> partial: array<f32, 256>;

fn r16(v: f32) -> f32 {
  return unpack2x16float(pack2x16float(vec2<f32>(v, 0.0))).x;
}

fn w_at(i: u32) -> f32 {
  let p = unpack2x16float(w[i >> 1u]);
  return select(p.x, p.y, (i & 1u) == 1u);
}

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
  let row = wg.x;
  let base = row * P.cols;
  var s = 0.0;
  for (var i = lid.x; i < P.cols; i += 256u) {
    let v = x[base + i];
    s += v * v;
  }
  partial[lid.x] = s;
  workgroupBarrier();
  for (var stride = 128u; stride > 0u; stride >>= 1u) {
    if (lid.x < stride) { partial[lid.x] += partial[lid.x + stride]; }
    workgroupBarrier();
  }
  let inv = 1.0 / sqrt(partial[0] / f32(P.cols) + P.eps);
  for (var i = lid.x; i < P.cols; i += 256u) {
    var v = x[base + i] * inv;
    if (P.round16 != 0u) { v = r16(v); }
    v = v * w_at(i);
    if (P.round16 != 0u) { v = r16(v); }
    y[base + i] = v;
  }
}
