// Shared head of the matmul kernels: params, activations, bias, helpers. A weights_*.wgsl snippet follows
// (binding 3, and 5/6 for q8/q4) and then matmul_tiled.wgsl or matvec.wgsl supplies main().
//   y[T, out] = x[T, in] · W[out, in]^T (+ bias)      flags: bit0 = add bias, bit1 = round result to fp16

struct Params {
  T: u32,
  in_dim: u32,
  out_dim: u32,
  flags: u32,
}

@group(0) @binding(0) var<uniform> P: Params;
@group(0) @binding(1) var<storage, read> x: array<f32>;
@group(0) @binding(2) var<storage, read_write> y: array<f32>;
@group(0) @binding(4) var<storage, read> bias: array<u32>;   // fp16 pairs [out/2]

fn r16(v: f32) -> f32 {
  return unpack2x16float(pack2x16float(vec2<f32>(v, 0.0))).x;
}

fn f16_at(buf_word: u32, i: u32) -> f32 {
  let p = unpack2x16float(buf_word);
  return select(p.x, p.y, (i & 1u) == 1u);
}

fn bias_at(o: u32) -> f32 {
  return f16_at(bias[o >> 1u], o);
}

fn finish(o: u32, acc: f32) -> f32 {
  var v = acc * out_scale(o);
  if ((P.flags & 1u) != 0u) { v = v + bias_at(o); }
  if ((P.flags & 2u) != 0u) { v = r16(v); }
  return v;
}
