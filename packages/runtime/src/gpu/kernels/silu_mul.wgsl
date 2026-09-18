// y = silu(gate) * up, elementwise over n values. round16: silu(gate) and the product are each rounded to fp16.

struct Params {
  n: u32,
  round16: u32,
  _p0: u32,
  _p1: u32,
}

@group(0) @binding(0) var<uniform> P: Params;
@group(0) @binding(1) var<storage, read> gate: array<f32>;
@group(0) @binding(2) var<storage, read> up: array<f32>;
@group(0) @binding(3) var<storage, read_write> y: array<f32>;

fn r16(v: f32) -> f32 {
  return unpack2x16float(pack2x16float(vec2<f32>(v, 0.0))).x;
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= P.n) { return; }
  let g = gate[i];
  var s: f32;
  if ((P.round16 & 2u) != 0u) {
    var sg = 1.0 / (1.0 + exp(-g));
    if ((P.round16 & 1u) != 0u) { sg = r16(sg); }
    s = g * sg;
  } else {
    s = g / (1.0 + exp(-g));
  }
  if ((P.round16 & 1u) != 0u) { s = r16(s); }
  var v = s * up[i];
  if ((P.round16 & 1u) != 0u) { v = r16(v); }
  y[i] = v;
}
