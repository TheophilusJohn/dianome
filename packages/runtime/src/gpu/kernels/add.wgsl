// y = a + b over n values (residual add). round16: the sum is rounded to fp16 like an fp16 tensor add.

struct Params {
  n: u32,
  round16: u32,
  _p0: u32,
  _p1: u32,
}

@group(0) @binding(0) var<uniform> P: Params;
@group(0) @binding(1) var<storage, read> a: array<f32>;
@group(0) @binding(2) var<storage, read> b: array<f32>;
@group(0) @binding(3) var<storage, read_write> y: array<f32>;

fn r16(v: f32) -> f32 {
  return unpack2x16float(pack2x16float(vec2<f32>(v, 0.0))).x;
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= P.n) { return; }
  var v = a[i] + b[i];
  if (P.round16 != 0u) { v = r16(v); }
  y[i] = v;
}
