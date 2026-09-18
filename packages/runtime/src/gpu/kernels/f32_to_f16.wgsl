// Packs 2*n_pairs f32 values into n_pairs u32 words of fp16 pairs (the hidden-state export).

struct Params {
  n_pairs: u32,
  _p0: u32,
  _p1: u32,
  _p2: u32,
}

@group(0) @binding(0) var<uniform> P: Params;
@group(0) @binding(1) var<storage, read> x: array<f32>;
@group(0) @binding(2) var<storage, read_write> y: array<u32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= P.n_pairs) { return; }
  y[i] = pack2x16float(vec2<f32>(x[2u * i], x[2u * i + 1u]));
}
