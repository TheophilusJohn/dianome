// q8 weights: int8 [out, in] four per u32 (row-major, little-endian bytes), fp16 scale[out] applied once per
// output in out_scale (sum(x·q)·scale == sum(x·(q·scale)) up to f32 rounding).
@group(0) @binding(3) var<storage, read> w: array<u32>;
@group(0) @binding(5) var<storage, read> scales: array<u32>;   // fp16 pairs [out/2]

fn load_w2(o: u32, k: u32) -> vec2<f32> {
  let idx = o * P.in_dim + k;          // even
  let word = w[idx >> 2u];
  let sh = (idx & 3u) * 8u;            // 0 or 16
  let a = f32(i32(word << (24u - sh)) >> 24u);
  let b = f32(i32(word << (16u - sh)) >> 24u);
  return vec2<f32>(a, b);
}

fn out_scale(o: u32) -> f32 {
  let p = unpack2x16float(scales[o >> 1u]);
  return select(p.x, p.y, (o & 1u) == 1u);
}
