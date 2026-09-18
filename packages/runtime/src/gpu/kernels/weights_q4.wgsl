// q4 weights: nibbles [out, in] eight per u32 (element i in the low nibble of byte i/2 when even, high when
// odd), fp16 scale[out, in/128] and u8 zero[out, in/128] per group of 128 along `in`, applied at load.
@group(0) @binding(3) var<storage, read> w: array<u32>;
@group(0) @binding(5) var<storage, read> scales: array<u32>;   // fp16 pairs
@group(0) @binding(6) var<storage, read> zeros: array<u32>;    // u8 x4

fn load_w2(o: u32, k: u32) -> vec2<f32> {
  let idx = o * P.in_dim + k;          // even
  let word = w[idx >> 3u];
  let sh = (idx & 7u) * 4u;
  let qa = f32((word >> sh) & 15u);
  let qb = f32((word >> (sh + 4u)) & 15u);
  let g = o * (P.in_dim >> 7u) + (k >> 7u);
  let sp = unpack2x16float(scales[g >> 1u]);
  let s = select(sp.x, sp.y, (g & 1u) == 1u);
  let z = f32((zeros[g >> 2u] >> ((g & 3u) * 8u)) & 255u);
  return vec2<f32>((qa - z) * s, (qb - z) * s);
}

fn out_scale(o: u32) -> f32 {
  return 1.0;
}
