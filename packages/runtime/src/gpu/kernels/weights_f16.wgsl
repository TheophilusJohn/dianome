// fp16 weights [out, in] packed two per u32 (element 2i in the low half).
@group(0) @binding(3) var<storage, read> w: array<u32>;

// W[o][k], W[o][k+1] for even k.
fn load_w2(o: u32, k: u32) -> vec2<f32> {
  return unpack2x16float(w[(o * P.in_dim + k) >> 1u]);
}

fn out_scale(o: u32) -> f32 {
  return 1.0;
}
