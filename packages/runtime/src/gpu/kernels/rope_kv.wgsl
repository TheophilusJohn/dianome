// RoPE (rotate-half form) on q in place and on k, then k and v are stored as fp16 into the block's KV cache at
// rows pos0..pos0+T-1. One thread per (row, slot): a q slot rotates four elements of one head (2j, 2j+1 and
// their partners 2j+half, 2j+half+1) so the packed fp16 pairs it writes are whole words; a v slot copies a pair.
// cos/sin: [positions, head_dim] f32 tables (fp16-rounded when the model is emulated), second half == first.
// round16 emulates fp16 PyTorch: q*cos, rotate_half(q)*sin and their sum are each rounded.

struct Params {
  T: u32,
  pos0: u32,
  heads: u32,
  kv_heads: u32,
  head_dim: u32,
  round16: u32,
  q_slots: u32,
  k_slots: u32,
  v_slots: u32,
  per_t: u32,
  kv_cols: u32,
  _pad: u32,
}

@group(0) @binding(0) var<uniform> P: Params;
@group(0) @binding(1) var<storage, read_write> q: array<f32>;
@group(0) @binding(2) var<storage, read> k: array<f32>;
@group(0) @binding(3) var<storage, read> v: array<f32>;
@group(0) @binding(4) var<storage, read> cos_t: array<f32>;
@group(0) @binding(5) var<storage, read> sin_t: array<f32>;
@group(0) @binding(6) var<storage, read_write> kc: array<u32>;   // fp16 pairs [maxCtx, kv_cols/2]
@group(0) @binding(7) var<storage, read_write> vc: array<u32>;

fn r16(v: f32) -> f32 {
  return unpack2x16float(pack2x16float(vec2<f32>(v, 0.0))).x;
}

fn rr(v: f32) -> f32 {
  if (P.round16 != 0u) { return r16(v); }
  return v;
}

// x0,x1 = elements i, i+1 (first half); x2,x3 = elements i+half, i+half+1.
fn rot4(x0: f32, x1: f32, x2: f32, x3: f32, c0: f32, c1: f32, s0: f32, s1: f32) -> vec4<f32> {
  let o0 = rr(rr(x0 * c0) + rr(-x2 * s0));
  let o1 = rr(rr(x1 * c1) + rr(-x3 * s1));
  let o2 = rr(rr(x2 * c0) + rr(x0 * s0));
  let o3 = rr(rr(x3 * c1) + rr(x1 * s1));
  return vec4<f32>(o0, o1, o2, o3);
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let g = gid.x;
  if (g >= P.T * P.per_t) { return; }
  let t = g / P.per_t;
  let slot = g % P.per_t;
  let p = P.pos0 + t;
  let hd = P.head_dim;
  let half = hd / 2u;
  let quads = hd / 4u;
  if (slot < P.q_slots) {
    let h = slot / quads;
    let j = slot % quads;
    let i = 2u * j;
    let base = t * P.heads * hd + h * hd;
    let c0 = cos_t[p * hd + i];
    let c1 = cos_t[p * hd + i + 1u];
    let s0 = sin_t[p * hd + i];
    let s1 = sin_t[p * hd + i + 1u];
    let o = rot4(q[base + i], q[base + i + 1u], q[base + i + half], q[base + i + half + 1u], c0, c1, s0, s1);
    q[base + i] = o.x;
    q[base + i + 1u] = o.y;
    q[base + i + half] = o.z;
    q[base + i + half + 1u] = o.w;
  } else if (slot < P.q_slots + P.k_slots) {
    let ks = slot - P.q_slots;
    let h = ks / quads;
    let j = ks % quads;
    let i = 2u * j;
    let base = t * P.kv_cols + h * hd;
    let c0 = cos_t[p * hd + i];
    let c1 = cos_t[p * hd + i + 1u];
    let s0 = sin_t[p * hd + i];
    let s1 = sin_t[p * hd + i + 1u];
    let o = rot4(k[base + i], k[base + i + 1u], k[base + i + half], k[base + i + half + 1u], c0, c1, s0, s1);
    let row = p * P.kv_cols + h * hd;
    kc[(row + i) >> 1u] = pack2x16float(vec2<f32>(o.x, o.y));
    kc[(row + i + half) >> 1u] = pack2x16float(vec2<f32>(o.z, o.w));
  } else {
    let vs = slot - P.q_slots - P.k_slots;
    let col = vs * 2u;
    vc[(p * P.kv_cols + col) >> 1u] = pack2x16float(vec2<f32>(v[t * P.kv_cols + col], v[t * P.kv_cols + col + 1u]));
  }
}
