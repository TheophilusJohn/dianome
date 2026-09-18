// Causal attention, one workgroup of HD threads per (query row, head): scores over the cached keys of this
// block, two-pass softmax in workgroup memory, then a weighted sum of the cached values. GQA maps head h to
// kv head h / (heads / kv_heads). Prefill and decode use the same kernel (T rows starting at pos0; T = 1 for
// decode). MAX_KEYS and HD are substituted at pipeline creation.
// flags: bit0 = round output to fp16, bit1 = round probabilities to fp16 (eager PyTorch casts softmax to fp16
// before multiplying V), bit2 = round scores to fp16.

struct Params {
  T: u32,
  pos0: u32,
  heads: u32,
  kv_heads: u32,
  head_dim: u32,
  kv_cols: u32,
  flags: u32,
  _pad: u32,
}

const MAX_KEYS: u32 = __MAX_KEYS__u;
const HD: u32 = __HD__u;

@group(0) @binding(0) var<uniform> P: Params;
@group(0) @binding(1) var<storage, read> q: array<f32>;
@group(0) @binding(2) var<storage, read> kc: array<u32>;   // fp16 pairs [maxCtx, kv_cols/2]
@group(0) @binding(3) var<storage, read> vc: array<u32>;
@group(0) @binding(4) var<storage, read_write> out: array<f32>;

var<workgroup> scores: array<f32, MAX_KEYS>;
var<workgroup> red: array<f32, HD>;
var<workgroup> qs: array<f32, HD>;

fn r16(v: f32) -> f32 {
  return unpack2x16float(pack2x16float(vec2<f32>(v, 0.0))).x;
}

@compute @workgroup_size(HD)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
  let t = wg.x;
  let h = wg.y;
  let kh = h / (P.heads / P.kv_heads);
  let nkeys = P.pos0 + t + 1u;
  let d = lid.x;
  let scale = 1.0 / sqrt(f32(HD));

  qs[d] = q[t * P.heads * HD + h * HD + d];
  workgroupBarrier();

  var m = -3.0e38;
  for (var j = d; j < nkeys; j += HD) {
    var s = 0.0;
    let kb = (j * P.kv_cols + kh * HD) >> 1u;
    for (var i = 0u; i < HD; i += 2u) {
      let p = unpack2x16float(kc[kb + (i >> 1u)]);
      s += qs[i] * p.x + qs[i + 1u] * p.y;
    }
    s = s * scale;
    if ((P.flags & 4u) != 0u) { s = r16(s); }
    scores[j] = s;
    m = max(m, s);
  }
  red[d] = m;
  workgroupBarrier();
  for (var st = HD / 2u; st > 0u; st >>= 1u) {
    if (d < st) { red[d] = max(red[d], red[d + st]); }
    workgroupBarrier();
  }
  m = red[0];
  workgroupBarrier();

  var sum = 0.0;
  for (var j = d; j < nkeys; j += HD) {
    let e = exp(scores[j] - m);
    scores[j] = e;
    sum += e;
  }
  red[d] = sum;
  workgroupBarrier();
  for (var st = HD / 2u; st > 0u; st >>= 1u) {
    if (d < st) { red[d] = red[d] + red[d + st]; }
    workgroupBarrier();
  }
  let total = red[0];

  var acc = 0.0;
  for (var j = 0u; j < nkeys; j++) {
    var p = scores[j] / total;
    if ((P.flags & 2u) != 0u) { p = r16(p); }
    let vw = unpack2x16float(vc[(j * P.kv_cols + kh * HD + d) >> 1u]);
    let vv = select(vw.x, vw.y, (d & 1u) == 1u);
    acc += p * vv;
  }
  if ((P.flags & 1u) != 0u) { acc = r16(acc); }
  out[t * P.heads * HD + h * HD + d] = acc;
}
