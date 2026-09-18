// Matrix-vector path for T == 1 (decode): a workgroup of 64 threads computes 4 output rows, 16 lanes per row
// striding over the packed input pairs, then a shared-memory reduction. Adjacent lanes read adjacent words.
// in_dim must be even (all Qwen2 dims are).

const ROWS: u32 = 4u;
const LANES: u32 = 16u;

var<workgroup> red: array<f32, 64>;

@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
  let lane = lid.x & (LANES - 1u);
  let sub = lid.x / LANES;
  let o = wg.x * ROWS + sub;
  let pairs = P.in_dim >> 1u;
  var acc = 0.0;
  if (o < P.out_dim) {
    for (var p = lane; p < pairs; p += LANES) {
      let k = p * 2u;
      let wv = load_w2(o, k);
      acc += x[k] * wv.x + x[k + 1u] * wv.y;
    }
  }
  red[lid.x] = acc;
  workgroupBarrier();
  for (var s = LANES / 2u; s > 0u; s >>= 1u) {
    if (lane < s) { red[lid.x] += red[lid.x + s]; }
    workgroupBarrier();
  }
  if (lane == 0u && o < P.out_dim) {
    y[o] = finish(o, red[lid.x]);
  }
}
