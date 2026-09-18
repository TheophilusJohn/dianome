// Tiled matmul for T > 1 (prefill): 32 x 64 output tile per workgroup of 16x16 threads, K step 32, each thread
// 2 rows x 4 cols. in_dim must be a multiple of 32 and out_dim a multiple of 64 (checked at op creation);
// T is arbitrary (rows past T are neither read nor written). Weights arrive dequantised through load_w2.

const TM: u32 = 32u;
const TN: u32 = 64u;
const TK: u32 = 32u;

var<workgroup> xs: array<f32, 1024>;   // [TM][TK]
var<workgroup> ws: array<f32, 2048>;   // [TN][TK]

@compute @workgroup_size(16, 16)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
  let row0 = wg.y * TM;
  let col0 = wg.x * TN;
  let tid = lid.y * 16u + lid.x;
  var acc: array<f32, 8>;
  for (var j = 0u; j < 8u; j++) { acc[j] = 0.0; }

  for (var k0 = 0u; k0 < P.in_dim; k0 += TK) {
    for (var i = tid; i < TM * TK; i += 256u) {
      let r = i / TK;
      let c = i % TK;
      let gr = row0 + r;
      var v = 0.0;
      if (gr < P.T) { v = x[gr * P.in_dim + k0 + c]; }
      xs[i] = v;
    }
    for (var i = tid; i < TN * (TK / 2u); i += 256u) {
      let r = i / (TK / 2u);
      let c2 = i % (TK / 2u);
      let v = load_w2(col0 + r, k0 + c2 * 2u);
      ws[r * TK + c2 * 2u] = v.x;
      ws[r * TK + c2 * 2u + 1u] = v.y;
    }
    workgroupBarrier();
    let xr0 = (lid.y * 2u) * TK;
    let xr1 = xr0 + TK;
    let wc = (lid.x * 4u) * TK;
    for (var k = 0u; k < TK; k++) {
      let x0 = xs[xr0 + k];
      let x1 = xs[xr1 + k];
      let w0 = ws[wc + k];
      let w1 = ws[wc + TK + k];
      let w2 = ws[wc + 2u * TK + k];
      let w3 = ws[wc + 3u * TK + k];
      acc[0] += x0 * w0; acc[1] += x0 * w1; acc[2] += x0 * w2; acc[3] += x0 * w3;
      acc[4] += x1 * w0; acc[5] += x1 * w1; acc[6] += x1 * w2; acc[7] += x1 * w3;
    }
    workgroupBarrier();
  }

  for (var i = 0u; i < 2u; i++) {
    let r = row0 + lid.y * 2u + i;
    if (r < P.T) {
      for (var j = 0u; j < 4u; j++) {
        let o = col0 + lid.x * 4u + j;
        y[r * P.out_dim + o] = finish(o, acc[i * 4u + j]);
      }
    }
  }
}
