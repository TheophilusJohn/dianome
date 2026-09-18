// Sequential matvec for T == 1: one thread per output row accumulating over `in` in the same order as the
// tiled kernel (one product at a time, k ascending), so a decode step reproduces the prefill path bit for bit
// under fp16 emulation. Less parallel than matvec.wgsl (out_dim threads instead of 16 per row); measured
// against it in the bench.

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let o = gid.x;
  if (o >= P.out_dim) { return; }
  var acc = 0.0;
  for (var k = 0u; k < P.in_dim; k += 2u) {
    let wv = load_w2(o, k);
    acc += x[k] * wv.x;
    acc += x[k + 1u] * wv.y;
  }
  y[o] = finish(o, acc);
}
