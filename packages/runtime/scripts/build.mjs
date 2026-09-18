// ESM bundles of the runtime: the main entry (kernels inlined as strings) plus the tokenizer, planner and protocol
// subpath entries, code-split so shared modules live in one chunk. Declarations come from tsc.
import { build } from "esbuild";
import { rmSync } from "node:fs";

rmSync("dist", { recursive: true, force: true });
const result = await build({
  entryPoints: { index: "src/index.ts", tokenizer: "src/entries/tokenizer.ts", planner: "src/entries/planner.ts", protocol: "src/entries/protocol.ts", session: "src/entries/session.ts" },
  outdir: "dist",
  bundle: true,
  format: "esm",
  platform: "browser",
  target: ["es2022"],
  splitting: true,
  external: ["dianome"],
  loader: { ".wgsl": "text" },
  sourcemap: true,
  minifySyntax: true,
  minifyWhitespace: true,
  treeShaking: true,
  legalComments: "none",
  metafile: true,
  logLevel: "info",
});
for (const [f, o] of Object.entries(result.metafile.outputs).filter(([f]) => f.endsWith(".js"))) console.log(`${f}\t${o.bytes} bytes`);
