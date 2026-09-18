// ESM bundle of the runtime (one entry, `.wgsl` files inlined as strings). Declarations come from tsc.
import { build } from "esbuild";
import { rmSync } from "node:fs";

rmSync("dist", { recursive: true, force: true });
const result = await build({
  entryPoints: { index: "src/index.ts" },
  outdir: "dist",
  bundle: true,
  format: "esm",
  platform: "browser",
  target: ["es2022"],
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
