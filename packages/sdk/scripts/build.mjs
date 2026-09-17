// Bundles the three entry points with esbuild (ESM, code-split so adapters share the core), no minification of
// identifiers beyond whitespace so stack traces stay readable. Declarations come from `tsc -p tsconfig.build.json`.
import { build } from "esbuild";
import { rmSync } from "node:fs";

rmSync("dist", { recursive: true, force: true });
const result = await build({
  entryPoints: { index: "src/index.ts", transformersjs: "src/adapters/transformersjs.ts", webllm: "src/adapters/webllm.ts" },
  outdir: "dist",
  bundle: true,
  format: "esm",
  platform: "browser",
  target: ["es2022"],
  splitting: true,
  sourcemap: true,
  minifySyntax: true,
  minifyWhitespace: true,
  treeShaking: true,
  legalComments: "none",
  metafile: true,
  logLevel: "info",
});
const outputs = Object.entries(result.metafile.outputs).filter(([f]) => f.endsWith(".js"));
for (const [f, o] of outputs) console.log(`${f}\t${o.bytes} bytes`);
