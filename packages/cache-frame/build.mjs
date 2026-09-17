// Builds the frame to plain static files in dist/: frame.js (bundled, ESM), index.html, optin.html.
// scripts/publish-frame.sh uploads dist/ to R2 under frame/v1/ (Theo runs it).
import { build } from "esbuild";
import { copyFileSync, mkdirSync, rmSync, statSync } from "node:fs";

rmSync("dist", { recursive: true, force: true });
mkdirSync("dist");
await build({
  entryPoints: { frame: "src/frame.ts" },
  outdir: "dist",
  bundle: true,
  format: "esm",
  platform: "browser",
  target: ["es2022"],
  minifySyntax: true,
  minifyWhitespace: true,
  sourcemap: false,
  legalComments: "none",
});
for (const f of ["index.html", "optin.html"]) copyFileSync(`src/${f}`, `dist/${f}`);
for (const f of ["frame.js", "index.html", "optin.html"]) console.log(`dist/${f}\t${statSync(`dist/${f}`).size} bytes`);
