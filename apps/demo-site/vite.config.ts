import { defineConfig } from "vite";
import { resolve } from "node:path";

// Three pages: the model-load demo, and one page per adapter. Served locally with `?api=&cdn=` overrides pointing
// at `node scripts/serve-store.mjs` (repo root). Deployed twice to Pages by Theo (see README.md).
export default defineConfig({
  // No COEP here: a COEP page can only embed cross-origin frames that carry COEP themselves (see scripts/publish-frame.sh).
  server: { port: 5175 },
  preview: { port: 5175 },
  build: {
    target: "es2022",
    rollupOptions: { input: { main: resolve(__dirname, "index.html"), transformersjs: resolve(__dirname, "transformersjs.html"), webllm: resolve(__dirname, "webllm.html") } },
  },
  optimizeDeps: { exclude: ["@huggingface/transformers", "@mlc-ai/web-llm"] },
});
