import { defineConfig } from "vite";
import { resolve } from "node:path";
import { writeupPlugin } from "./plugins/writeup";

// dianome.dev (Phase 7 Part B, Phase 6 dashboard): one Vite site, five pages, plain DOM. Deployed as the `dianome.dev` Pages project by Theo
// (build output: dist/). Local:
//   pnpm --filter site dev  →  http://localhost:5178/summarize/?api=http://127.0.0.1:8787&cdn=http://localhost:8788
// with `node scripts/serve-store.mjs` (8788), `wrangler dev` in packages/worker (8787) and `dianome-server serve` (8765).
// `?split=ws://…&token=…` bypasses the Worker's session endpoint (static bearer), as in apps/split-demo.
export default defineConfig({
  server: { port: 5178 },
  preview: { port: 5178 },
  plugins: [writeupPlugin(resolve(__dirname, "../../docs/writeup.md"))],
  build: {
    target: "es2022",
    rollupOptions: {
      input: {
        main: resolve(__dirname, "index.html"),
        summarize: resolve(__dirname, "summarize/index.html"),
        stats: resolve(__dirname, "stats/index.html"),
        writeup: resolve(__dirname, "writeup/index.html"),
        dashboard: resolve(__dirname, "dashboard/index.html"),
      },
    },
  },
});
