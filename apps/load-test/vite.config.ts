import { defineConfig } from "vite";

// Talks straight to the Worker (http://localhost:8787 under `pnpm --filter worker dev`) and to
// cdn.dianome.dev; both answer with Access-Control-Allow-Origin: *, so no proxy is needed.
export default defineConfig({ server: { port: 5174 } });
