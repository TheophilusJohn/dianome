import { defineConfig } from "vite";

// Served at dianome.dev/stats later, hence base "/stats/". The API base comes from VITE_API_BASE
// (default http://localhost:8787, i.e. `pnpm --filter worker dev`) or a ?api= query parameter.
export default defineConfig({ base: "/stats/", server: { port: 5175 } });
