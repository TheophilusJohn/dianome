import { defineConfig } from "vite";

// `./store/...` relative to the dev page is proxied to the static store server
// (pnpm --filter manifest-browser serve-store, port 8788), so the default
// manifest URL `./store/manifests/<id>/latest.json` works out of the box.
export default defineConfig({
  server: {
    port: 5173,
    proxy: {
      "/store": {
        target: "http://localhost:8788",
        rewrite: (p) => p.replace(/^\/store/, ""),
      },
    },
  },
});
