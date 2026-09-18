// Renders docs/writeup.md (Phase 7 Part C) to HTML at build time, exposed as the virtual module "virtual:writeup".
// Links into the repo (`docs/phase-4-notes.md#cost-harness`, `../server/bench/results/…`) become GitHub blob URLs at
// `main`, so every number's citation link resolves from the deployed page. The client ships no markdown library.
import { existsSync, readFileSync, statSync } from "node:fs";
import { Marked } from "marked";
import type { Plugin } from "vite";

const ID = "virtual:writeup";
const RID = "\0" + ID;
const REPO_BLOB = "https://github.com/TheophilusJohn/dianome/blob/main/";
const STUB = "# Write-up\n\n_`docs/writeup.md` is not written yet (Phase 7, Part C)._\n";

export function repoLink(href: string): string {
  if (/^[a-z]+:/i.test(href) || href.startsWith("#") || href.startsWith("/")) return href;
  const rel = href.replace(/^(\.\.\/)+/, "").replace(/^\.\//, "");
  return REPO_BLOB + rel;
}

export function renderWriteup(md: string): string {
  const m = new Marked({ gfm: true });
  m.use({ renderer: { link({ href, title, text }) { return `<a href="${repoLink(href)}"${title ? ` title="${title}"` : ""}>${text}</a>`; } } });
  return m.parse(md, { async: false });
}

export function writeupPlugin(mdPath: string): Plugin {
  return {
    name: "dianome-writeup",
    resolveId(source) { return source === ID ? RID : null; },
    load(id) {
      if (id !== RID) return null;
      const exists = existsSync(mdPath);
      const md = exists ? readFileSync(mdPath, "utf8") : STUB;
      const words = md.replace(/```[\s\S]*?```/g, "").split(/\s+/).filter(Boolean).length;
      const modified = exists ? statSync(mdPath).mtime.toISOString() : null;
      return `export const html = ${JSON.stringify(renderWriteup(md))};\nexport const exists = ${exists};\nexport const words = ${words};\nexport const modified = ${JSON.stringify(modified)};\n`;
    },
    configureServer(server) {
      server.watcher.add(mdPath);
      server.watcher.on("change", (p) => { if (p === mdPath) { const mod = server.moduleGraph.getModuleById(RID); if (mod) server.moduleGraph.invalidateModule(mod); server.ws.send({ type: "full-reload" }); } });
    },
  };
}
