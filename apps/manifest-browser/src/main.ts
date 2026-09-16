import {
  dedupStats, fmtBytes, isModel, variantNames,
  type FileEntry, type FilesManifest, type Manifest, type ModelManifest,
} from "./manifest";

const DEFAULT_ID = "qwen2.5-0.5b-instruct";
const app = document.getElementById("app")!;
const status = document.getElementById("status")!;
const urlInput = document.getElementById("url") as HTMLInputElement;
const form = document.getElementById("form") as HTMLFormElement;

const params = new URLSearchParams(location.search);
urlInput.value = params.get("manifest") ?? `./store/manifests/${params.get("id") ?? DEFAULT_ID}/latest.json`;

form.addEventListener("submit", (ev) => { ev.preventDefault(); void load(urlInput.value); });
void load(urlInput.value);

async function load(url: string): Promise<void> {
  status.textContent = `loading ${url} …`;
  app.replaceChildren();
  try {
    const res = await fetch(url, { cache: "no-store" });
    if (!res.ok) throw new Error(`${res.status} ${res.statusText}`);
    const m = (await res.json()) as Manifest;
    if (m.schema !== 1) throw new Error(`unsupported schema ${String(m.schema)}`);
    status.textContent = `loaded ${url} (${new Date().toLocaleTimeString()})`;
    render(m);
  } catch (e) {
    status.textContent = `failed to load ${url}: ${(e as Error).message}`;
  }
}

// ---------------------------------------------------------------- DOM helpers

function el<K extends keyof HTMLElementTagNameMap>(tag: K, text?: string | number, cls?: string): HTMLElementTagNameMap[K] {
  const e = document.createElement(tag);
  if (text !== undefined) e.textContent = String(text);
  if (cls) e.className = cls;
  return e;
}

function table(headers: { label: string; num?: boolean }[], rows: (string | number)[][]): HTMLTableElement {
  const t = el("table");
  const tr = el("tr");
  for (const h of headers) tr.append(el("th", h.label, h.num ? "num" : undefined));
  const thead = el("thead");
  thead.append(tr);
  t.append(thead);
  const body = t.appendChild(el("tbody"));
  for (const r of rows) {
    const row = el("tr");
    r.forEach((cell, i) => row.append(el("td", typeof cell === "number" ? cell.toLocaleString() : cell, headers[i]?.num ? "num" : undefined)));
    body.append(row);
  }
  return t;
}

function kv(pairs: [string, string | number][]): HTMLTableElement {
  return table([{ label: "field" }, { label: "value" }], pairs.map(([k, v]) => [k, typeof v === "number" ? v : String(v)]));
}

// ---------------------------------------------------------------- rendering

function render(m: Manifest): void {
  const chunkCount = Object.keys(m.chunks).length;
  const chunkBytes = Object.values(m.chunks).reduce((s, c) => s + c.bytes, 0);
  const head = el("section");
  head.append(el("h2", isModel(m) ? "Model" : "Artifact"));
  head.append(kv([
    ["id", m.id],
    ["source repo", m.source.repo],
    ["source revision", m.source.revision || "(none)"],
    ...(isModel(m) ? ([["family", m.family]] as [string, string][]) : ([["runtime", m.runtime]] as [string, string][])),
    ["chunk size", `${m.chunk_size.toLocaleString()} bytes`],
    ["unique chunks", `${chunkCount.toLocaleString()} (${fmtBytes(chunkBytes)})`],
  ]));
  app.append(head);
  if (isModel(m)) renderModel(m);
  else renderFiles(m);
}

function renderFiles(m: FilesManifest): void {
  const s = el("section");
  s.append(el("h2", `Files (${m.files.length})`));
  s.append(filesTable(m.files));
  s.append(el("h3", "Bytes per file, in manifest order"));
  s.append(bars(m.files.map((f) => [f.name, f.bytes])));
  app.append(s);
}

function filesTable(files: FileEntry[]): HTMLTableElement {
  return table(
    [{ label: "file" }, { label: "bytes", num: true }, { label: "size", num: true }, { label: "chunks", num: true }],
    files.map((f) => [f.name, f.bytes, fmtBytes(f.bytes), f.chunks.length]),
  );
}

function renderModel(m: ModelManifest): void {
  const cfg = el("section");
  cfg.append(el("h2", "Config summary"));
  cfg.append(kv(Object.entries(m.config).map(([k, v]) => [k, String(v)])));
  app.append(cfg);

  for (const v of variantNames(m)) {
    const variant = m.variants[v]!;
    const listed = variant.groups.flatMap((g) => g.chunks);
    const s = el("section");
    s.append(el("h2", `Variant ${v}`));
    s.append(el("p", `total ${variant.bytes.toLocaleString()} bytes (${fmtBytes(variant.bytes)}), ${listed.length} chunks listed, ${new Set(listed).size} unique, ${variant.groups.length} groups`));
    s.append(table(
      [{ label: "group" }, { label: "bytes", num: true }, { label: "size", num: true }, { label: "chunks", num: true }, { label: "entries", num: true }, { label: "note" }],
      variant.groups.map((g) => [g.name, g.bytes, fmtBytes(g.bytes), g.chunks.length, g.entries.length, g.tied ? "tied → embed" : ""]),
    ));
    s.append(el("h3", "Bytes per group, in download order"));
    s.append(bars(variant.groups.map((g) => [g.name, g.bytes])));
    app.append(s);
  }

  const d = dedupStats(m);
  const s = el("section");
  s.append(el("h2", "Dedup across variants"));
  s.append(kv([
    ["chunks listed across variants", `${d.listedChunks} (${fmtBytes(d.listedBytes)})`],
    ["unique chunks", `${d.uniqueChunks} (${fmtBytes(d.uniqueBytes)})`],
    ["chunks shared between variants", `${d.sharedChunks} (${fmtBytes(d.sharedBytes)})`],
    ["bytes saved", `${d.bytesSaved.toLocaleString()} (${fmtBytes(d.bytesSaved)})`],
  ]));
  s.append(table(
    [{ label: "pair" }, { label: "shared chunks", num: true }, { label: "shared bytes", num: true }],
    d.pairs.map((p) => [`${p.a} ∩ ${p.b}`, p.chunks, p.bytes]),
  ));
  s.append(el("h3", "Groups with identical chunk lists across variants"));
  s.append(d.sharedGroups.length
    ? table([{ label: "group" }, { label: "variants" }, { label: "chunks", num: true }, { label: "bytes", num: true }],
        d.sharedGroups.map((g) => [g.group, g.variants.join(" = "), g.chunks, g.bytes]))
    : el("p", "none"));
  app.append(s);

  const t = el("section");
  t.append(el("h2", `Tokenizer files (${m.tokenizer.files.length})`));
  t.append(filesTable(m.tokenizer.files));
  app.append(t);
}

function bars(items: [string, number][]): HTMLDivElement {
  const max = Math.max(1, ...items.map(([, b]) => b));
  const box = el("div", undefined, "bars");
  for (const [label, bytes] of items) {
    const row = el("div");
    row.append(el("span", label, "label"));
    const bar = el("span", undefined, "bar");
    bar.style.width = `${Math.max(1, (bytes / max) * 480)}px`;
    bar.title = `${bytes.toLocaleString()} bytes`;
    row.append(bar, el("span", fmtBytes(bytes)));
    box.append(row);
  }
  return box;
}
