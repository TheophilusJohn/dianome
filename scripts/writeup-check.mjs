#!/usr/bin/env node
// Checks docs/writeup.md (Phase 7 Part C). Exit 1 on any violation; the site build runs this first.
//
//   1. Every number outside code spans, fenced blocks, headings and HTML comments must sit in a block (paragraph,
//      list item or table row) that carries a citation: a markdown link to a repo file (`../<path>`, must exist; a
//      `#anchor` must match a heading of that markdown file, GitHub slug rules) or to an http(s) URL.
//      Numbers that are part of identifiers (0.5B, fp16, q4, L4, 2026-09-17, SHA-256, Qwen2.5, v0.2.0) are not counted,
//      nor are numbers after a structural word (Phase, Gate, Spike, section, step, block, boundary, schema, N, L, T).
//   2. `<!-- pending: <glob> -->` placeholders are allowed only while no file matches the glob (repo-relative); once
//      the GPU day has produced the results the placeholder must be replaced by the table.
//   3. Fewer than 4,000 words outside tables, code blocks and comments.
//
//   node scripts/writeup-check.mjs [docs/writeup.md]
import { existsSync, readFileSync, readdirSync, statSync } from "node:fs";
import { dirname, join, resolve, sep } from "node:path";

const repo = resolve(dirname(new URL(import.meta.url).pathname), "..");
const file = resolve(process.argv[2] ?? join(repo, "docs/writeup.md"));
const src = readFileSync(file, "utf8");
const WORD_LIMIT = 4000;
const errors = [];
const err = (line, msg) => { const e = `${file.replace(repo + sep, "")}:${line}: ${msg}`; if (!errors.includes(e)) errors.push(e); };

export function slug(heading) {
  // GitHub's heading anchors: lowercase, drop punctuation except hyphen/underscore, spaces to hyphens.
  return heading.trim().toLowerCase().replace(/[^\p{L}\p{N}\s_-]/gu, "").replace(/\s/g, "-");
}
function headingsOf(path) {
  const seen = new Map();
  const out = new Set();
  for (const line of readFileSync(path, "utf8").split("\n")) {
    const m = /^#{1,6}\s+(.*?)\s*#*\s*$/.exec(line);
    if (!m) continue;
    let s = slug(m[1].replace(/`/g, ""));
    const n = seen.get(s) ?? 0;
    seen.set(s, n + 1);
    if (n > 0) s = `${s}-${n}`;
    out.add(s);
  }
  return out;
}
const headingCache = new Map();
function hasAnchor(path, anchor) {
  if (!headingCache.has(path)) headingCache.set(path, headingsOf(path));
  return headingCache.get(path).has(anchor);
}

// -- placeholders --------------------------------------------------------------------------------------------------
function globMatches(pattern) {
  const parts = pattern.split("/");
  const dir = join(repo, ...parts.slice(0, -1));
  const re = new RegExp("^" + parts[parts.length - 1].replace(/[.+^${}()|[\]\\]/g, "\\$&").replace(/\*/g, ".*") + "$");
  if (!existsSync(dir) || !statSync(dir).isDirectory()) return [];
  return readdirSync(dir).filter((f) => re.test(f));
}
const lines = src.split("\n");
let pendingCount = 0;
lines.forEach((line, i) => {
  const m = /<!--\s*pending:\s*(\S+)\s*-->/.exec(line);
  if (!m) return;
  pendingCount++;
  const hits = globMatches(m[1]);
  if (hits.length) err(i + 1, `placeholder for ${m[1]} but results exist (${hits.join(", ")}): fill the table and remove the marker`);
});

// -- blocks -----------------------------------------------------------------------------------------------------------
// Strip fenced code and HTML comments (keep line numbers), then split into blocks.
let inFence = false, inComment = false;
const clean = lines.map((line) => {
  if (inFence) { if (/^\s*```/.test(line)) inFence = false; return ""; }
  if (/^\s*```/.test(line)) { inFence = true; return ""; }
  let s = line;
  if (inComment) { const e = s.indexOf("-->"); if (e < 0) return ""; s = s.slice(e + 3); inComment = false; }
  s = s.replace(/<!--.*?-->/g, "");
  const st = s.indexOf("<!--"); if (st >= 0) { s = s.slice(0, st); inComment = true; }
  return s;
});
const blocks = []; // { start, text, table }
let cur = null;
clean.forEach((line, i) => {
  const isTable = /^\s*\|/.test(line);
  const isHeading = /^#{1,6}\s/.test(line);
  const isList = /^\s*([-*+]|\d+\.)\s/.test(line);
  if (!line.trim() || isHeading) { if (cur) { blocks.push(cur); cur = null; } return; }
  if (isTable) { if (cur) { blocks.push(cur); cur = null; } if (!/^\s*\|\s*-{2,}/.test(line)) blocks.push({ start: i + 1, text: line, table: true }); return; }
  if (isList && cur) { blocks.push(cur); cur = null; }
  if (!cur) cur = { start: i + 1, text: line, table: false }; else cur.text += "\n" + line;
});
if (cur) blocks.push(cur);

// -- numbers and citations ---------------------------------------------------------------------------------------------
const LINK = /\[[^\]]*\]\(([^)\s]+)\)/g;
// "Phase 4", "sections 5 and 6", "N = 24", "boundaries 0 to 24": a structural word followed only by numbers and joiners.
const STRUCTURAL = /(?:^|[\s(])(?:phase|gate|gates|spike|section|sections|part|step|steps|block|blocks|boundary|boundaries|schema|run|N|L|T|top)(?:\s*(?:=|∈|to|,|and|–|\{|\d+))*\s*$/i;
const NUMBER = /(?<![\p{L}\p{N}._\-/])\d[\d,]*(?:\.\d+)?(?:e[-+]?\d+)?%?(?![\p{L}\p{N}_\-/]|\.\d)/gu;
let numbers = 0, citations = 0;
const tableHeaderLines = new Set();
lines.forEach((l, i) => { if (/^\s*\|\s*-{2,}/.test(l) && i > 0) tableHeaderLines.add(i); }); // the line above a separator is a header row

const citationLinks = (text) => [...text.replace(/`[^`]*`/g, "").matchAll(LINK)].map((m) => m[1]);
/** A table's rows inherit the citation of the block immediately before or after the table (the "Source:" line). */
function inheritedLinks(idx) {
  const b = blocks[idx];
  if (!b.table) return [];
  let a = idx, z = idx;
  while (a > 0 && blocks[a - 1].table && blocks[a - 1].start === blocks[a].start - 1) a--;
  while (z + 1 < blocks.length && blocks[z + 1].table && blocks[z + 1].start === blocks[z].start + 1) z++;
  const near = [];
  if (a > 0 && !blocks[a - 1].table && blocks[a].start - lastLine(blocks[a - 1]) <= 2) near.push(blocks[a - 1]);
  if (z + 1 < blocks.length && !blocks[z + 1].table && blocks[z + 1].start - blocks[z].start <= 2) near.push(blocks[z + 1]);
  return near.flatMap((n) => citationLinks(n.text));
}
const lastLine = (b) => b.start + b.text.split("\n").length - 1;

for (const [idx, b] of blocks.entries()) {
  if (b.table && tableHeaderLines.has(b.start)) continue;
  const noCode = b.text.replace(/`[^`]*`/g, "");
  const noLinks = noCode.replace(LINK, (m) => m.replace(/\([^)]*\)$/, "()"));   // link targets never count as numbers
  const found = [];
  for (const m of noLinks.matchAll(NUMBER)) {
    const before = noLinks.slice(Math.max(0, m.index - 40), m.index);
    if (STRUCTURAL.test(before)) continue;
    if (/^\d{4}$/.test(m[0]) && /\b(19|20)\d\d$/.test(m[0]) && /(?:in|since|of|,)\s*$/.test(before) === false && /^\s*\)/.test(noLinks.slice(m.index + 4))) continue;
    found.push(m[0]);
  }
  if (!found.length) continue;
  numbers += found.length;
  let cited = false;
  for (const target of [...citationLinks(noCode), ...inheritedLinks(idx)]) {
    if (/^https?:\/\//.test(target)) { cited = true; citations++; continue; }
    const [pathPart, anchor] = target.split("#");
    const abs = resolve(dirname(file), pathPart);
    if (!abs.startsWith(repo + sep)) { err(b.start, `link outside the repo: ${target}`); continue; }
    if (!existsSync(abs)) { err(b.start, `link target does not exist: ${target}`); continue; }
    if (anchor !== undefined) {
      if (!abs.endsWith(".md")) { err(b.start, `anchor on a non-markdown file: ${target}`); continue; }
      if (!hasAnchor(abs, anchor)) { err(b.start, `no heading "${anchor}" in ${pathPart}`); continue; }
    }
    cited = true; citations++;
  }
  if (!cited) err(b.start, `numbers without a citation link: ${found.slice(0, 6).join(", ")}${found.length > 6 ? ", …" : ""}`);
}

// -- links that carry no numbers still have to resolve -----------------------------------------------------------------
for (const b of blocks) {
  for (const m of b.text.matchAll(LINK)) {
    const target = m[1];
    if (/^https?:\/\//.test(target)) continue;
    const [pathPart, anchor] = target.split("#");
    const abs = resolve(dirname(file), pathPart);
    if (!existsSync(abs)) err(b.start, `link target does not exist: ${target}`);
    else if (anchor !== undefined && abs.endsWith(".md") && !hasAnchor(abs, anchor)) err(b.start, `no heading "${anchor}" in ${pathPart}`);
  }
}

// -- word count outside tables -----------------------------------------------------------------------------------------
const words = blocks.filter((b) => !b.table).map((b) => b.text.replace(LINK, (m) => m.replace(/\([^)]*\)$/, ""))).join(" ").split(/\s+/).filter((w) => /[\p{L}\p{N}]/u.test(w)).length;
if (words > WORD_LIMIT) err(1, `${words} words outside tables exceeds the ${WORD_LIMIT}-word limit`);

console.log(`writeup-check: ${blocks.length} blocks, ${numbers} numbers, ${citations} citations verified, ${pendingCount} pending placeholders, ${words} words outside tables`);
if (errors.length) { for (const e of errors) console.error("  " + e); console.error(`writeup-check: ${errors.length} problem${errors.length === 1 ? "" : "s"}`); process.exit(1); }
console.log("writeup-check: ok");
