// Shared by the four pages: endpoint resolution (?api=&cdn= overrides for local servers), the nav, DOM helpers.
import "./site.css";

export const params = new URLSearchParams(location.search);
export const API = (params.get("api") ?? (import.meta.env.VITE_API_BASE as string | undefined) ?? "https://api.dianome.dev").replace(/\/+$/, "");
export const CDN = (params.get("cdn") ?? (import.meta.env.VITE_CDN_BASE as string | undefined) ?? "https://cdn.dianome.dev").replace(/\/+$/, "");

/** Keeps ?api=&cdn= on internal links so a local session stays local across pages. */
export function localQuery(): string {
  const q = new URLSearchParams();
  for (const k of ["api", "cdn", "split", "token"]) { const v = params.get(k); if (v) q.set(k, v); }
  const s = q.toString();
  return s ? `?${s}` : "";
}

export function nav(current: string): void {
  const pages: [string, string][] = [["/", "home"], ["/summarize/", "summarise"], ["/stats/", "stats"], ["/writeup/", "write-up"]];
  const n = document.querySelector("nav");
  if (!n) return;
  const brand = el("span", "dianome", "brand");
  n.replaceChildren(brand, ...pages.map(([href, label]) => {
    const a = el("a", label); a.href = href + localQuery();
    if (href === current) a.setAttribute("aria-current", "page");
    return a;
  }));
}

export const $ = <T extends HTMLElement>(id: string): T => document.getElementById(id) as T;

export function el<K extends keyof HTMLElementTagNameMap>(tag: K, text?: string | number, cls?: string): HTMLElementTagNameMap[K] {
  const e = document.createElement(tag);
  if (text !== undefined) e.textContent = String(text);
  if (cls) e.className = cls;
  return e;
}

export function table(headers: { label: string; num?: boolean }[], rows: (string | number)[][], emptyText: string): HTMLElement {
  if (rows.length === 0) return el("p", emptyText, "empty");
  const t = el("table");
  const tr = el("tr");
  for (const h of headers) tr.append(el("th", h.label, h.num ? "num" : undefined));
  const thead = el("thead"); thead.append(tr); t.append(thead);
  const body = t.appendChild(el("tbody"));
  for (const r of rows) {
    const row = el("tr");
    r.forEach((cell, i) => row.append(el("td", typeof cell === "number" ? cell.toLocaleString() : cell, headers[i]?.num ? "num" : undefined)));
    body.append(row);
  }
  return t;
}

export const fmtBytes = (n: number): string => (n >= 1024 ** 3 ? `${(n / 1024 ** 3).toFixed(2)} GiB` : `${(n / 1024 ** 2).toFixed(1)} MiB`);
export const pct = (x: number): string => `${(100 * x).toFixed(0)} %`;
export const sec = (ms: number): string => `${(ms / 1000).toFixed(1)} s`;
export const f = (n: number | null | undefined, digits = 1): string => (n === null || n === undefined || !Number.isFinite(n) ? "–" : n.toFixed(digits));
