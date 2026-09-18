// Byte-level BPE tokenizer for Qwen2's tokenizer.json (HF `tokenizers` format): NFC normalizer, the GPT-2 style
// regex pre-tokenizer + ByteLevel mapping, BPE merges by rank, ByteLevel decoder. Added tokens (the <|...|>
// specials) are matched on the raw text before normalization, as `tokenizers` does. No wasm, no dependencies.

export interface AddedToken { id: number; content: string; special: boolean }

export interface TokenizerJson {
  added_tokens?: AddedToken[];
  normalizer?: { type: string } | null;
  pre_tokenizer?: { type: string; pretokenizers?: { type: string; pattern?: { Regex?: string } }[]; pattern?: { Regex?: string } } | null;
  model: { type: string; vocab: Record<string, number>; merges: (string | [string, string])[] };
}

// Unicode White_Space (what Oniguruma's `\s` means in the HF regex); JS `\s` differs (adds U+FEFF, lacks U+0085).
const WS = "\\t\\n\\v\\f\\r \\u0085\\u00a0\\u1680\\u2000-\\u200a\\u2028\\u2029\\u202f\\u205f\\u3000";

/** Translate the HF Split regex into a JS `u`-flag regex: inline (?i:) group, Unicode `\s`. */
export function translatePattern(pattern: string): RegExp {
  let p = pattern;
  p = p.replace(/\(\?i:([^)]*)\)/g, (_m, alt: string) => "(?:" + alt.split("|").map((a) => [...a].map((c) => {
    const lo = c.toLowerCase(), up = c.toUpperCase();
    return lo !== up ? `[${lo}${up}]` : c.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
  }).join("")).join("|") + ")");
  // \s / \S inside and outside character classes
  p = p.replace(/\[\^([^\]]*)\\s([^\]]*)\]/g, (_m, a: string, b: string) => `[^${a}${WS}${b}]`);
  p = p.replace(/\\s/g, `[${WS}]`).replace(/\\S/g, `[^${WS}]`);
  return new RegExp(p, "gu");
}

const QWEN2_PATTERN = "(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+";

function byteToUnicodeTables(): { b2u: string[]; u2b: Map<string, number> } {
  const bs: number[] = [];
  for (let b = 33; b <= 126; b++) bs.push(b);
  for (let b = 161; b <= 172; b++) bs.push(b);
  for (let b = 174; b <= 255; b++) bs.push(b);
  const cs = bs.slice();
  let n = 0;
  for (let b = 0; b < 256; b++) if (!bs.includes(b)) { bs.push(b); cs.push(256 + n); n++; }
  const b2u: string[] = new Array(256);
  const u2b = new Map<string, number>();
  for (let i = 0; i < bs.length; i++) { const ch = String.fromCodePoint(cs[i]!); b2u[bs[i]!] = ch; u2b.set(ch, bs[i]!); }
  return { b2u, u2b };
}

export class Tokenizer {
  private readonly vocab: Map<string, number>;
  private readonly ids: Map<number, string>;
  private readonly ranks: Map<string, number>;
  private readonly added: Map<string, AddedToken>;
  private readonly addedById: Map<number, AddedToken>;
  private readonly addedRe: RegExp | null;
  private readonly splitRe: RegExp;
  private readonly nfc: boolean;
  private readonly b2u: string[];
  private readonly u2b: Map<string, number>;
  private readonly cache = new Map<string, number[]>();
  private readonly enc = new TextEncoder();
  private readonly dec = new TextDecoder("utf-8", { fatal: false });

  constructor(json: TokenizerJson) {
    if (json.model.type !== "BPE") throw new Error(`unsupported tokenizer model ${json.model.type}`);
    this.vocab = new Map(Object.entries(json.model.vocab));
    this.ids = new Map();
    for (const [tok, id] of this.vocab) this.ids.set(id, tok);
    this.ranks = new Map();
    json.model.merges.forEach((m, i) => { const key = typeof m === "string" ? m : `${m[0]} ${m[1]}`; this.ranks.set(key, i); });
    this.added = new Map();
    this.addedById = new Map();
    for (const a of json.added_tokens ?? []) { this.added.set(a.content, a); this.addedById.set(a.id, a); }
    const contents = [...this.added.keys()].sort((a, b) => b.length - a.length);
    this.addedRe = contents.length ? new RegExp(contents.map((c) => c.replace(/[.*+?^${}()|[\]\\]/g, "\\$&")).join("|"), "gu") : null;
    this.nfc = json.normalizer?.type === "NFC";
    let pattern = QWEN2_PATTERN;
    const pre = json.pre_tokenizer;
    const split = pre?.type === "Sequence" ? pre.pretokenizers?.find((p) => p.type === "Split") : pre?.type === "Split" ? pre : undefined;
    if (split?.pattern?.Regex) pattern = split.pattern.Regex;
    this.splitRe = translatePattern(pattern);
    ({ b2u: this.b2u, u2b: this.u2b } = byteToUnicodeTables());
  }

  get size(): number { return this.vocab.size + [...this.addedById.keys()].filter((id) => !this.ids.has(id)).length; }

  tokenToId(token: string): number | undefined { return this.added.get(token)?.id ?? this.vocab.get(token); }

  encode(text: string): number[] {
    const out: number[] = [];
    if (this.addedRe) {
      let last = 0;
      for (const m of text.matchAll(this.addedRe)) {
        if (m.index > last) this.encodeText(text.slice(last, m.index), out);
        out.push(this.added.get(m[0])!.id);
        last = m.index + m[0].length;
      }
      if (last < text.length) this.encodeText(text.slice(last), out);
    } else this.encodeText(text, out);
    return out;
  }

  private encodeText(text: string, out: number[]): void {
    if (this.nfc) text = text.normalize("NFC");
    for (const m of text.matchAll(this.splitRe)) {
      const piece = m[0];
      if (!piece) continue;
      let ids = this.cache.get(piece);
      if (!ids) {
        ids = this.bpe(piece);
        if (this.cache.size > 50_000) this.cache.clear();
        this.cache.set(piece, ids);
      }
      for (const id of ids) out.push(id);
    }
  }

  private bpe(piece: string): number[] {
    const bytes = this.enc.encode(piece);
    let symbols: string[] = new Array(bytes.length);
    for (let i = 0; i < bytes.length; i++) symbols[i] = this.b2u[bytes[i]!]!;
    while (symbols.length > 1) {
      let best = -1, bestRank = Infinity;
      for (let i = 0; i < symbols.length - 1; i++) {
        const r = this.ranks.get(symbols[i] + " " + symbols[i + 1]);
        if (r !== undefined && r < bestRank) { bestRank = r; best = i; }
      }
      if (best < 0) break;
      const a = symbols[best]!, b = symbols[best + 1]!, ab = a + b;
      const next: string[] = [];
      for (let i = 0; i < symbols.length; i++) {
        if (i < symbols.length - 1 && symbols[i] === a && symbols[i + 1] === b) { next.push(ab); i++; } else next.push(symbols[i]!);
      }
      symbols = next;
    }
    return symbols.map((s) => {
      const id = this.vocab.get(s);
      if (id === undefined) throw new Error(`token ${JSON.stringify(s)} not in vocab`);
      return id;
    });
  }

  decode(ids: ArrayLike<number>): string {
    let text = "";
    let bytes: number[] = [];
    const flush = () => { if (bytes.length) { text += this.dec.decode(new Uint8Array(bytes)); bytes = []; } };
    for (let i = 0; i < ids.length; i++) {
      const id = ids[i]!;
      const added = this.addedById.get(id);
      if (added) { flush(); text += added.content; continue; }
      const tok = this.ids.get(id);
      if (tok === undefined) throw new Error(`unknown token id ${id}`);
      for (const ch of tok) {
        const b = this.u2b.get(ch);
        if (b === undefined) throw new Error(`token ${JSON.stringify(tok)} has a non byte-level char`);
        bytes.push(b);
      }
    }
    flush();
    return text;
  }
}

/** Build from the raw bytes of tokenizer.json (e.g. the `tokenizer.json` entry of a dianome manifest). */
export function tokenizerFromBytes(bytes: Uint8Array): Tokenizer {
  return new Tokenizer(JSON.parse(new TextDecoder().decode(bytes)) as TokenizerJson);
}
