// The Phase 4 split protocol, browser side. Frame = u32 magic "DNM1" (0x444E4D31 little-endian) | u32 header_len
// | JSON header | payload. Payloads are little-endian fp16 [T, d_model] hidden states (N >= 1) or int32 token
// ids (N = 0). Mirrors server/dianome_server/protocol.py.

export const MAGIC = 0x444e4d31;

export type MessageType = "open" | "opened" | "prefill" | "decode" | "token" | "stats" | "close" | "error" | "ping" | "pong";

export interface Message { type: MessageType; header: Record<string, unknown>; payload: Uint8Array }

export function encodeFrame(type: MessageType, header: Record<string, unknown> = {}, payload: ArrayBufferView | null = null): ArrayBuffer {
  const hb = new TextEncoder().encode(JSON.stringify({ ...header, type }));
  const plen = payload ? payload.byteLength : 0;
  const buf = new ArrayBuffer(8 + hb.byteLength + plen);
  const dv = new DataView(buf);
  dv.setUint32(0, MAGIC, true);
  dv.setUint32(4, hb.byteLength, true);
  const u8 = new Uint8Array(buf);
  u8.set(hb, 8);
  if (payload) u8.set(new Uint8Array(payload.buffer, payload.byteOffset, payload.byteLength), 8 + hb.byteLength);
  return buf;
}

export function decodeFrame(buf: ArrayBuffer): Message {
  const dv = new DataView(buf);
  if (buf.byteLength < 8) throw new Error(`frame too short (${buf.byteLength} bytes)`);
  const magic = dv.getUint32(0, true);
  if (magic !== MAGIC) throw new Error(`bad magic 0x${magic.toString(16)}`);
  const hlen = dv.getUint32(4, true);
  if (8 + hlen > buf.byteLength) throw new Error("header_len exceeds frame");
  const header = JSON.parse(new TextDecoder().decode(new Uint8Array(buf, 8, hlen))) as Record<string, unknown>;
  const type = header.type as MessageType;
  if (typeof type !== "string") throw new Error("header has no type");
  return { type, header, payload: new Uint8Array(buf, 8 + hlen) };
}

export interface TokenMessage { id: number; position: number; busy_ms: number; done: boolean }

/** One session over a WebSocket. Browsers cannot set upgrade headers, so the token travels as `?token=`. */
export class SplitClient {
  private ws: WebSocket | null = null;
  private queue: ((m: Message) => void)[] = [];
  private inbox: Message[] = [];
  private closed: Error | null = null;
  opened: Record<string, unknown> | null = null;

  constructor(readonly url: string, readonly token: string) {}

  async connect(): Promise<void> {
    const u = new URL(this.url);
    u.searchParams.set("token", this.token);
    const ws = new WebSocket(u.toString());
    ws.binaryType = "arraybuffer";
    this.ws = ws;
    await new Promise<void>((resolve, reject) => {
      ws.onopen = () => resolve();
      ws.onerror = () => reject(new Error(`WebSocket to ${this.url} failed`));
    });
    ws.onmessage = (ev) => {
      if (!(ev.data instanceof ArrayBuffer)) return;
      const m = decodeFrame(ev.data);
      const w = this.queue.shift();
      if (w) w(m); else this.inbox.push(m);
    };
    ws.onclose = (ev) => { this.closed = new Error(`WebSocket closed (${ev.code} ${ev.reason})`); for (const w of this.queue.splice(0)) w({ type: "error", header: { code: "closed", message: this.closed.message }, payload: new Uint8Array() }); };
    ws.onerror = () => { /* onclose follows */ };
  }

  private next(): Promise<Message> {
    const m = this.inbox.shift();
    if (m) return Promise.resolve(m);
    if (this.closed) return Promise.reject(this.closed);
    return new Promise((resolve) => this.queue.push(resolve));
  }

  async send(type: MessageType, header: Record<string, unknown> = {}, payload: ArrayBufferView | null = null): Promise<Message> {
    if (!this.ws) throw new Error("not connected");
    this.ws.send(encodeFrame(type, header, payload));
    const m = await this.next();
    if (m.type === "error") throw new Error(`server error ${String(m.header.code)}: ${String(m.header.message)}`);
    return m;
  }

  async open(model: string, N: number, maxCtx = 4096, sampling: Record<string, unknown> = { temperature: 0 }): Promise<Record<string, unknown>> {
    const m = await this.send("open", { model, N, max_ctx: maxCtx, sampling });
    if (m.type !== "opened") throw new Error(`expected opened, got ${m.type}`);
    this.opened = m.header;
    return m.header;
  }

  /** fp16 [T, d_model] bits (N >= 1). */
  async prefill(hidden: Uint16Array, T: number, start = 0): Promise<TokenMessage> {
    const m = await this.send("prefill", { T, positions: [start, start + T] }, hidden);
    if (m.type !== "token") throw new Error(`expected token, got ${m.type}`);
    return m.header as unknown as TokenMessage;
  }

  async decode(hidden: Uint16Array, position: number): Promise<TokenMessage> {
    const m = await this.send("decode", { position }, hidden);
    if (m.type !== "token") throw new Error(`expected token, got ${m.type}`);
    return m.header as unknown as TokenMessage;
  }

  /** int32 token ids [T] (N = 0). */
  async prefillIds(ids: Int32Array, start = 0): Promise<TokenMessage> {
    const m = await this.send("prefill", { T: ids.length, positions: [start, start + ids.length] }, ids);
    if (m.type !== "token") throw new Error(`expected token, got ${m.type}`);
    return m.header as unknown as TokenMessage;
  }

  async decodeId(id: number, position: number): Promise<TokenMessage> {
    const m = await this.send("decode", { position }, new Int32Array([id]));
    if (m.type !== "token") throw new Error(`expected token, got ${m.type}`);
    return m.header as unknown as TokenMessage;
  }

  /** Round-trip time of one ping/pong in ms (allowed before open). */
  async ping(): Promise<number> {
    const t0 = performance.now();
    const m = await this.send("ping", { t: t0 });
    if (m.type !== "pong") throw new Error(`expected pong, got ${m.type}`);
    return performance.now() - t0;
  }

  /** Median of `n` pings. */
  async rtt(n = 5): Promise<number> {
    const s: number[] = [];
    for (let i = 0; i < n; i++) s.push(await this.ping());
    s.sort((a, b) => a - b);
    return s[Math.floor(s.length / 2)]!;
  }

  async stats(): Promise<Record<string, unknown>> { return (await this.send("stats", {})).header; }

  close(): void {
    if (this.ws && this.ws.readyState === WebSocket.OPEN) { this.ws.send(encodeFrame("close", {})); this.ws.close(); }
    this.ws = null;
  }
}
