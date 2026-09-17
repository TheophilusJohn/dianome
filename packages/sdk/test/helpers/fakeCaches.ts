// In-memory CacheStorage: enough of the Cache API for the per-site store, with a byte capacity that makes
// `put` throw QuotaExceededError the way browsers do.

import { quotaError } from "../../src/errors";

export class FakeCache {
  readonly entries = new Map<string, { body: Uint8Array; headers: Record<string, string> }>();
  constructor(private readonly owner: FakeCacheStorage) {}
  private key(req: RequestInfo | URL): string { return typeof req === "string" ? req : req instanceof URL ? req.href : req.url; }
  async match(req: RequestInfo | URL): Promise<Response | undefined> {
    const e = this.entries.get(this.key(req));
    if (!e) return undefined;
    return new Response(e.body.slice(), { headers: e.headers });
  }
  async put(req: RequestInfo | URL, res: Response): Promise<void> {
    const body = new Uint8Array(await res.arrayBuffer());
    const k = this.key(req);
    const prev = this.entries.get(k)?.body.length ?? 0;
    if (this.owner.used - prev + body.length > this.owner.capacity) throw quotaError();
    const headers: Record<string, string> = {};
    res.headers.forEach((v, h) => { headers[h] = v; });
    this.entries.set(k, { body, headers });
    this.owner.puts++;
  }
  async delete(req: RequestInfo | URL): Promise<boolean> { return this.entries.delete(this.key(req)); }
  async keys(): Promise<Request[]> { return [...this.entries.keys()].map((u) => new Request(u)); }
  get bytes(): number { let n = 0; for (const e of this.entries.values()) n += e.body.length; return n; }
}

export class FakeCacheStorage {
  readonly caches = new Map<string, FakeCache>();
  puts = 0;
  constructor(public capacity = Number.POSITIVE_INFINITY) {}
  get used(): number { let n = 0; for (const c of this.caches.values()) n += c.bytes; return n; }
  async open(name: string): Promise<Cache> {
    let c = this.caches.get(name);
    if (!c) { c = new FakeCache(this); this.caches.set(name, c); }
    return c as unknown as Cache;
  }
  async delete(name: string): Promise<boolean> { return this.caches.delete(name); }
  async has(name: string): Promise<boolean> { return this.caches.has(name); }
  async keys(): Promise<string[]> { return [...this.caches.keys()]; }
  async match(): Promise<Response | undefined> { return undefined; }
  asCacheStorage(): CacheStorage { return this as unknown as CacheStorage; }
}
