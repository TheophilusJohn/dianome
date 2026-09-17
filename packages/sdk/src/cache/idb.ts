// LRU metadata for cached chunks in IndexedDB `dianome-meta`, object store `chunks` keyed by sha with an index on
// lastAccess. Kept out of the Cache API so listing/eviction never opens Response bodies.

import { META_DB } from "./types";

export interface ChunkMeta {
  sha: string;
  bytes: number;
  lastAccess: number;
  modelIds: string[];
}

const STORE = "chunks";
const VERSION = 1;

function req<T>(r: IDBRequest<T>): Promise<T> {
  return new Promise((resolve, reject) => {
    r.onsuccess = () => resolve(r.result);
    r.onerror = () => reject(r.error ?? new Error("IndexedDB request failed"));
  });
}
function done(tx: IDBTransaction): Promise<void> {
  return new Promise((resolve, reject) => {
    tx.oncomplete = () => resolve();
    tx.onerror = () => reject(tx.error ?? new Error("IndexedDB transaction failed"));
    tx.onabort = () => reject(tx.error ?? new Error("IndexedDB transaction aborted"));
  });
}

export class MetaStore {
  private db: Promise<IDBDatabase> | null = null;
  constructor(private readonly factory: IDBFactory, private readonly name = META_DB) {}

  private open(): Promise<IDBDatabase> {
    this.db ??= new Promise((resolve, reject) => {
      const r = this.factory.open(this.name, VERSION);
      r.onupgradeneeded = () => {
        const db = r.result;
        if (!db.objectStoreNames.contains(STORE)) {
          const os = db.createObjectStore(STORE, { keyPath: "sha" });
          os.createIndex("lastAccess", "lastAccess", { unique: false });
        }
      };
      r.onsuccess = () => {
        const db = r.result;
        db.onversionchange = () => { db.close(); this.db = null; };
        resolve(db);
      };
      r.onerror = () => { this.db = null; reject(r.error ?? new Error("IndexedDB open failed")); };
      r.onblocked = () => { this.db = null; reject(new Error("IndexedDB open blocked")); };
    });
    return this.db;
  }

  async get(sha: string): Promise<ChunkMeta | undefined> {
    const db = await this.open();
    return req(db.transaction(STORE, "readonly").objectStore(STORE).get(sha)) as Promise<ChunkMeta | undefined>;
  }

  async put(meta: ChunkMeta): Promise<void> {
    const db = await this.open();
    const tx = db.transaction(STORE, "readwrite");
    tx.objectStore(STORE).put(meta);
    await done(tx);
  }

  /** Records an access: bumps lastAccess and adds modelId to the owner set. No-op when the sha is unknown. */
  async touch(sha: string, modelId: string | undefined, at: number): Promise<void> {
    const db = await this.open();
    const tx = db.transaction(STORE, "readwrite");
    const os = tx.objectStore(STORE);
    const cur = (await req(os.get(sha))) as ChunkMeta | undefined;
    if (cur) {
      cur.lastAccess = at;
      if (modelId && !cur.modelIds.includes(modelId)) cur.modelIds.push(modelId);
      os.put(cur);
    }
    await done(tx);
  }

  async delete(shas: string[]): Promise<void> {
    if (shas.length === 0) return;
    const db = await this.open();
    const tx = db.transaction(STORE, "readwrite");
    const os = tx.objectStore(STORE);
    for (const sha of shas) os.delete(sha);
    await done(tx);
  }

  /** Every record, least recently used first. */
  async all(): Promise<ChunkMeta[]> {
    const db = await this.open();
    return req(db.transaction(STORE, "readonly").objectStore(STORE).index("lastAccess").getAll()) as Promise<ChunkMeta[]>;
  }

  async clear(): Promise<void> {
    const db = await this.open();
    const tx = db.transaction(STORE, "readwrite");
    tx.objectStore(STORE).clear();
    await done(tx);
  }

  close(): void {
    const p = this.db;
    this.db = null;
    void p?.then((db) => db.close()).catch(() => {});
  }
}
