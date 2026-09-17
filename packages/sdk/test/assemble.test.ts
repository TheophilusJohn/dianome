import { describe, expect, it } from "vitest";
import { assembleBytes, assembleGroup } from "../src/assemble";
import { planVariant } from "../src/plan";
import { synthModel } from "./fixtures/synth";

const toBuffers = (chunks: Map<string, Uint8Array>): Map<string, ArrayBuffer> =>
  new Map([...chunks].map(([sha, u]) => [sha, u.buffer.slice(u.byteOffset, u.byteOffset + u.byteLength) as ArrayBuffer]));

describe("assemble", () => {
  it("single-segment entries are zero-copy views at 256-byte-aligned offsets", async () => {
    const { manifest, chunks, raw } = await synthModel();
    const plan = planVariant(manifest, "fp16");
    const bufs = toBuffers(chunks);
    const layer0 = plan.groups[1]!;
    const entries = assembleGroup(layer0, bufs);
    expect([...entries.keys()]).toEqual(layer0.entries.map((e) => e.name));
    for (const [name, e] of entries) {
      expect(e.copied).toBe(false);
      expect(e.bytes.byteOffset % 256).toBe(0);
      expect(e.bytes.buffer).toBe(bufs.get(layer0.needs[0]!));
      expect(e.bytes).toEqual(raw.get(`fp16/${name}`));
    }
  });

  it("multi-chunk entries are concatenated once into a fresh buffer", async () => {
    // 64x256 fp16 embed = 32768 bytes; 4096-byte chunks → 8 segments.
    const { manifest, chunks, raw } = await synthModel({ chunkSize: 4096 });
    const plan = planVariant(manifest, "fp16");
    const embed = plan.groups[0]!;
    expect(embed.entries[0]!.segments.length).toBe(8);
    const e = assembleGroup(embed, toBuffers(chunks)).get("model.embed_tokens.weight")!;
    expect(e.copied).toBe(true);
    expect(e.bytes.byteOffset).toBe(0);
    expect(e.bytes.byteLength).toBe(32768);
    expect(e.bytes).toEqual(raw.get("fp16/model.embed_tokens.weight"));
  });

  it("an entry that straddles a chunk boundary inside a group stream is concatenated correctly", async () => {
    // layer stream: norm (512 B padded), q_proj (131072 B), bias (512 B). With 65536-byte chunks q_proj spans 3 chunks.
    const { manifest, chunks, raw } = await synthModel({ chunkSize: 65536 });
    const plan = planVariant(manifest, "fp16");
    const layer = plan.groups[1]!;
    const q = layer.entries.find((e) => e.role === "attn.q_proj.weight")!;
    expect(q.segments.length).toBeGreaterThan(1);
    expect(q.segments[0]!.offset).toBe(512);
    const entries = assembleGroup(layer, toBuffers(chunks));
    expect(entries.get(q.name)!.bytes).toEqual(raw.get(`fp16/${q.name}`));
    expect(entries.get(q.name)!.copied).toBe(true);
    const bias = layer.entries.find((e) => e.role === "attn.q_proj.bias")!;
    expect(entries.get(bias.name)!.copied).toBe(false);
    expect(entries.get(bias.name)!.bytes).toEqual(raw.get(`fp16/${bias.name}`));
  });

  it("quantized parts are sub-views of the entry at their 256-aligned offsets", async () => {
    const { manifest, chunks, raw } = await synthModel();
    const plan = planVariant(manifest, "q4");
    const layer = plan.groups[1]!;
    const entries = assembleGroup(layer, toBuffers(chunks));
    const q = entries.get("model.layers.0.self_attn.q_proj.weight")!;
    expect(q.storage.kind).toBe("q4");
    const st = q.storage as Extract<typeof q.storage, { kind: "q4" }>;
    const rawQ = raw.get(`q4/${q.name}`)!;
    expect(q.parts!.weights.byteLength).toBe(st.parts.weights.length);
    expect(q.parts!.scales.byteOffset - q.bytes.byteOffset).toBe(st.parts.scales.offset);
    expect(q.parts!.zeros!.byteOffset - q.bytes.byteOffset).toBe(st.parts.zeros.offset);
    expect(q.parts!.zeros!).toEqual(rawQ.subarray(st.parts.zeros.offset, st.parts.zeros.offset + st.parts.zeros.length));
    expect(q.parts!.weights.buffer).toBe(q.bytes.buffer);
    const embed = assembleGroup(plan.groups[0]!, toBuffers(chunks)).get("model.embed_tokens.weight")!;
    expect(embed.storage.kind).toBe("q8");
    expect(embed.parts!.zeros).toBeUndefined();
  });

  it("tied lm_head resolves from the embed group's chunks without chunks of its own", async () => {
    const { manifest, chunks, raw } = await synthModel();
    const plan = planVariant(manifest, "fp16");
    const lm = plan.groups.at(-1)!;
    expect(lm.tied).toBe(true);
    expect(lm.bytes).toBe(0);
    expect(lm.needs).toEqual(plan.groups[0]!.needs);
    const e = assembleGroup(lm, toBuffers(chunks)).get("lm_head.weight")!;
    expect(e.tied).toBe(true);
    expect(e.bytes).toEqual(raw.get("fp16/lm_head.weight"));
    // The download order lists embed's chunks once, attributed to embed; lm_head adds nothing.
    expect(plan.order.filter((c) => c.group === "lm_head")).toHaveLength(0);
    expect(new Set(plan.order.map((c) => c.sha)).size).toBe(plan.order.length);
    // lastUse of embed's chunks is the lm_head group, so the loader keeps them until then.
    for (const sha of plan.groups[0]!.needs) expect(plan.lastUse.get(sha)).toBe(lm.index);
  });

  it("throws when a chunk is missing or a segment overruns", () => {
    const sha = "a".repeat(64);
    expect(() => assembleBytes([{ chunk: sha, offset: 0, length: 4 }], new Map())).toThrow(/not loaded/);
    expect(() => assembleBytes([{ chunk: sha, offset: 0, length: 4 }], new Map([[sha, new ArrayBuffer(2)]]))).toThrow(/exceeds chunk/);
    expect(() => assembleBytes([], new Map())).toThrow(/no segments/);
  });
});
