import { readFileSync } from "node:fs";
import { resolve } from "node:path";
import Ajv2020 from "ajv/dist/2020";
import { describe, expect, it } from "vitest";
import { ManifestError } from "../src/errors";
import { fetchManifest, validateManifest } from "../src/manifest";
import { synthFiles, synthModel } from "./fixtures/synth";
import { FakeFetch } from "./helpers/fakeFetch";

const schema = JSON.parse(readFileSync(resolve(__dirname, "../../../schemas/manifest.v1.json"), "utf8"));
const ajv = new Ajv2020({ allErrors: true, strict: false });
const bySchema = ajv.compile(schema);

const clone = <T>(v: T): T => JSON.parse(JSON.stringify(v)) as T;

describe("validateManifest", () => {
  it("accepts synthetic model and files manifests, and ajv agrees", async () => {
    const m = (await synthModel()).manifest;
    const f = (await synthFiles([{ name: "a/b.bin", length: 1000, seed: 1 }, { name: "empty", length: 0, seed: 2 }])).manifest;
    for (const x of [m, f]) {
      expect(bySchema(x), JSON.stringify(bySchema.errors)).toBe(true);
      expect(validateManifest(clone(x))).toEqual(x);
    }
  });

  it("accepts the real manifests in ./store when present (model, onnx, mlc)", () => {
    for (const id of ["qwen2.5-0.5b-instruct", "qwen2.5-0.5b-instruct-onnx", "qwen2.5-0.5b-instruct-mlc"]) {
      let raw: string;
      try { raw = readFileSync(resolve(__dirname, `../../../store/manifests/${id}/latest.json`), "utf8"); } catch { continue; }
      const parsed = JSON.parse(raw);
      expect(bySchema(parsed), id).toBe(true);
      const v = validateManifest(parsed);
      expect(v.id).toBe(id);
    }
  });

  // Every mutation below is rejected by ajv AND by the hand validator, so the two cannot drift apart silently.
  const mutations: [string, (m: any) => void][] = [
    ["wrong schema", (m) => { m.schema = 2; }],
    ["bad id", (m) => { m.id = "Bad Id"; }],
    ["unknown top-level key", (m) => { m.extra = 1; }],
    ["chunk_size", (m) => { m.chunk_size = 1024; }],
    ["chunk key not sha", (m) => { m.chunks["zz"] = { bytes: 1 }; }],
    ["chunk bytes 0", (m) => { const k = Object.keys(m.chunks)[0]!; m.chunks[k].bytes = 0; }],
    ["group name", (m) => { m.variants.fp16.groups[1].name = "block.0"; }],
    ["segment offset too large", (m) => { m.variants.fp16.groups[1].entries[0].segments[0].offset = 8388608; }],
    ["part offset unaligned", (m) => { m.variants.q4.groups[1].entries[1].storage.parts.scales.offset += 1; }],
    ["q4 group_size", (m) => { m.variants.q4.groups[1].entries[1].storage.group_size = 64; }],
    ["storage kind", (m) => { m.variants.fp16.groups[1].entries[0].storage.kind = "int4"; }],
    ["tied not true", (m) => { m.variants.fp16.groups.at(-1).tied = false; }],
    ["entry missing role", (m) => { delete m.variants.fp16.groups[1].entries[0].role; }],
    ["shape dim 0", (m) => { m.variants.fp16.groups[1].entries[0].shape = [0]; }],
    ["model manifest with files", (m) => { m.files = []; }],
    ["config missing", (m) => { delete m.config; }],
    ["runtime enum", (m) => { delete m.variants; delete m.config; delete m.tokenizer; delete m.family; m.runtime = "torch"; m.files = []; }],
    ["variants missing fp16", (m) => { delete m.variants.fp16; }],
  ];
  for (const [name, mutate] of mutations) {
    it(`rejects: ${name}`, async () => {
      const m = clone((await synthModel()).manifest) as any;
      mutate(m);
      expect(bySchema(m), "ajv should reject too").toBe(false);
      expect(() => validateManifest(m)).toThrow(ManifestError);
    });
  }

  it("rejects cross-reference errors the schema cannot express", async () => {
    const base = (await synthModel()).manifest;
    const unknownChunk = clone(base) as any;
    unknownChunk.variants.fp16.groups[1].chunks[0] = "f".repeat(64);
    expect(() => validateManifest(unknownChunk)).toThrow(/not in chunk table/);

    const wrongSum = clone(base) as any;
    wrongSum.variants.fp16.groups[1].bytes += 1;
    expect(() => validateManifest(wrongSum)).toThrow(/chunks sum to/);

    const segTooLong = clone(base) as any;
    segTooLong.variants.fp16.groups[1].entries[0].segments[0].length = 8388608;
    expect(() => validateManifest(segTooLong)).toThrow(/exceeds chunk/);

    const tiedWithChunks = clone(base) as any;
    const lm = tiedWithChunks.variants.fp16.groups.at(-1);
    lm.chunks = [tiedWithChunks.variants.fp16.groups[0].chunks[0]];
    expect(() => validateManifest(tiedWithChunks)).toThrow(/tied group|chunks sum/);

    const f = (await synthFiles([{ name: "x", length: 10, seed: 1 }])).manifest;
    const fileSum = clone(f) as any;
    fileSum.files[0].bytes = 11;
    expect(() => validateManifest(fileSum)).toThrow(/chunks sum to|segment/);
  });

  it("reports the path of the offending field", async () => {
    const m = clone((await synthModel()).manifest) as any;
    m.variants.q4.groups[1].entries[1].storage.parts.zeros.length = -1;
    expect(() => validateManifest(m)).toThrow(/manifest\.variants\.q4\.groups\[1\]\.entries\[1\]\.storage\.parts\.zeros\.length/);
  });
});

describe("fetchManifest", () => {
  it("GETs /v1/models/<id>/manifest, validates, and returns the sha header", async () => {
    const { manifest } = await synthModel();
    const ff = new FakeFetch("https://api.test", "https://cdn.test", new Map(), new Map([[manifest.id, manifest]]), "ab".repeat(32));
    const r = await fetchManifest("https://api.test/", manifest.id, { fetch: ff.fetch });
    expect(r.manifest).toEqual(manifest);
    expect(r.sha).toBe("ab".repeat(32));
    expect(ff.requests[0]!.url).toBe("https://api.test/v1/models/synth-model/manifest");
  });
  it("throws ManifestError on 404, on invalid JSON, on a bad id and on an id mismatch", async () => {
    const { manifest } = await synthModel();
    const ff = new FakeFetch("https://api.test", "https://cdn.test", new Map(), new Map([["other-id", manifest]]));
    await expect(fetchManifest("https://api.test", "missing", { fetch: ff.fetch })).rejects.toMatchObject({ code: "manifest_http" });
    await expect(fetchManifest("https://api.test", "other-id", { fetch: ff.fetch })).rejects.toMatchObject({ code: "manifest_invalid" });
    await expect(fetchManifest("https://api.test", "Bad/Id", { fetch: ff.fetch })).rejects.toMatchObject({ code: "manifest_invalid" });
    const bad = async () => new Response("{not json", { status: 200 });
    await expect(fetchManifest("https://api.test", "x", { fetch: bad })).rejects.toMatchObject({ code: "manifest_invalid" });
  });
});
