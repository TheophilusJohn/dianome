// api.dianome.dev — the only Worker on the path. Chunk bytes never come through here (cdn.dianome.dev is R2 + edge cache).

import { error, preflight, json, withCors } from "./http";
import { hashedManifest, latestManifest, listModels } from "./manifest";
import { loadStats } from "./stats";
import { createSession, planProxy, rates, servers } from "./split";
import { ingestLoad } from "./telemetry";
import type { Env } from "./types";

async function route(request: Request, env: Env, ctx: ExecutionContext): Promise<Response> {
  const url = new URL(request.url);
  const path = url.pathname.replace(/\/+$/, "") || "/";
  const method = request.method === "HEAD" ? "GET" : request.method;

  if (request.method === "OPTIONS") return preflight();

  if (path === "/healthz" && method === "GET") return json({ ok: true, version: env.GIT_SHA ?? "dev" }, { headers: { "Cache-Control": "no-store" } });

  const parts = path.split("/").slice(1);
  if (parts[0] === "v1") {
    if (parts[1] === "models") {
      if (parts.length === 2 && method === "GET") return listModels(env);
      if (parts.length === 4 && parts[3] === "manifest" && method === "GET") return latestManifest(request, env, ctx, parts[2]!);
      if (parts.length === 5 && parts[3] === "manifest" && method === "GET") return hashedManifest(request, env, parts[2]!, parts[4]!);
    }
    if (path === "/v1/telemetry/load") {
      if (request.method === "POST") return ingestLoad(request, env);
      return error(405, "method_not_allowed");
    }
    if (path === "/v1/stats/loads" && method === "GET") return loadStats(env, ctx);
    if (path === "/v1/split/session") {
      if (request.method === "POST") return createSession(request, env);
      return error(405, "method_not_allowed");
    }
    if (path === "/v1/split/rates" && method === "GET") return rates();
    if (path === "/v1/split/servers" && method === "GET") return servers(env);
    if (path === "/v1/split/plan" && method === "GET") return planProxy(request, env);
  }
  return error(404, "not_found");
}

export default {
  async fetch(request, env, ctx): Promise<Response> {
    try {
      return withCors(await route(request, env, ctx));
    } catch (e) {
      console.error(e);
      return withCors(error(500, "internal"));
    }
  },
} satisfies ExportedHandler<Env>;
