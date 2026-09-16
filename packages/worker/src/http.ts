// Response helpers. Every response carries the CORS headers; OPTIONS is answered in index.ts.

export const CORS_HEADERS: Record<string, string> = {
  "Access-Control-Allow-Origin": "*",
  "Access-Control-Expose-Headers": "ETag, Cache-Control, Content-Length, X-Dianome-Manifest-Sha",
};

export function withCors(res: Response): Response {
  for (const [k, v] of Object.entries(CORS_HEADERS)) res.headers.set(k, v);
  return res;
}

export function json(body: unknown, init: ResponseInit = {}): Response {
  const headers = new Headers(init.headers);
  headers.set("Content-Type", "application/json; charset=utf-8");
  return new Response(JSON.stringify(body), { ...init, headers });
}

export function error(status: number, code: string, detail?: string): Response {
  return json(detail ? { error: code, detail } : { error: code }, { status });
}

export function preflight(): Response {
  return new Response(null, {
    status: 204,
    headers: {
      "Access-Control-Allow-Methods": "GET, HEAD, POST, OPTIONS",
      "Access-Control-Allow-Headers": "Content-Type, If-None-Match",
      "Access-Control-Max-Age": "86400",
    },
  });
}

export function sha256Hex(data: ArrayBuffer | Uint8Array): Promise<string> {
  return crypto.subtle.digest("SHA-256", data as BufferSource).then((d) =>
    Array.from(new Uint8Array(d), (b) => b.toString(16).padStart(2, "0")).join(""),
  );
}
