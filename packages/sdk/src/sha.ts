// SHA-256 helpers on WebCrypto; used on every chunk from every source.

export function hex(buf: ArrayBuffer): string {
  const u = new Uint8Array(buf);
  let s = "";
  for (let i = 0; i < u.length; i++) s += HEX[u[i]!];
  return s;
}
const HEX = Array.from({ length: 256 }, (_, i) => i.toString(16).padStart(2, "0"));

export async function sha256(data: ArrayBuffer | Uint8Array): Promise<string> {
  return hex(await crypto.subtle.digest("SHA-256", data as BufferSource));
}

export const SHA_RE = /^[0-9a-f]{64}$/;
