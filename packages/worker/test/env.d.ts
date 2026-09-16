// `env` from cloudflare:test is typed as Cloudflare.Env; bind it to the Worker's own Env.
import type { Env as WorkerEnv } from "../src/types";
declare global {
  namespace Cloudflare {
    interface Env extends WorkerEnv {}
  }
}
export {};
