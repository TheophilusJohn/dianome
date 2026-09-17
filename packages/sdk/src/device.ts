// Device facts for the load report and the planner: WebGPU limits, storage estimate, browser family (coarse), and
// throughput measured from the load itself. Nothing finer than the family is ever derived from the user agent.

export type Browser = "chrome" | "firefox" | "safari" | "other";

export interface DeviceInfo {
  browser: Browser;
  webgpu: boolean;
  maxBufferSize: number | null;
  maxStorageBufferBindingSize: number | null;
  maxComputeWorkgroupStorageSize: number | null;
  quotaBytes: number | null;
  usageBytes: number | null;
}

export function detectBrowser(ua: string): Browser {
  if (/firefox\//i.test(ua)) return "firefox";
  if (/edg\/|chrome\/|chromium\//i.test(ua)) return "chrome";
  if (/safari\//i.test(ua) && !/chrome\//i.test(ua)) return "safari";
  return "other";
}

interface GpuLike { requestAdapter(): Promise<{ limits: Record<string, number | undefined> } | null> }

function num(v: unknown): number | null { return typeof v === "number" && Number.isFinite(v) ? v : null; }

export async function probeDevice(nav: Partial<Navigator> | undefined = typeof navigator !== "undefined" ? navigator : undefined): Promise<DeviceInfo> {
  const info: DeviceInfo = {
    browser: detectBrowser(nav?.userAgent ?? ""), webgpu: false,
    maxBufferSize: null, maxStorageBufferBindingSize: null, maxComputeWorkgroupStorageSize: null,
    quotaBytes: null, usageBytes: null,
  };
  const gpu = (nav as { gpu?: GpuLike } | undefined)?.gpu;
  if (gpu) {
    try {
      const adapter = await gpu.requestAdapter();
      if (adapter) {
        info.webgpu = true;
        info.maxBufferSize = num(adapter.limits.maxBufferSize);
        info.maxStorageBufferBindingSize = num(adapter.limits.maxStorageBufferBindingSize);
        info.maxComputeWorkgroupStorageSize = num(adapter.limits.maxComputeWorkgroupStorageSize);
      }
    } catch { /* no adapter */ }
  }
  try {
    const est = await nav?.storage?.estimate();
    info.quotaBytes = num(est?.quota);
    info.usageBytes = num(est?.usage);
  } catch { /* not reported */ }
  return info;
}
