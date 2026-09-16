// Top-level probe: WebGPU adapter limits, device memory, concurrency, UA.
// Run from the top-level page, not the iframe. Never throws; missing values
// are null.

export async function probe() {
  const out = {
    webgpu: false,
    maxBufferSize: null,
    maxStorageBufferBindingSize: null,
    maxComputeWorkgroupStorageSize: null,
    adapter: { vendor: null, architecture: null, device: null, description: null },
    deviceMemory: null,
    hardwareConcurrency: null,
    userAgent: null,
  };

  try { out.deviceMemory = navigator.deviceMemory ?? null; } catch { /* keep null */ }
  try { out.hardwareConcurrency = navigator.hardwareConcurrency ?? null; } catch { /* keep null */ }
  try { out.userAgent = navigator.userAgent ?? null; } catch { /* keep null */ }

  try {
    if (!navigator.gpu) return out;
    const adapter = await navigator.gpu.requestAdapter();
    if (!adapter) return out;
    out.webgpu = true;

    const limits = adapter.limits ?? {};
    out.maxBufferSize = limits.maxBufferSize ?? null;
    out.maxStorageBufferBindingSize = limits.maxStorageBufferBindingSize ?? null;
    out.maxComputeWorkgroupStorageSize = limits.maxComputeWorkgroupStorageSize ?? null;

    let info = null;
    if (adapter.info) {
      info = adapter.info;
    } else if (typeof adapter.requestAdapterInfo === 'function') {
      info = await adapter.requestAdapterInfo();
    }
    if (info) {
      out.adapter.vendor = info.vendor ?? null;
      out.adapter.architecture = info.architecture ?? null;
      out.adapter.device = info.device ?? null;
      out.adapter.description = info.description ?? null;
    }
  } catch { /* WebGPU probing failed; report what we have */ }

  return out;
}
