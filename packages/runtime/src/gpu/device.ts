// Adapter/device acquisition and limits. shader-f16 is detected and reported but never required: weights are
// unpacked with unpack2x16float, activations are f32. No single buffer may exceed 1 GiB (Firefox's floor from
// Phase 0), so maxBufferSize is requested capped at that even when the adapter allows more.

export const MAX_BUFFER = 1 << 30;

export interface GpuInfo {
  vendor: string;
  architecture: string;
  device: string;
  description: string;
  shaderF16: boolean;
  subgroups: boolean;
  timestampQuery: boolean;
  limits: { maxBufferSize: number; maxStorageBufferBindingSize: number; maxComputeWorkgroupStorageSize: number; maxComputeInvocationsPerWorkgroup: number; maxComputeWorkgroupsPerDimension: number };
}

export interface AcquiredDevice { device: GPUDevice; adapter: GPUAdapter; info: GpuInfo }

export async function acquireDevice(opts: { powerPreference?: GPUPowerPreference } = {}): Promise<AcquiredDevice> {
  const gpu = (globalThis.navigator as Navigator | undefined)?.gpu;
  if (!gpu) throw new Error("WebGPU is not available (navigator.gpu is undefined; secure context and a supporting browser are required)");
  const adapter = await gpu.requestAdapter(opts.powerPreference ? { powerPreference: opts.powerPreference } : {});
  if (!adapter) throw new Error("WebGPU: no adapter");
  const lim = adapter.limits;
  const requiredLimits: Record<string, number> = {
    maxBufferSize: Math.min(lim.maxBufferSize, MAX_BUFFER),
    maxStorageBufferBindingSize: Math.min(lim.maxStorageBufferBindingSize, MAX_BUFFER),
  };
  // Larger workgroup memory helps the attention kernel hold longer contexts; ask for what the adapter has.
  if (lim.maxComputeWorkgroupStorageSize > 16384) requiredLimits.maxComputeWorkgroupStorageSize = lim.maxComputeWorkgroupStorageSize;
  const requiredFeatures: GPUFeatureName[] = [];
  const timestampQuery = adapter.features.has("timestamp-query");
  if (timestampQuery) requiredFeatures.push("timestamp-query");
  const device = await adapter.requestDevice({ requiredLimits, requiredFeatures });
  const ai = (adapter as GPUAdapter & { info?: GPUAdapterInfo }).info ?? ({} as Partial<GPUAdapterInfo>);
  const info: GpuInfo = {
    vendor: ai.vendor ?? "", architecture: ai.architecture ?? "", device: ai.device ?? "", description: ai.description ?? "",
    shaderF16: adapter.features.has("shader-f16"),
    subgroups: adapter.features.has("subgroups" as GPUFeatureName),
    timestampQuery,
    limits: {
      maxBufferSize: device.limits.maxBufferSize,
      maxStorageBufferBindingSize: device.limits.maxStorageBufferBindingSize,
      maxComputeWorkgroupStorageSize: device.limits.maxComputeWorkgroupStorageSize,
      maxComputeInvocationsPerWorkgroup: device.limits.maxComputeInvocationsPerWorkgroup,
      maxComputeWorkgroupsPerDimension: device.limits.maxComputeWorkgroupsPerDimension,
    },
  };
  return { device, adapter, info };
}
