// `dianome-runtime/session`: SplitSession, the split WebSocket client, the frame codec and the CPU sampler. No
// WebGPU, no kernels (the Runtime is only referenced by type): server mode (N = 0) runs entirely on this entry
// plus the tokenizer, and the SDK's run() imports the main entry only when a plan puts blocks on this device.
export { SplitSession, type SessionMode, type SessionOptions, type StepTiming, type GeneratedToken } from "../session";
export { SplitClient, encodeFrame, decodeFrame, MAGIC, type Message, type MessageType, type TokenMessage } from "../protocol";
export { Sampler, Prng, argmax, topk, type SamplingOptions } from "../lmhead";
