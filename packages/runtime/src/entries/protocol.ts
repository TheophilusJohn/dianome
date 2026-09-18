// `dianome-runtime/protocol`: the split WebSocket client and frame codec (no WebGPU). Server mode (N = 0) needs
// only this and the tokenizer.
export { SplitClient, encodeFrame, decodeFrame, MAGIC, type Message, type MessageType, type TokenMessage } from "../protocol";
