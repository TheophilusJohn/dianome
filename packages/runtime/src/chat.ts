// Qwen2.5 chat template (ChatML), the non-tool branch of the HF `chat_template` in tokenizer_config.json:
//   <|im_start|>system\n{system}<|im_end|>\n   (Qwen's default system prompt when the first message is not a system one)
//   <|im_start|>{role}\n{content}<|im_end|>\n  for every message (a system message after the first is rendered in place)
//   <|im_start|>assistant\n                    when add_generation_prompt
// Validated against fixtures/.../chat_template_cases.json (HF apply_chat_template: rendered text and token ids).

export type ChatRole = "system" | "user" | "assistant";
export interface ChatMessage { role: ChatRole; content: string }

export const QWEN_DEFAULT_SYSTEM = "You are Qwen, created by Alibaba Cloud. You are a helpful assistant.";
export const IM_START = "<|im_start|>";
export const IM_END = "<|im_end|>";

export interface ChatTemplateOptions { addGenerationPrompt?: boolean; defaultSystem?: string }

export function renderChat(messages: readonly ChatMessage[], opts: ChatTemplateOptions = {}): string {
  const first = messages[0];
  let out = "";
  if (first && first.role === "system") out += `${IM_START}system\n${first.content}${IM_END}\n`;
  else out += `${IM_START}system\n${opts.defaultSystem ?? QWEN_DEFAULT_SYSTEM}${IM_END}\n`;
  messages.forEach((m, i) => {
    if (m.role === "system" && i === 0) return;
    if (m.role !== "system" && m.role !== "user" && m.role !== "assistant") throw new Error(`unsupported chat role ${String(m.role)}`);
    out += `${IM_START}${m.role}\n${m.content}${IM_END}\n`;
  });
  if (opts.addGenerationPrompt ?? true) out += `${IM_START}assistant\n`;
  return out;
}
