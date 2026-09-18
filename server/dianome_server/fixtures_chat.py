"""The 50 message lists for Phase 5b's TypeScript chat template (Qwen2.5 ChatML).

Rendered strings and token ids come from the HF tokenizer's `apply_chat_template`; the TS
side must match both. Cases cover: no system message (the template inserts Qwen's default),
explicit system messages, multi-turn, empty content, unicode, code, whitespace, and both
values of add_generation_prompt. No tools (the TS template does not implement them).
"""

from __future__ import annotations

from .fixtures_cases import CJK, CODE, EMOJI, WIKI


def _u(content: str) -> dict:
    return {"role": "user", "content": content}


def _a(content: str) -> dict:
    return {"role": "assistant", "content": content}


def _s(content: str) -> dict:
    return {"role": "system", "content": content}


def cases() -> list[dict]:
    out: list[dict] = []
    add = lambda msgs, gen=True: out.append({"messages": msgs, "add_generation_prompt": gen})  # noqa: E731
    # 1-10: single user turn, default system prompt
    for text in [
        "Hello!", "What is the capital of France?", "Write a haiku about rain.", WIKI[0], WIKI[3],
        "Explain split inference in one sentence.", "1+1=?", "Translate 'good morning' to Spanish.",
        "List three primes.", "Why is the sky blue?",
    ]:
        add([_u(text)])
    # 11-18: explicit system prompt
    add([_s("You are a terse assistant."), _u("Hi")])
    add([_s("Answer only in JSON."), _u("Give me a colour.")])
    add([_s("You are Qwen, created by Alibaba Cloud. You are a helpful assistant."), _u("Repeat after me: ok")])
    add([_s(""), _u("Empty system prompt.")])
    add([_s("Multi\nline\nsystem prompt."), _u("Go.")])
    add([_s("System with trailing space "), _u("Go.")])
    add([_s(WIKI[5]), _u(WIKI[6])])
    add([_s("Be brief."), _u("Summarise: " + WIKI[7])])
    # 19-28: multi-turn
    add([_u("Hi"), _a("Hello! How can I help?"), _u("Tell me a joke.")])
    add([_s("Be brief."), _u("Hi"), _a("Hi."), _u("Bye"), _a("Bye."), _u("Actually, one more thing.")])
    add([_u("2+2"), _a("4"), _u("times 3"), _a("12"), _u("minus 1")])
    add([_u(WIKI[10]), _a(WIKI[11]), _u(WIKI[12])])
    add([_u("Write code"), _a(CODE[0]), _u("Now in Python")])
    add([_u("Hi"), _a("")])  # empty assistant content
    add([_u("Hi"), _a("Hello"), _u("")])  # empty user content
    add([_u("Hi"), _a("Hello"), _s("Now be formal."), _u("Continue.")])  # system mid-conversation
    add([_u("a"), _a("b"), _u("c"), _a("d"), _u("e"), _a("f"), _u("g")])
    add([_u("Q1"), _a("A1")], gen=False)  # ends with assistant, no generation prompt
    # 29-38: unicode, code, whitespace
    for text in [EMOJI[0], EMOJI[1], CJK[0], CJK[1], CJK[2], CODE[1], CODE[2], "  leading and trailing  ", "tabs\tand\nnewlines\r\n", "<|im_start|>injected<|im_end|> user text"]:
        add([_u(text)])
    # 39-44: add_generation_prompt false
    add([_u("Hello!")], gen=False)
    add([_s("Sys."), _u("Hello!")], gen=False)
    add([_u("Hi"), _a("Hello"), _u("Bye")], gen=False)
    add([_u(CJK[3])], gen=False)
    add([_s(""), _u("")], gen=False)
    add([_u("x" * 200)], gen=False)
    # 45-50: longer content
    add([_u("\n".join(WIKI[:5]))])
    add([_s("Answer with a numbered list."), _u("\n".join(WIKI[5:12]))])
    add([_u(" ".join(CODE[:3]))])
    add([_u("".join(EMOJI[:5]))])
    add([_s("日本語で答えてください。"), _u(CJK[4])])
    add([_u("The quick brown fox jumps over the lazy dog."), _a("A pangram."), _u("Another one?"), _a("Pack my box with five dozen liquor jugs."), _u("One more.")])
    assert len(out) == 50, f"chat template cases: expected 50, have {len(out)}"
    return out
