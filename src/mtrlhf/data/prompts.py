"""The translation prompt shared by every Qwen-based system, and reply cleanup."""

from __future__ import annotations

import re

LANG_NAMES = {"en": "English", "de": "German", "zh": "Chinese", "fr": "French"}

_QUOTES_OPEN = "\"“„'«»"
_QUOTES_CLOSE = "\"”“'»«"
_THINK = re.compile(r"<think>.*?</think>\s*|</?think>", re.DOTALL)
_LABEL = re.compile(
    r"^\s*(?:\*\*)?(?:german|deutsch|english|englisch|chinese|translation|übersetzung)"
    r"(?:\s+translation)?\s*(?:\*\*)?\s*[:：]\s*(?:\*\*)?\s*",
    re.IGNORECASE,
)


def build_messages(src: str, src_lang: str = "en", tgt_lang: str = "de") -> list[dict]:
    s, t = LANG_NAMES[src_lang], LANG_NAMES[tgt_lang]
    return [{"role": "user",
             "content": f"Translate the following {s} text into {t}. "
                        f"Reply with the {t} translation only.\n\n{src}"}]


def extract_translation(reply: str, src: str | None = None) -> tuple[str, bool]:
    """Strip labels, wrapping quotes and trailing notes from a model reply.

    Returns (translation, changed). The share of changed replies is logged,
    because heavy cleanup can hide problems.
    """
    text = reply.strip()
    out = _THINK.sub("", text).strip()  # Qwen3 reasoning blocks, should non-thinking mode leak
    out = _LABEL.sub("", out)
    # A single-paragraph source should give a single-paragraph translation: drop
    # anything after a blank line (typically "Note: ..." explanations).
    if "\n\n" in out and (src is None or "\n\n" not in src):
        out = out.split("\n\n")[0].strip()
    src_quoted = bool(src) and src.strip()[:1] in _QUOTES_OPEN
    if len(out) >= 2 and out[0] in _QUOTES_OPEN and out[-1] in _QUOTES_CLOSE and not src_quoted:
        out = out[1:-1].strip()
    return out, out != text
