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


def build_messages(src: str, src_lang: str = "en", tgt_lang: str = "de", variant: int = 0) -> list[dict]:
    """The translation prompt. variant 0 is used everywhere at inference; SFT also trains on
    variants 1-2 (instruction after the text, a shorter wording) so the model has to read it."""
    s, t = LANG_NAMES[src_lang], LANG_NAMES[tgt_lang]
    if variant == 1:
        content = f"{src}\n\nTranslate the {s} text above into {t}. Reply with the {t} translation only."
    elif variant == 2:
        content = f"{s} → {t}:\n{src}"
    else:
        content = f"Translate the following {s} text into {t}. Reply with the {t} translation only.\n\n{src}"
    return [{"role": "user", "content": content}]


def prompt_variant(row_id: str, share_other: float = 0.3) -> int:
    """Deterministic per example: variant 0 for most rows, 1 or 2 for `share_other` of them."""
    import zlib

    h = zlib.crc32(row_id.encode()) % 1000 / 1000
    return 0 if h >= share_other else (1 if h < share_other / 2 else 2)


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
