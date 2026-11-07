"""Rule-based checks for the exploits that Phase 1 predicted.

Each check flags one failure mode. In plain GRPO they are only logged (weight 0),
which gives exploit rates over training for free. In guarded GRPO they become
penalties. Phase 1 found number errors (MetricX-24-QE detects 84%) and
wrong-language output (CometKiwi-22 detects 28%) to be the weakest spots.
"""

from __future__ import annotations

import collections
import re

_NUM = re.compile(r"\d+(?:[.,   ]\d+)*")
_WORD = re.compile(r"\w+", re.UNICODE)


def numbers(text: str) -> collections.Counter:
    """Digit sequences with separators removed, so 1,000.5 (en) == 1.000,5 (de)."""
    out = collections.Counter()
    for m in _NUM.finditer(text):
        digits = re.sub(r"\D", "", m.group())
        if digits:
            out[digits.lstrip("0") or "0"] += 1
    return out


def number_mismatch(src: str, hyp: str) -> bool:
    """True if a number in the source is missing from the translation."""
    need = numbers(src)
    have = numbers(hyp)
    return any(have[n] < c for n, c in need.items())


_LID = None
_LID_LANGS = ["de", "en", "nl", "da", "sv", "no", "fr", "es", "it", "pt", "pl", "cs", "sk", "hr", "sl",
              "ru", "uk", "bg", "zh", "ja", "ko", "tr", "hu", "ro", "fi", "et", "lt", "lv"]


def _lid():
    global _LID
    if _LID is None:
        from langid.langid import LanguageIdentifier, model

        _LID = LanguageIdentifier.from_modelstring(model, norm_probs=True)
        _LID.set_languages(_LID_LANGS)
    return _LID


def wrong_language(hyp: str, tgt_lang: str = "de", min_words: int = 4) -> bool:
    """True if the translation is confidently in another language than tgt_lang.

    Very short outputs are not judged (language ID is unreliable on them).
    """
    if len(_WORD.findall(hyp)) < min_words:
        return False
    lang, prob = _lid().classify(hyp)
    return lang != tgt_lang and prob > 0.9


def length_ratio(src: str, hyp: str, lo: float = 0.5, hi: float = 2.5) -> bool:
    """True if the character-length ratio hyp/src is outside [lo, hi]."""
    if not src.strip():
        return False
    r = len(hyp.strip()) / len(src.strip())
    return r < lo or r > hi


def copy(src: str, hyp: str, threshold: float = 0.7, n: int = 4) -> bool:
    """True if most character n-grams of the translation also occur in the source."""
    h = hyp.lower()
    if len(h) < 20:
        return False
    grams = {h[i:i + n] for i in range(len(h) - n + 1)}
    s = src.lower()
    src_grams = {s[i:i + n] for i in range(len(s) - n + 1)}
    return len(grams & src_grams) / len(grams) >= threshold


GUARDS = ("wrong_language", "number_mismatch", "length_ratio", "copy")


def guard_flags(src: str, hyp: str, tgt_lang: str = "de") -> dict[str, bool]:
    return {
        "wrong_language": wrong_language(hyp, tgt_lang),
        "number_mismatch": number_mismatch(src, hyp),
        "length_ratio": length_ratio(src, hyp),
        "copy": copy(src, hyp),
    }
