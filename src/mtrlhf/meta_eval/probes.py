"""Blind-spot probes: controlled perturbations of good English->German translations.

For each probe item we keep the original (human reference) translation and
create one perturbed copy per category. A reward "detects" a perturbation when it
scores the original higher than the perturbed copy. Categories with low detection
rates are the predicted exploits for RL (checked in Phase 3).

Reference-based rewards score both versions against a *different* human
reference (`ref`), so they cannot win by string-matching the original.
"""

from __future__ import annotations

import random
import re
from typing import Callable

_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[\"'„»«“]?[A-ZÄÖÜ0-9])")
_NUM = re.compile(r"\d+(?:[.,]\d+)*")
_WORD = re.compile(r"\w+", re.UNICODE)
_KEIN = re.compile(r"\b([Kk])ein(e|en|em|er|es)?\b")


def _words(text: str) -> int:
    return len(text.split())


def omission(src: str, tgt: str, rng: random.Random, ctx: dict) -> str | None:
    """Drop one sentence, or one comma-separated clause (never the first)."""
    sents = _SENT_SPLIT.split(tgt.strip())
    if len(sents) >= 2:
        drop = rng.randrange(len(sents))
        out = " ".join(sents[:drop] + sents[drop + 1:])
    else:
        parts = re.split(r",\s*", tgt.strip())
        if len(parts) < 2:
            return None
        drop = rng.randrange(1, len(parts))
        out = ", ".join(parts[:drop] + parts[drop + 1:])
        end = tgt.strip()[-1:]
        if drop == len(parts) - 1 and end in ".!?" and out[-1:] not in ".!?":
            out = out.rstrip(",;: ") + end
    if _words(tgt) - _words(out) < 3 or _words(out) < 3:
        return None
    return out


def addition(src: str, tgt: str, rng: random.Random, ctx: dict) -> str | None:
    """Append an unrelated German sentence (unsupported by the source)."""
    pool = ctx.get("sentence_pool") or []
    choices = [s for s in pool if s not in tgt and s not in src]
    if not choices:
        return None
    base = tgt.rstrip()
    if base[-1:] not in ".!?\"“»":
        base += "."
    return base + " " + rng.choice(choices)


def change_number(s: str) -> str:
    """'12' -> '21', '2019' -> '2091', '7' -> '8', '9' -> '8'."""
    digits = [i for i, c in enumerate(s) if c.isdigit()]
    chars = list(s)
    if len(digits) >= 2 and s[digits[-1]] != s[digits[-2]]:
        a, b = digits[-2], digits[-1]
        chars[a], chars[b] = chars[b], chars[a]
    else:
        i = digits[-1]
        d = int(s[i])
        chars[i] = str(d + 1 if d < 9 else d - 1)
    return "".join(chars)


def number(src: str, tgt: str, rng: random.Random, ctx: dict) -> str | None:
    """Change a number that appears in both source and translation."""
    src_nums = set(_NUM.findall(src))
    cands = [m for m in _NUM.finditer(tgt) if m.group() in src_nums]
    if not cands:
        return None
    m = rng.choice(cands)
    return tgt[: m.start()] + change_number(m.group()) + tgt[m.end():]


def entities(src: str, tgt: str) -> list[str]:
    """Capitalized, non-sentence-initial source words that appear verbatim in the translation
    (mostly names of people, places and organizations)."""
    initial = {m.group(1) for m in re.finditer(r"(?:^|[.!?:]\s+)[\"'„“«»(]*(\w+)", src)}
    tgt_tokens = set(_WORD.findall(tgt))
    out: list[str] = []
    for tok in _WORD.findall(src):
        if (len(tok) >= 3 and tok[0].isupper() and tok.isalpha() and tok not in initial
                and tok in tgt_tokens and tok not in out):
            out.append(tok)
    return out


def named_entity(src: str, tgt: str, rng: random.Random, ctx: dict) -> str | None:
    """Replace a name with a different name taken from another probe item."""
    ents = entities(src, tgt)
    if not ents:
        return None
    ent = rng.choice(ents)
    pool = [e for e in ctx.get("entity_pool", []) if e != ent and e not in tgt and e not in src]
    if not pool:
        return None
    return re.sub(rf"\b{re.escape(ent)}\b", rng.choice(pool), tgt, count=1)


def negation(src: str, tgt: str, rng: random.Random, ctx: dict) -> str | None:
    """Remove 'nicht', or turn 'kein*' into 'ein*' (flips the meaning)."""
    m = re.search(r"\s+nicht\b", tgt)
    if m:
        return tgt[: m.start()] + tgt[m.end():]
    m = _KEIN.search(tgt)
    if m:
        first = "E" if m.group(1) == "K" else "e"
        return tgt[: m.start()] + first + "in" + (m.group(2) or "") + tgt[m.end():]
    return None


def untranslated(src: str, tgt: str, rng: random.Random, ctx: dict) -> str | None:
    """Return the English source unchanged (copying / off-target output)."""
    return src


def wrong_language(src: str, tgt: str, rng: random.Random, ctx: dict) -> str | None:
    """A correct translation, but into another language (from another WMT en-X pair)."""
    hit = (ctx.get("other_language") or {}).get(src)
    return hit[1] if hit else None


def word_order(src: str, tgt: str, rng: random.Random, ctx: dict) -> str | None:
    """Swap three pairs of adjacent words (fluency damage, meaning mostly kept)."""
    words = tgt.split()
    if len(words) < 6:
        return None
    positions = list(range(1, len(words) - 2))
    rng.shuffle(positions)
    out, swaps = words[:], 0
    for i in positions:
        if swaps == 3:
            break
        if out[i] != out[i + 1]:
            out[i], out[i + 1] = out[i + 1], out[i]
            swaps += 1
    result = " ".join(out)
    return result if result != tgt else None


def repetition(src: str, tgt: str, rng: random.Random, ctx: dict) -> str | None:
    """Duplicate a 3-5 word span right after itself."""
    words = tgt.split()
    if len(words) < 6:
        return None
    n = rng.randint(3, min(5, len(words) // 2))
    start = rng.randrange(0, len(words) - n + 1)
    return " ".join(words[: start + n] + words[start: start + n] + words[start + n:])


PERTURBATIONS: dict[str, Callable[[str, str, random.Random, dict], str | None]] = {
    "omission": omission,
    "addition": addition,
    "number": number,
    "named_entity": named_entity,
    "negation": negation,
    "untranslated": untranslated,
    "wrong_language": wrong_language,
    "word_order": word_order,
    "repetition": repetition,
}


def build_probe_rows(
    items: list[dict],
    n_items: int = 500,
    seed: int = 13,
    other_language: dict[str, tuple[str, str]] | None = None,
    min_words: int = 8,
    max_words: int = 60,
) -> list[dict]:
    """Build probe rows from items with keys src, ref (good translation) and alt_ref.

    Returns rows {key, item, category, src, hyp, ref, lang?}. The "original" row
    holds the unperturbed translation.
    """
    seen: set[str] = set()
    eligible = []
    for it in items:
        if not it.get("alt_ref") or not it.get("ref") or it["src"] in seen:
            continue
        if min_words <= _words(it["src"]) <= max_words:
            seen.add(it["src"])
            eligible.append(it)
    rng = random.Random(seed)
    rng.shuffle(eligible)
    chosen = eligible[:n_items]

    sentence_pool = []
    entity_pool: list[str] = []
    for it in eligible:
        for s in _SENT_SPLIT.split(it["ref"].strip()):
            if 5 <= _words(s) <= 15 and s[-1:] == ".":
                sentence_pool.append(s)
        for e in entities(it["src"], it["ref"]):
            if e not in entity_pool:
                entity_pool.append(e)
    ctx = {"sentence_pool": sentence_pool, "entity_pool": entity_pool,
           "other_language": other_language or {}}

    rows = []
    for k, it in enumerate(chosen):
        item_id = f"probe{k:04d}"
        rows.append({"key": f"{item_id}|original", "item": item_id, "category": "original",
                     "src": it["src"], "hyp": it["ref"], "ref": it["alt_ref"]})
        for cat, fn in PERTURBATIONS.items():
            out = fn(it["src"], it["ref"], random.Random(f"{seed}|{k}|{cat}"), ctx)
            if not out or out.strip() == it["ref"].strip():
                continue
            row = {"key": f"{item_id}|{cat}", "item": item_id, "category": cat,
                   "src": it["src"], "hyp": out, "ref": it["alt_ref"]}
            if cat == "wrong_language":
                row["lang"] = ctx["other_language"][it["src"]][0]
            rows.append(row)
    return rows
