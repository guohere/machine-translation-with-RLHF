"""English-German training data from News Commentary, filtered and split.

Splits (disjoint, deterministic for a seed):
  sft_train / sft_val  pairs for supervised fine-tuning
  prompts              English sentences for RSFT, DPO and GRPO (the German side is
                       kept only for analysis; the RL reward never sees it)
"""

from __future__ import annotations

import gzip
import os
import random
import re
import urllib.request

NEWS_COMMENTARY_URLS = [
    "https://data.statmt.org/news-commentary/v18.1/training/news-commentary-v18.de-en.tsv.gz",
    "https://data.statmt.org/news-commentary/v17/training/news-commentary-v17.de-en.tsv.gz",
    "https://data.statmt.org/news-commentary/v16/training/news-commentary-v16.de-en.tsv.gz",
]

_DE_WORDS = {"der", "die", "das", "und", "ist", "nicht", "mit", "sich", "auf", "für", "ein", "eine", "dass"}
_EN_WORDS = {"the", "and", "is", "of", "to", "that", "with", "for", "not", "this", "are", "be"}


def download_news_commentary(dest_dir: str, urls: list[str] | None = None) -> str:
    os.makedirs(dest_dir, exist_ok=True)
    errors = []
    for url in urls or NEWS_COMMENTARY_URLS:
        path = os.path.join(dest_dir, os.path.basename(url))
        if os.path.exists(path) and os.path.getsize(path) > 0:
            return path
        try:
            urllib.request.urlretrieve(url, path)
            return path
        except Exception as e:  # try the next release
            errors.append(f"{url}: {e}")
            if os.path.exists(path):
                os.remove(path)
    raise RuntimeError("Could not download News Commentary:\n" + "\n".join(errors))


def _score(text: str, words: set[str]) -> int:
    return sum(w in words for w in re.findall(r"[a-zäöüß]+", text.lower()))


def read_tsv_pairs(path: str) -> list[tuple[str, str]]:
    """Read a two-column TSV and return (english, german) pairs, whatever the column order."""
    opener = gzip.open if path.endswith(".gz") else open
    rows = []
    with opener(path, "rt", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            parts = line.rstrip("\n").split("\t")
            if len(parts) >= 2:
                rows.append((parts[0].strip(), parts[1].strip()))
    sample = [r for r in rows[:2000] if r[0] and r[1]]
    col0_de = sum(_score(a, _DE_WORDS) - _score(a, _EN_WORDS) for a, _ in sample)
    col1_de = sum(_score(b, _DE_WORDS) - _score(b, _EN_WORDS) for _, b in sample)
    if col0_de > col1_de:  # German first
        rows = [(b, a) for a, b in rows]
    return rows


def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


def keep_pair(en: str, de: str, min_words: int = 5, max_words: int = 60,
              min_ratio: float = 0.7, max_ratio: float = 2.0) -> bool:
    """Cheap filters: both sides present, sentence-sized, plausible length ratio, not a copy."""
    if not en or not de or en == de:
        return False
    n = len(en.split())
    if n < min_words or n > max_words:
        return False
    ratio = len(de) / max(len(en), 1)
    if ratio < min_ratio or ratio > max_ratio:
        return False
    if "http" in en or "http" in de:
        return False
    return True


def language_ok(en: str, de: str) -> bool:
    """Language-ID check on both sides (skipped if langid is not installed)."""
    try:
        from mtrlhf.rewards.guards import _lid
    except ImportError:
        return True
    try:
        lid = _lid()
    except ImportError:
        return True
    return lid.classify(en)[0] == "en" and lid.classify(de)[0] == "de"


def make_splits(pairs: list[tuple[str, str]], sizes: dict[str, int], exclude: set[str] | None = None,
                seed: int = 13, check_language: bool = True) -> tuple[dict[str, list[dict]], dict]:
    """Shuffle, filter lazily (in shuffled order) and cut disjoint splits of the given sizes."""
    exclude = exclude or set()
    order = list(range(len(pairs)))
    random.Random(seed).shuffle(order)
    want = sum(sizes.values())
    kept, seen = [], set()
    stats = {"raw": len(pairs), "dropped_filters": 0, "dropped_duplicate": 0, "dropped_test_overlap": 0,
             "dropped_language": 0}
    for i in order:
        if len(kept) >= want:
            break
        en, de = pairs[i]
        if not keep_pair(en, de):
            stats["dropped_filters"] += 1
            continue
        key = normalize(en)
        if key in seen:
            stats["dropped_duplicate"] += 1
            continue
        if key in exclude:
            stats["dropped_test_overlap"] += 1
            continue
        if check_language and not language_ok(en, de):
            stats["dropped_language"] += 1
            continue
        seen.add(key)
        kept.append({"src": en, "ref": de})
    if len(kept) < want:
        raise RuntimeError(f"Only {len(kept)} usable pairs, {want} requested")
    out, start = {}, 0
    for name, n in sizes.items():
        out[name] = [dict(r, id=f"{name}-{j:05d}") for j, r in enumerate(kept[start:start + n])]
        start += n
    stats.update({f"n_{k}": len(v) for k, v in out.items()})
    return out, stats
