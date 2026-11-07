"""Evaluation sets: FLORES-200 (dev, devtest) and WMT24 en-de.

Each set is written as JSONL rows {id, src, ref, src_lang, tgt_lang}:
  flores_dev.en-de       checkpoint selection, over-optimization curves
  flores_devtest.en-de   primary test set
  flores_devtest.de-en   forgetting check (reverse direction)
  flores_devtest.en-zh   forgetting check (unrelated pair)
  wmt24.en-de            secondary test set (paragraph-level)
"""

from __future__ import annotations

import os
import tarfile
import urllib.request

FLORES_URL = "https://dl.fbaipublicfiles.com/nllb/flores200_dataset.tar.gz"
FLORES_CODES = {"en": "eng_Latn", "de": "deu_Latn", "zh": "zho_Hans"}
FLORES_SETS = [("dev", "en", "de"), ("devtest", "en", "de"), ("devtest", "de", "en"), ("devtest", "en", "zh")]


def _flores_from_tarball(dest_dir: str) -> dict[tuple[str, str], list[str]]:
    os.makedirs(dest_dir, exist_ok=True)
    path = os.path.join(dest_dir, "flores200_dataset.tar.gz")
    if not os.path.exists(path):
        urllib.request.urlretrieve(FLORES_URL, path)
    wanted = {f"{split}/{code}.{split}": (split, lang)
              for split in ("dev", "devtest") for lang, code in FLORES_CODES.items()}
    out = {}
    with tarfile.open(path, "r:gz") as tar:
        for member in tar:
            for suffix, key in wanted.items():
                if member.name.endswith(suffix):
                    out[key] = tar.extractfile(member).read().decode("utf-8").rstrip("\n").split("\n")
    if len(out) != len(wanted):
        raise RuntimeError(f"FLORES archive is missing files: {set(wanted.values()) - set(out)}")
    return out


def _flores_from_hub() -> dict[tuple[str, str], list[str]]:
    """Fallback: the gated openlanguagedata/flores_plus dataset (needs HF_TOKEN and accepted terms)."""
    from datasets import load_dataset

    out = {}
    for lang, code in FLORES_CODES.items():
        for split in ("dev", "devtest"):
            ds = load_dataset("openlanguagedata/flores_plus", code, split=split)
            out[(split, lang)] = list(ds["text"])
    return out


def load_flores(dest_dir: str) -> dict[tuple[str, str], list[str]]:
    try:
        return _flores_from_tarball(dest_dir)
    except Exception as e:
        print(f"FLORES tarball failed ({e}); trying openlanguagedata/flores_plus on the Hub")
        return _flores_from_hub()


def flores_sets(dest_dir: str) -> dict[str, list[dict]]:
    texts = load_flores(dest_dir)
    out = {}
    for split, s, t in FLORES_SETS:
        src, ref = texts[(split, s)], texts[(split, t)]
        assert len(src) == len(ref)
        name = f"flores_{split}.{s}-{t}"
        out[name] = [{"id": f"{name}-{i:04d}", "src": a, "ref": b, "src_lang": s, "tgt_lang": t}
                     for i, (a, b) in enumerate(zip(src, ref))]
    return out


def wmt24_set(mtme_path: str | None = None) -> dict[str, list[dict]]:
    from mtrlhf.meta_eval import wmt

    if mtme_path is None:
        mtme_path = wmt.download_mtme(["wmt24"], lps=["en-de"], keep_metric_scores=False)
    evs = wmt.load_eval_set("wmt24", "en-de", mtme_path, shipped=False)
    ref = evs.all_refs[evs.std_ref]
    rows = [{"id": f"wmt24.en-de-{i:04d}", "src": s, "ref": r, "src_lang": "en", "tgt_lang": "de"}
            for i, (s, r) in enumerate(zip(evs.src, ref)) if s and r]
    return {"wmt24.en-de": rows}
