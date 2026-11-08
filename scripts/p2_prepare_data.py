#!/usr/bin/env python
"""Phase 2, step 1: training data and test sets (CPU, ~5 min).

Writes to OUT (default data/p2):
  sft_train.jsonl, sft_val.jsonl   News Commentary en-de pairs for SFT
  prompts.jsonl                    English prompts for RSFT / DPO / GRPO (disjoint from SFT)
  testsets/<name>.jsonl            FLORES-200 dev/devtest (en-de, de-en, en-zh) and WMT24 en-de
  data_stats.json

Training sentences that also occur in any test set (or in the Phase 1 WMT data)
are removed.

Example:
  python scripts/p2_prepare_data.py --out data/p2
"""

from __future__ import annotations

import argparse
import os
import random
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from mtrlhf.data import parallel, testsets  # noqa: E402
from mtrlhf.io import write_json, write_jsonl  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="data/p2")
    ap.add_argument("--sft-size", type=int, default=20000)
    ap.add_argument("--val-size", type=int, default=500)
    ap.add_argument("--prompt-size", type=int, default=5000)
    ap.add_argument("--reverse-size", type=int, default=10000,
                    help="extra German->English SFT pairs (disjoint from the rest); 0 = English->German only")
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--parallel-tsv", default=None, help="use this en-de TSV instead of downloading News Commentary")
    ap.add_argument("--flores-dir", default=None, help="where to cache the FLORES archive (default OUT/raw)")
    ap.add_argument("--mtme-path", default=None, help="mt-metrics-eval data dir with wmt24 (default: download)")
    ap.add_argument("--skip-wmt24", action="store_true")
    ap.add_argument("--no-langid", action="store_true", help="skip the language-ID filter")
    args = ap.parse_args()

    raw = os.path.join(args.out, "raw")
    sets = testsets.flores_sets(args.flores_dir or raw)
    if not args.skip_wmt24:
        try:
            sets.update(testsets.wmt24_set(args.mtme_path))
        except Exception as e:
            print(f"!! WMT24 not available ({e}); continuing without it", file=sys.stderr)
    for name, rows in sets.items():
        write_jsonl(os.path.join(args.out, "testsets", f"{name}.jsonl"), rows)
        print(f"test set {name}: {len(rows)} segments")

    exclude = {parallel.normalize(r["src"]) for rows in sets.values() for r in rows}
    exclude |= {parallel.normalize(r["ref"]) for rows in sets.values() for r in rows}
    try:  # also keep the Phase 1 WMT sources out of training
        from mtrlhf.io import read_jsonl

        exclude |= {parallel.normalize(r["src"]) for r in read_jsonl("runs/p1/segments.jsonl")}
    except Exception:
        pass

    tsv = args.parallel_tsv or parallel.download_news_commentary(raw)
    pairs = parallel.read_tsv_pairs(tsv)
    splits, stats = parallel.make_splits(
        pairs, {"sft_train": args.sft_size, "sft_val": args.val_size, "prompts": args.prompt_size,
                **({"sft_reverse": args.reverse_size} if args.reverse_size else {})},
        exclude=exclude, seed=args.seed, check_language=not args.no_langid)
    # SFT trains both directions: English->German (the task) plus German->English, so the model
    # keeps reading which direction the instruction asks for (SFT on one direction only made
    # Qwen3 answer every non-English input in its own language; results/p2/archive/).
    fwd = [dict(r, src_lang="en", tgt_lang="de") for r in splits["sft_train"]]
    rev = [{"id": r["id"], "src": r["ref"], "ref": r["src"], "src_lang": "de", "tgt_lang": "en"}
           for r in splits.pop("sft_reverse", [])]
    both = fwd + rev
    random.Random(args.seed).shuffle(both)
    splits["sft_train"] = both
    for name, rows in splits.items():
        write_jsonl(os.path.join(args.out, f"{name}.jsonl"), rows)
    stats.update({"n_sft_train": len(both), "n_sft_en_de": len(fwd), "n_sft_de_en": len(rev)})
    stats.pop("n_sft_reverse", None)
    stats.update({"source": os.path.basename(tsv), "seed": args.seed,
                  "testsets": {k: len(v) for k, v in sets.items()}})
    write_json(os.path.join(args.out, "data_stats.json"), stats)
    print("training data:", {k: v for k, v in stats.items() if k.startswith(("n_", "dropped", "raw"))})


if __name__ == "__main__":
    main()
