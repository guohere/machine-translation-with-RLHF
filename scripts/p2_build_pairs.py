#!/usr/bin/env python
"""Phase 2: turn scored samples into RSFT targets and DPO preference pairs (CPU, seconds).

RSFT: the best-scored sample per prompt (if it clears --rsft-min-score).
DPO:  (best, worst) per prompt, kept only if the score margin is at least
      --dpo-margin. Phase 1 showed that the reward disagrees with experts on about
      a third of same-source pairs, mostly when scores are close, so close pairs
      are dropped.

Example:
  python scripts/p2_build_pairs.py --candidates data/p2/candidates.jsonl --out data/p2
"""

from __future__ import annotations

import argparse
import os
import statistics
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from mtrlhf.io import read_jsonl, write_json, write_jsonl  # noqa: E402


def build(cands: list[dict], rsft_min_score: float, dpo_margin: float) -> tuple[list[dict], list[dict], dict]:
    rsft, dpo, margins = [], [], []
    for c in cands:
        s = [x for x in c["samples"] if x["hyp"].strip()]
        if not s:
            continue
        best = max(s, key=lambda x: x["score"])
        worst = min(s, key=lambda x: x["score"])
        if best["score"] >= rsft_min_score:
            rsft.append({"id": c["id"], "src": c["src"], "tgt": best["hyp"], "score": best["score"]})
        margin = best["score"] - worst["score"]
        if margin >= dpo_margin and best["hyp"] != worst["hyp"]:
            dpo.append({"id": c["id"], "src": c["src"], "chosen": best["hyp"], "rejected": worst["hyp"],
                        "margin": margin})
            margins.append(margin)
    stats = {"prompts": len(cands), "rsft": len(rsft), "dpo_pairs": len(dpo),
             "dpo_margin_mean": statistics.mean(margins) if margins else None,
             "rsft_min_score": rsft_min_score, "dpo_margin": dpo_margin}
    return rsft, dpo, stats


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--candidates", default="data/p2/candidates.jsonl")
    ap.add_argument("--out", default="data/p2")
    ap.add_argument("--rsft-min-score", type=float, default=-5.0,
                    help="MetricX-24-QE: keep best samples with error <= 5 (score >= -5)")
    ap.add_argument("--dpo-margin", type=float, default=1.0, help="minimum best-worst score difference")
    args = ap.parse_args()

    rsft, dpo, stats = build(read_jsonl(args.candidates), args.rsft_min_score, args.dpo_margin)
    write_jsonl(os.path.join(args.out, "rsft.jsonl"), rsft)
    write_jsonl(os.path.join(args.out, "dpo_pairs.jsonl"), dpo)
    write_json(os.path.join(args.out, "pairs_stats.json"), stats)
    print(stats)


if __name__ == "__main__":
    main()
