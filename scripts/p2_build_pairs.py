#!/usr/bin/env python
"""Phase 2: turn scored samples into RSFT targets and DPO preference pairs (CPU, seconds).

RSFT: the best-scored sample per prompt (if it clears --rsft-min-score).
DPO:  (best, worst) per prompt, kept only if the score margin is at least
      --dpo-margin. Phase 1 showed that the reward disagrees with experts on about
      a third of same-source pairs, mostly when scores are close, so close pairs
      are dropped.

Example:
  python scripts/p2_build_pairs.py --model qwen3-1.7b     # reads/writes data/p2/qwen3-1.7b/
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
    ap.add_argument("--model", default=None, help="policy model: qwen3-1.7b | gemma3-1b (sets the model id and per-model folders)")
    ap.add_argument("--candidates", default=None, help="default: data/p2/<model>/candidates.jsonl")
    ap.add_argument("--out", default=None, help="default: data/p2/<model>")
    ap.add_argument("--rsft-min-score", type=float, default=-5.0,
                    help="MetricX-24-QE: keep best samples with error <= 5 (score >= -5)")
    ap.add_argument("--dpo-margin", type=float, default=1.0, help="minimum best-worst score difference")
    args = ap.parse_args()
    from mtrlhf.models import model_paths

    model_data = model_paths(args.model)["model_data"]
    args.candidates = args.candidates or os.path.join(model_data, "candidates.jsonl")
    args.out = args.out or model_data

    cands = read_jsonl(args.candidates)
    if not cands:
        sys.exit(f"No sampled candidates in {args.candidates}: run p2_sample.py first (did it fail?)")
    rsft, dpo, stats = build(cands, args.rsft_min_score, args.dpo_margin)
    from mtrlhf.io import read_json
    from mtrlhf.sampling import group_stats

    meta_path = os.path.splitext(args.candidates)[0] + ".meta.json"
    stats["sampling"] = read_json(meta_path) if os.path.exists(meta_path) else None
    stats["candidates"] = group_stats([[(x["hyp"], x["score"]) for x in c["samples"]] for c in cands])
    write_jsonl(os.path.join(args.out, "rsft.jsonl"), rsft)
    write_jsonl(os.path.join(args.out, "dpo_pairs.jsonl"), dpo)
    write_json(os.path.join(args.out, "pairs_stats.json"), stats)
    print(stats)


if __name__ == "__main__":
    main()
