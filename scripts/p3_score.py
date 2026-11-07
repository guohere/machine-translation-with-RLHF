#!/usr/bin/env python
"""Phase 3: fill in the scores the analysis needs (GPU; skips what is already scored).

  * every translation file of a model gets scores_<metric> for each --metrics entry
    (the Phase 2 quick evaluation usually did this already);
  * the stored best-of-N candidates on FLORES dev also get the held-out score, so
    best-of-k can be plotted on the held-out metric.

Example:
  python scripts/p3_score.py --model qwen3-1.7b
"""

from __future__ import annotations

import argparse
import glob
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from mtrlhf.io import read_jsonl, write_jsonl  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--results", default="results/p2")
    ap.add_argument("--metrics", nargs="+", default=["chrf", "metricx24_qe", "comet22"])
    ap.add_argument("--held-out", default="comet22")
    ap.add_argument("--bon-testsets", nargs="+", default=["flores_dev.en-de"])
    args = ap.parse_args()

    from mtrlhf.rewards import get_reward, reward_info

    files = sorted(glob.glob(os.path.join(args.results, args.model, "outputs", "*", "*.jsonl")))
    for metric in args.metrics:
        todo = [f for f in files if (rows := read_jsonl(f)) and f"scores_{metric}" not in rows[0]]
        if not todo:
            continue
        info, reward = reward_info(metric), get_reward(metric)
        for f in todo:
            rows = read_jsonl(f)
            s = reward.score([r["src"] for r in rows], [r["hyp"] for r in rows],
                             [r["ref"] for r in rows] if info["needs_reference"] else None)
            for r, v in zip(rows, s):
                r[f"scores_{metric}"] = v
            write_jsonl(f, rows)
            print(f"{metric}: scored {f}", flush=True)
        reward.close()

    bon = [os.path.join(args.results, args.model, "outputs", "sft+bo8", f"{ts}.jsonl") for ts in args.bon_testsets]
    bon = [f for f in bon if os.path.exists(f) and any(args.held_out not in c for r in read_jsonl(f)
                                                       for c in r.get("candidates", []))]
    if bon:
        info, reward = reward_info(args.held_out), get_reward(args.held_out)
        for f in bon:
            rows = read_jsonl(f)
            flat = [(r, c) for r in rows for c in r.get("candidates", [])]
            s = reward.score([r["src"] for r, _ in flat], [c["hyp"] for _, c in flat],
                             [r["ref"] for r, _ in flat] if info["needs_reference"] else None)
            for (_, c), v in zip(flat, s):
                c[args.held_out] = v
            write_jsonl(f, rows)
            print(f"{args.held_out}: scored {len(flat)} best-of-N candidates in {f}", flush=True)
        reward.close()
    print("scores complete")


if __name__ == "__main__":
    main()
