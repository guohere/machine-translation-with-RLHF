#!/usr/bin/env python
"""Phase 2: choose the sampling setting for all sampling-based methods (~30-40 GPU-minutes).

Samples N translations for held-out sentences under a grid of temperature x top-p,
scores them with the training reward, and measures per setting:
  broken %          samples with MetricX error > 10 (wasted budget, trivial DPO pairs)
  spread            mean within-group std of the reward (the signal DPO / GRPO learn from)
  identical groups  % of sources whose N samples are all the same (no signal at all)
  best / mean       best-of-N vs. average quality (headroom)
Then applies the rule from mtrlhf/sampling.py and, with --write, saves the choice to
results/p2/sampling.json, which candidate sampling, best-of-N and GRPO all read.

Example:
  python scripts/p2_sampling_pilot.py --model qwen3-1.7b --write results/p2/sampling.json
"""

from __future__ import annotations

import argparse
import gzip
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from mtrlhf.data.prompts import extract_translation  # noqa: E402
from mtrlhf.io import read_jsonl, write_json  # noqa: E402
from mtrlhf.sampling import BROKEN_ERROR, MAX_BROKEN_PCT, choose, group_stats  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default=None, help="policy model: qwen3-1.7b | gemma3-1b")
    ap.add_argument("--system", default="sft")
    ap.add_argument("--prompts", default="data/p2/sft_val.jsonl", help="held-out sentences (not test data)")
    ap.add_argument("--limit", type=int, default=200)
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--temperatures", type=float, nargs="+", default=[0.6, 0.8, 1.0])
    ap.add_argument("--top-ps", type=float, nargs="+", default=[0.95, 1.0])
    ap.add_argument("--reward", default="metricx24_qe")
    ap.add_argument("--max-broken-pct", type=float, default=MAX_BROKEN_PCT)
    ap.add_argument("--broken-error", type=float, default=BROKEN_ERROR)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--max-new-tokens", type=int, default=256)
    ap.add_argument("--out", default=None, help="report folder (default: results/p2/<model>/sampling_pilot)")
    ap.add_argument("--write", default=None, help="write the chosen setting to this file (e.g. results/p2/sampling.json)")
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--hub-repo", default=None)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    from mtrlhf.models import CheckpointStore, Translator, default_device, model_paths
    from mtrlhf.rewards import get_reward, reward_info
    from mtrlhf.rewards.guards import guard_flags

    paths = model_paths(args.model)
    out_dir = args.out or os.path.join(os.path.dirname(paths["outputs"]), "sampling_pilot")
    prompts = read_jsonl(args.prompts)[: args.limit]
    if not prompts:
        sys.exit(f"No prompts in {args.prompts}: run p2_prepare_data.py first")
    store = CheckpointStore.from_env(args.ckpt or paths["ckpt"], args.hub_repo, paths["hub_prefix"])
    tr = Translator.for_system(args.system, store, paths["model_id"], device=default_device(0),
                               batch_size=args.batch_size, max_new_tokens=args.max_new_tokens)
    info = reward_info(args.reward)
    reward = get_reward(args.reward, **({"device": default_device(1)} if info["kind"] == "metricx" else {}))

    results, all_samples = [], []
    for t in args.temperatures:
        for p in args.top_ps:
            t0 = time.time()
            raw = tr.generate([r["src"] for r in prompts], n=args.n, do_sample=True, temperature=t, top_p=p,
                              seed=args.seed)
            hyps = [[extract_translation(o, r["src"])[0] for o in outs] for r, outs in zip(prompts, raw)]
            flat_src = [r["src"] for r, hs in zip(prompts, hyps) for _ in hs]
            flat_ref = [r.get("ref") for r, hs in zip(prompts, hyps) for _ in hs]
            flat = reward.score(flat_src, [h for hs in hyps for h in hs],
                                flat_ref if info["needs_reference"] else None)
            groups, k = [], 0
            for r, hs in zip(prompts, hyps):
                groups.append([(h, flat[k + j]) for j, h in enumerate(hs)])
                k += len(hs)
            flags = [any(guard_flags(src, h).values()) for src, h in zip(flat_src, [h for hs in hyps for h in hs])]
            stats = {"temperature": t, "top_p": p, **group_stats(groups, args.broken_error),
                     "guard_flag_pct": 100 * sum(flags) / max(len(flags), 1), "seconds": round(time.time() - t0)}
            results.append(stats)
            all_samples.extend({"temperature": t, "top_p": p, "id": r["id"], "src": r["src"],
                                "samples": [{"hyp": h, "score": s} for h, s in g]} for r, g in zip(prompts, groups))
            print(f"T={t:<4} top_p={p:<5} broken {stats['broken_pct']:5.1f}%  spread {stats['spread']:5.2f}  "
                  f"identical {stats['identical_groups_pct']:5.1f}%  mean {stats['mean_score']:6.2f}  "
                  f"best {stats['mean_best']:6.2f}  ({stats['seconds']}s)", flush=True)
    reward.close()
    tr.close()

    best, why = choose(results, args.max_broken_pct)
    lines = ["# Sampling pilot", "",
             f"Model `{paths['model_id']}` (system `{args.system}`), {len(prompts)} held-out sentences "
             f"(`{args.prompts}`), {args.n} samples each, scored with `{args.reward}`. "
             f"Broken = MetricX error > {args.broken_error:g}. Spread = mean within-group std of the reward.", "",
             "| Temperature | top-p | Broken % | Guard flags % | Spread | Identical groups % | Unique / group "
             "| Mean | Best-of-N |",
             "|---|---|---|---|---|---|---|---|---|"]
    for r in results:
        mark = " **← chosen**" if r is best else ""
        lines.append(f"| {r['temperature']} | {r['top_p']} | {r['broken_pct']:.1f} | {r['guard_flag_pct']:.1f} | "
                     f"{r['spread']:.2f} | {r['identical_groups_pct']:.1f} | {r['unique_per_group']:.1f} | "
                     f"{r['mean_score']:.2f} | {r['mean_best']:.2f}{mark} |")
    lines += ["", f"**Chosen: temperature {best['temperature']}, top-p {best['top_p']}** — {why}.", ""]
    md = "\n".join(lines)
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "report.md"), "w", encoding="utf-8") as fh:
        fh.write(md)
    write_json(os.path.join(out_dir, "results.json"), {"results": results, "chosen": best, "why": why,
                                                       "args": vars(args)})
    with gzip.open(os.path.join(out_dir, "samples.jsonl.gz"), "wt", encoding="utf-8") as fh:
        for row in all_samples:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    print("\n" + md)
    if args.write:
        write_json(args.write, {"temperature": best["temperature"], "top_p": best["top_p"], "why": why,
                                "chosen_on": {"model": args.model, "system": args.system, "prompts": args.prompts,
                                              "n_sentences": len(prompts), "n": args.n, "reward": args.reward},
                                "report": os.path.join(out_dir, "report.md")})
        print(f"wrote {args.write}")


if __name__ == "__main__":
    main()
