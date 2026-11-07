#!/usr/bin/env python
"""Phase 2: sample N translations per prompt from a system and score them with R_train.

This file is the matched reward budget shared by RSFT and DPO (5,000 prompts x 8
samples = 40,000 reward calls). Each output row:
  {id, src, ref, samples: [{hyp, raw, score, flags}]}
Resumable: prompts already in OUT are skipped.

Example:
  python scripts/p2_sample.py --model qwen3-1.7b --system sft --n 8 --reward metricx24_qe
  (writes data/p2/qwen3-1.7b/candidates.jsonl)
"""

from __future__ import annotations

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from mtrlhf.data.prompts import extract_translation  # noqa: E402
from mtrlhf.io import append_jsonl, read_jsonl  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--system", default="sft")
    ap.add_argument("--prompts", default="data/p2/prompts.jsonl")
    ap.add_argument("--model", default=None, help="policy model: qwen3-1.7b | gemma3-1b (sets the model id and per-model folders)")
    ap.add_argument("--out", default=None, help="default: data/p2/<model>/candidates.jsonl")
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top-p", type=float, default=1.0)
    ap.add_argument("--reward", default="metricx24_qe")
    ap.add_argument("--chunk", type=int, default=500, help="prompts per generate+score round")
    ap.add_argument("--batch-size", type=int, default=64, help="sequences per generation batch")
    ap.add_argument("--max-new-tokens", type=int, default=256)
    ap.add_argument("--base-model", default=None)
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--hub-repo", default=None,
                    help='private HF repo mirroring the adapters ("auto" = <user>/mtrlhf-en-de-adapters, "none" = off; default: $MTRLHF_HUB_REPO)')
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    from mtrlhf.models import CheckpointStore, Translator, default_device, model_paths
    from mtrlhf.rewards import get_reward, reward_info
    from mtrlhf.rewards.guards import guard_flags

    paths = model_paths(args.model)
    args.out = args.out or os.path.join(paths["model_data"], "candidates.jsonl")
    prompts = read_jsonl(args.prompts)
    if args.limit:
        prompts = prompts[: args.limit]
    done = {r["id"] for r in read_jsonl(args.out)}
    todo = [p for p in prompts if p["id"] not in done]
    print(f"{len(todo)} prompts to sample ({len(done)} already done), n={args.n}")
    if not todo:
        return

    store = CheckpointStore.from_env(args.ckpt or paths["ckpt"], args.hub_repo, paths["hub_prefix"])
    tr = Translator.for_system(args.system, store, args.base_model or paths["model_id"], device=default_device(0),
                               batch_size=args.batch_size, max_new_tokens=args.max_new_tokens)
    info = reward_info(args.reward)
    kw = {"device": default_device(1)} if info["kind"] == "metricx" else {}
    reward = get_reward(args.reward, **kw)

    t0 = time.time()
    for start in range(0, len(todo), args.chunk):
        chunk = todo[start: start + args.chunk]
        raw = tr.generate([p["src"] for p in chunk], n=args.n, do_sample=True, temperature=args.temperature,
                          top_p=args.top_p, seed=args.seed + start)
        srcs, hyps, refs = [], [], []
        for p, outs in zip(chunk, raw):
            for o in outs:
                srcs.append(p["src"])
                hyps.append(extract_translation(o, p["src"])[0])
                refs.append(p.get("ref"))
        scores = reward.score(srcs, hyps, refs if info["needs_reference"] else None)
        rows, k = [], 0
        for p, outs in zip(chunk, raw):
            samples = []
            for o in outs:
                samples.append({"hyp": hyps[k], "raw": o, "score": scores[k], "flags": guard_flags(p["src"], hyps[k])})
                k += 1
            rows.append({"id": p["id"], "src": p["src"], "ref": p.get("ref"), "samples": samples})
        append_jsonl(args.out, rows)
        print(f"  {start + len(chunk)}/{len(todo)} prompts ({(start + len(chunk)) / (time.time() - t0):.2f} prompts/s)",
              flush=True)
    reward.close()
    tr.close()


if __name__ == "__main__":
    main()
