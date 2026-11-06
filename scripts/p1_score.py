#!/usr/bin/env python
"""Phase 1, step 2: score segments and probes with each reward (GPU for neural rewards).

Identical (source, translation, reference) inputs are scored once. Scores are
appended in chunks to RUN_DIR/scores/<reward>.jsonl, so an interrupted run
resumes where it stopped. Timing and memory go to <reward>.meta.json.

Examples:
  python scripts/p1_score.py --run runs/p1 --rewards chrf cometkiwi22 --limit 32   # smoke test
  python scripts/p1_score.py --run runs/p1 --rewards cometkiwi22 comet22 metricx24_qe metricx24_ref chrf bleu
"""

from __future__ import annotations

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from mtrlhf.io import append_jsonl, read_json, read_jsonl, write_json  # noqa: E402
from mtrlhf.rewards import get_reward, reward_info  # noqa: E402


def device_info() -> dict:
    try:
        import torch

        if torch.cuda.is_available():
            return {"device": "cuda", "gpu": torch.cuda.get_device_name(0),
                    "n_gpus": torch.cuda.device_count()}
    except ImportError:
        pass
    return {"device": "cpu"}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", default="runs/p1")
    ap.add_argument("--rewards", nargs="+", required=True)
    ap.add_argument("--limit", type=int, default=0,
                    help="score only the first N rows of each table (smoke test; writes to scores_smoke/)")
    ap.add_argument("--chunk-size", type=int, default=4000, help="unique inputs per scoring call")
    ap.add_argument("--batch-size", type=int, default=None, help="COMET batch size")
    ap.add_argument("--max-tokens-per-batch", type=int, default=None, help="MetricX token budget per batch")
    ap.add_argument("--dtype", default=None, help="MetricX dtype: auto | float32 | bfloat16")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    segments = read_jsonl(os.path.join(args.run, "segments.jsonl"))
    probes = read_jsonl(os.path.join(args.run, "probes.jsonl"))
    if not segments:
        sys.exit(f"No segments.jsonl in {args.run}; run p1_build_inputs.py first.")
    if args.limit:
        segments, probes = segments[: args.limit], probes[: args.limit]
    rows = segments + probes
    out_dir = os.path.join(args.run, "scores_smoke" if args.limit else "scores")
    os.makedirs(out_dir, exist_ok=True)

    overrides = {k: v for k, v in {"batch_size": args.batch_size,
                                   "max_tokens_per_batch": args.max_tokens_per_batch,
                                   "dtype": args.dtype}.items() if v is not None}

    for name in args.rewards:
        info = reward_info(name)
        path = os.path.join(out_dir, f"{name}.jsonl")
        if args.overwrite and os.path.exists(path):
            os.remove(path)
        done = {r["key"] for r in read_jsonl(path)}

        def unit(r):
            return (r["src"], r["hyp"], r["ref"] if info["needs_reference"] else None)

        todo = [r for r in rows if r["key"] not in done
                and (not info["needs_reference"] or r.get("ref"))]
        uniq: dict[tuple, list[str]] = {}
        for r in todo:
            uniq.setdefault(unit(r), []).append(r["key"])
        print(f"\n== {name}: {len(todo)} rows to score ({len(uniq)} unique inputs), "
              f"{len(done)} already done", flush=True)
        if not uniq:
            continue

        t0 = time.time()
        reward = get_reward(name, **overrides)
        load_s = time.time() - t0
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
        except ImportError:
            torch = None

        units = list(uniq)
        score_s = 0.0
        for start in range(0, len(units), args.chunk_size):
            chunk = units[start: start + args.chunk_size]
            t1 = time.time()
            scores = reward.score([u[0] for u in chunk], [u[1] for u in chunk],
                                  [u[2] for u in chunk] if info["needs_reference"] else None)
            score_s += time.time() - t1
            append_jsonl(path, ({"key": k, "score": s} for u, s in zip(chunk, scores) for k in uniq[u]))
            print(f"   {min(start + args.chunk_size, len(units))}/{len(units)} "
                  f"({len(chunk) / max(time.time() - t1, 1e-9):.1f} inputs/s)", flush=True)

        meta = {"reward": name, **info, **device_info(), "n_rows": len(todo), "n_unique": len(units),
                "load_seconds": round(load_s, 1), "score_seconds": round(score_s, 1),
                "inputs_per_second": round(len(units) / score_s, 2) if score_s else None}
        if torch is not None and torch.cuda.is_available():
            meta["peak_gpu_mem_gb"] = round(torch.cuda.max_memory_allocated() / 1e9, 2)
        meta_path = os.path.join(out_dir, f"{name}.meta.json")
        if done and os.path.exists(meta_path):  # resumed run: accumulate timing across runs
            prev = read_json(meta_path)
            meta["n_unique"] += prev.get("n_unique", 0)
            meta["score_seconds"] = round(meta["score_seconds"] + prev.get("score_seconds", 0), 1)
            meta["inputs_per_second"] = round(meta["n_unique"] / max(meta["score_seconds"], 1e-9), 2)
        write_json(meta_path, meta)
        print(f"   done in {score_s:.0f}s ({meta['inputs_per_second']} inputs/s, load {load_s:.0f}s)")
        reward.close()


if __name__ == "__main__":
    main()
