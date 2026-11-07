#!/usr/bin/env python
"""Phase 2: translate the test sets with each system (greedy, or best-of-N).

Outputs go to OUT/<system>/<testset>.jsonl with rows {id, src, ref, hyp, raw}.
Best-of-N (--best-of N) samples N translations, scores them with --rerank-reward
(the training reward, MetricX-24-QE) and keeps the best; the system is then named
<system>+bo<N> and every candidate is stored for the Phase 3 best-of-N curve.
Existing output files are skipped, so the script can be re-run after a stop.

--score adds a quick summary (mean score per reward and guard-flag rates) to
OUT/quick_eval.md. The held-out evaluation and hacking analysis are Phase 3.

Examples:
  python scripts/p2_translate.py --systems base sft rsft dpo grpo grpo_guarded opus-mt \
      --testsets flores_devtest.en-de wmt24.en-de flores_devtest.de-en flores_devtest.en-zh
  python scripts/p2_translate.py --systems sft --best-of 8 --testsets flores_devtest.en-de
  python scripts/p2_translate.py --summary-only --score chrf metricx24_qe comet22
"""

from __future__ import annotations

import argparse
import glob
import os
import statistics
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from mtrlhf.data.prompts import extract_translation  # noqa: E402
from mtrlhf.io import read_jsonl, write_jsonl  # noqa: E402

DEFAULT_SETS = ["flores_dev.en-de", "flores_devtest.en-de", "wmt24.en-de", "flores_devtest.de-en",
                "flores_devtest.en-zh"]


def translate(args) -> None:
    from mtrlhf.models import DEFAULT_BASE, EXTERNAL, CheckpointStore, Translator, default_device
    from mtrlhf.rewards import get_reward, reward_info

    store = CheckpointStore.from_env(args.ckpt, args.hub_repo)
    rerank = None
    for system in args.systems:
        name = f"{system}+bo{args.best_of}" if args.best_of > 1 else system
        jobs = []
        for ts in args.testsets:
            rows = read_jsonl(os.path.join(args.data, "testsets", f"{ts}.jsonl"))
            if args.limit:
                rows = rows[: args.limit]
            if not rows:
                print(f"!! test set {ts} not found; skipping")
                continue
            pair = (rows[0]["src_lang"], rows[0]["tgt_lang"])
            if system in EXTERNAL and "pairs" in EXTERNAL[system] and pair not in EXTERNAL[system]["pairs"]:
                continue
            path = os.path.join(args.out, name, f"{ts}.jsonl")
            if os.path.exists(path):
                print(f"{name} / {ts}: exists, skipping")
                continue
            jobs.append((ts, rows, path))
        if not jobs:
            continue
        tr = Translator.for_system(system, store, args.base_model or DEFAULT_BASE, device=default_device(0),
                                   batch_size=args.batch_size)
        if args.best_of > 1 and rerank is None:
            info = reward_info(args.rerank_reward)
            rerank = get_reward(args.rerank_reward,
                                **({"device": default_device(1)} if info["kind"] == "metricx" else {}))
        for ts, rows, path in jobs:
            long = ts.startswith("wmt24")
            src_lang, tgt_lang = rows[0]["src_lang"], rows[0]["tgt_lang"]
            srcs = [r["src"] for r in rows]
            max_new = args.max_new_tokens or (768 if long else 256)
            if args.best_of > 1:
                raw = tr.generate(srcs, src_lang, tgt_lang, n=args.best_of, do_sample=True,
                                  temperature=args.temperature, top_p=args.top_p, max_new_tokens=max_new)
                hyps = [[extract_translation(o, r["src"])[0] for o in outs] for r, outs in zip(rows, raw)]
                flat_src = [r["src"] for r, hs in zip(rows, hyps) for _ in hs]
                flat_ref = [r["ref"] for r, hs in zip(rows, hyps) for _ in hs]
                flat = rerank.score(flat_src, [h for hs in hyps for h in hs],
                                    flat_ref if rerank.needs_reference else None)
                out, k = [], 0
                for r, outs, hs in zip(rows, raw, hyps):
                    scores = flat[k: k + len(hs)]
                    k += len(hs)
                    best = max(range(len(hs)), key=lambda i: scores[i])
                    out.append({**r, "hyp": hs[best], "raw": outs[best],
                                "candidates": [{"hyp": h, "score": s} for h, s in zip(hs, scores)]})
            else:
                raw = tr.generate(srcs, src_lang, tgt_lang, max_new_tokens=max_new)
                out = []
                for r, outs in zip(rows, raw):
                    hyp = outs[0] if tr.kind == "seq2seq" else extract_translation(outs[0], r["src"])[0]
                    out.append({**r, "hyp": hyp, **({"raw": outs[0]} if outs[0].strip() != hyp else {})})
            write_jsonl(path, out)
            print(f"{name} / {ts}: {len(out)} translations -> {path}", flush=True)
        tr.close()
    if rerank is not None:
        rerank.close()


def summarize(args) -> None:
    """Mean score per (system, test set, reward) and guard-flag rates -> quick_eval.md."""
    from mtrlhf.rewards import get_reward, reward_info
    from mtrlhf.rewards.guards import guard_flags

    files = sorted(glob.glob(os.path.join(args.out, "*", "*.jsonl")))
    table: dict[tuple[str, str], dict] = {}
    for f in files:
        system, ts = os.path.basename(os.path.dirname(f)), os.path.basename(f)[: -len(".jsonl")]
        rows = read_jsonl(f)
        tgt = rows[0]["tgt_lang"] if rows else "de"
        flags = [guard_flags(r["src"], r["hyp"], tgt) for r in rows]
        table[(system, ts)] = {"n": len(rows), **{f"{k} %": 100 * statistics.mean(fl[k] for fl in flags)
                                                   for k in flags[0]}} if rows else {"n": 0}
    for name in args.score:
        info = reward_info(name)
        reward = get_reward(name)
        for f in files:
            system, ts = os.path.basename(os.path.dirname(f)), os.path.basename(f)[: -len(".jsonl")]
            rows = read_jsonl(f)
            if not rows:
                continue
            col = f"scores_{name}"
            if col not in rows[0]:  # cache scores inside the output file
                s = reward.score([r["src"] for r in rows], [r["hyp"] for r in rows],
                                 [r["ref"] for r in rows] if info["needs_reference"] else None)
                for r, v in zip(rows, s):
                    r[col] = v
                write_jsonl(f, rows)
            table[(system, ts)][name] = statistics.mean(r[col] for r in rows)
        reward.close()
    cols = list(dict.fromkeys(k for v in table.values() for k in v))
    lines = ["| System | Test set | " + " | ".join(cols) + " |", "|" + "---|" * (len(cols) + 2)]
    for (system, ts), v in sorted(table.items(), key=lambda x: (x[0][1], x[0][0])):
        lines.append(f"| {system} | {ts} | " + " | ".join(
            (f"{v[c]:.3f}" if isinstance(v.get(c), float) else str(v.get(c, "–"))) for c in cols) + " |")
    md = "# Phase 2 quick evaluation\n\nMeans over each test set. Guard columns: % of outputs flagged. " \
         "This is a sanity check; the held-out evaluation is Phase 3.\n\n" + "\n".join(lines) + "\n"
    with open(os.path.join(args.out, "quick_eval.md"), "w", encoding="utf-8") as fh:
        fh.write(md)
    print(md)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--systems", nargs="*", default=[])
    ap.add_argument("--testsets", nargs="+", default=DEFAULT_SETS)
    ap.add_argument("--data", default="data/p2")
    ap.add_argument("--out", default="results/p2/outputs")
    ap.add_argument("--best-of", type=int, default=1)
    ap.add_argument("--rerank-reward", default="metricx24_qe")
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top-p", type=float, default=1.0)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--max-new-tokens", type=int, default=None)
    ap.add_argument("--base-model", default=None)
    ap.add_argument("--ckpt", default="ckpt")
    ap.add_argument("--hub-repo", default=None,
                    help='private HF repo mirroring the adapters ("auto" = <user>/mtrlhf-en-de-adapters, "none" = off; default: $MTRLHF_HUB_REPO)')
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--score", nargs="*", default=[], help="rewards for the quick summary, e.g. chrf metricx24_qe")
    ap.add_argument("--summary-only", action="store_true")
    args = ap.parse_args()
    if not args.summary_only:
        translate(args)
    if args.score or args.summary_only:
        summarize(args)


if __name__ == "__main__":
    main()
