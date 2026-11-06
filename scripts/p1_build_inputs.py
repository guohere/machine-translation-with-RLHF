#!/usr/bin/env python
"""Phase 1, step 1: build the scoring inputs (CPU only).

Writes to RUN_DIR:
  segments.jsonl  one row per (system, segment) with an expert MQM score
  probes.jsonl    blind-spot probes (original + perturbed translations)
  inputs_meta.json

Example:
  python scripts/p1_build_inputs.py --run runs/p1 --download
"""

from __future__ import annotations

import argparse
import collections
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from mtrlhf.io import write_json, write_jsonl  # noqa: E402
from mtrlhf.meta_eval import wmt  # noqa: E402
from mtrlhf.meta_eval.probes import build_probe_rows  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", default="runs/p1", help="output directory")
    ap.add_argument("--testsets", nargs="+", default=["wmt22", "wmt23.sent"])
    ap.add_argument("--lp", default="en-de")
    ap.add_argument("--mtme-path", default=None,
                    help="mt-metrics-eval data dir (default: ~/.mt-metrics-eval/mt-metrics-eval-v2)")
    ap.add_argument("--download", action="store_true",
                    help="stream the mt-metrics-eval archive and extract only the needed test sets")
    ap.add_argument("--include-human-systems", action="store_true",
                    help="also treat MQM-scored human translations as systems")
    ap.add_argument("--probe-testset", default="wmt22")
    ap.add_argument("--probe-items", type=int, default=500)
    ap.add_argument("--seed", type=int, default=13)
    args = ap.parse_args()

    mtme_path = args.mtme_path
    if args.download:
        needed = sorted(set(args.testsets) | {args.probe_testset})
        print(f"Downloading mt-metrics-eval data for {needed} ...", flush=True)
        mtme_path = wmt.download_mtme(needed, lps=[args.lp])
        print(f"  -> {mtme_path}")

    all_rows, metas = [], {}
    for ts in args.testsets:
        try:
            rows, meta = wmt.load_mqm_rows(ts, args.lp, mtme_path, args.include_human_systems)
        except Exception as e:  # keep going with the other test sets
            print(f"!! could not load {ts} {args.lp}: {e}", file=sys.stderr)
            continue
        all_rows.extend(rows)
        metas[ts] = meta
        print(f"{ts} {args.lp}: {meta['n_rows']} scored translations, {meta['n_items']} source segments, "
              f"{len(meta['systems'])} systems, ref={meta['std_ref']}, alt_ref={meta['alt_ref']}, "
              f"stored metrics={len(meta['shipped_metrics'])}")
    if not all_rows:
        sys.exit("No test set could be loaded. Did you pass --download (or --mtme-path)?")

    # Probe items: unique source segments of the probe test set, with two human references.
    if args.probe_testset in metas:
        probe_source = [r for r in all_rows if r["testset"] == args.probe_testset]
    else:
        probe_source, _ = wmt.load_mqm_rows(args.probe_testset, args.lp, mtme_path)
    items, seen = [], set()
    for r in probe_source:
        if r["item"] not in seen:
            seen.add(r["item"])
            items.append({"src": r["src"], "ref": r["ref"], "alt_ref": r["alt_ref"]})
    other = wmt.other_language_references(args.probe_testset, args.lp, mtme_path)
    probes = build_probe_rows(items, n_items=args.probe_items, seed=args.seed, other_language=other)
    cats = collections.Counter(p["category"] for p in probes)
    print(f"probes: {cats['original']} items; per category: "
          + ", ".join(f"{c}={n}" for c, n in sorted(cats.items()) if c != "original"))
    if not other:
        print("  (no other en-X pair shares sources with this test set: wrong_language probe skipped)")

    os.makedirs(args.run, exist_ok=True)
    write_jsonl(os.path.join(args.run, "segments.jsonl"), all_rows)
    write_jsonl(os.path.join(args.run, "probes.jsonl"), probes)
    write_json(os.path.join(args.run, "inputs_meta.json"), {
        "lp": args.lp, "testsets": metas, "probe_testset": args.probe_testset,
        "probe_counts": dict(cats), "seed": args.seed, "mtme_path": mtme_path,
    })
    print(f"wrote {len(all_rows)} segment rows and {len(probes)} probe rows to {args.run}/")


if __name__ == "__main__":
    main()
