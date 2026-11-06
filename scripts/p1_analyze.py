#!/usr/bin/env python
"""Phase 1, step 3: reward benchmark, blind-spot profile and reward choice (CPU only).

Reads RUN_DIR/{segments,probes}.jsonl and RUN_DIR/scores/*.jsonl and writes
RUN_DIR/report/:
  summary.md           the short report (also printed): paste this to share results
  reward_benchmark.csv agreement statistics per reward and test set
  blind_spots.csv/.png detection rate per reward and perturbation category
  sanity.csv           our scores vs. the stored WMT scores of the same metrics
  results.json         everything above, machine-readable

Example:
  python scripts/p1_analyze.py --run runs/p1
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from mtrlhf.io import read_json, read_jsonl, write_json  # noqa: E402
from mtrlhf.meta_eval import analysis  # noqa: E402
from mtrlhf.rewards import REGISTRY  # noqa: E402


def f(x, nd=3) -> str:
    return "–" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:.{nd}f}"


def pct(x) -> str:
    return "–" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{100 * x:.0f}"


def md_table(header: list[str], rows: list[list[str]]) -> str:
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    lines += ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join(lines)


def write_csv(path: str, rows: list[dict]) -> None:
    if not rows:
        return
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", default="runs/p1")
    ap.add_argument("--scores", default="scores", help="scores subfolder (scores_smoke for a smoke test)")
    ap.add_argument("--bootstrap", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-crosscheck", action="store_true", help="skip the mt-metrics-eval acc_eq cross-check")
    args = ap.parse_args()

    out_dir = os.path.join(args.run, "report" if args.scores == "scores" else f"report_{args.scores}")
    os.makedirs(out_dir, exist_ok=True)
    segments = read_jsonl(os.path.join(args.run, "segments.jsonl"))
    probes = read_jsonl(os.path.join(args.run, "probes.jsonl"))
    inputs_meta = read_json(os.path.join(args.run, "inputs_meta.json"))
    scores, metas = analysis.load_scores(args.run, args.scores)
    if not scores:
        sys.exit(f"No scores in {args.run}/{args.scores}; run p1_score.py first.")

    names = [n for n in REGISTRY if n in scores] + sorted(n for n in scores if n not in REGISTRY)
    seg_keys = {r["key"] for r in segments}
    scored_segments = [r for r in segments if any(r["key"] in scores[n] for n in names)]
    testsets = sorted({r["testset"] for r in scored_segments})
    std_refs = {ts: m["std_ref"] for ts, m in inputs_meta["testsets"].items()}

    values = {n: (lambda r, sc=scores[n]: sc.get(r["key"])) for n in names}
    results, samples = analysis.benchmark(scored_segments, values, args.bootstrap, args.seed)
    comparisons = analysis.paired_comparisons(samples, names)

    shipped_names = sorted({k for r in scored_segments for k in r.get("shipped", {})})
    shipped_values = {f"stored:{k}": (lambda r, k=k: r["shipped"].get(k)) for k in shipped_names}
    context, _ = analysis.benchmark(scored_segments, shipped_values, n_boot=200, seed=args.seed)

    sanity = analysis.sanity_checks(scored_segments, scores, std_refs)
    crosscheck = [] if args.no_crosscheck else analysis.crosscheck_acc_eq(scored_segments, values)
    probe_rows = analysis.probe_table(probes, scores, [r for r in segments if r["key"] in seg_keys],
                                      args.bootstrap, args.seed)
    analysis.plot_blind_spots(probe_rows, names, os.path.join(out_dir, "blind_spots.png"))
    choice = analysis.choose_rewards(results, comparisons, metas, names)

    # ---- CSV / JSON -------------------------------------------------------------
    bench_rows = []
    for n in names:
        for ts, st in results[n]["per_testset"].items():
            bench_rows.append({"reward": n, "testset": ts, **st})
        if "pooled" in results[n]:
            bench_rows.append({"reward": n, "testset": "pooled", **results[n]["pooled"]})
    write_csv(os.path.join(out_dir, "reward_benchmark.csv"), bench_rows)
    write_csv(os.path.join(out_dir, "blind_spots.csv"), probe_rows)
    write_csv(os.path.join(out_dir, "sanity.csv"), sanity + crosscheck)
    write_json(os.path.join(out_dir, "results.json"), {
        "benchmark": results, "comparisons": comparisons, "stored_metric_context": context,
        "sanity": sanity, "acc_eq_crosscheck": crosscheck, "probes": probe_rows, "choice": choice,
        "score_meta": metas, "inputs": inputs_meta,
    })

    # ---- summary.md ---------------------------------------------------------------
    order = sorted(names, key=lambda n: -results[n].get("pooled", {}).get("within_acc", -1))
    md = ["# Phase 1 results · reward benchmark (English→German)", ""]
    md.append("Data: " + "; ".join(
        f"{ts}: {inputs_meta['testsets'][ts]['n_rows']} translations of "
        f"{inputs_meta['testsets'][ts]['n_items']} source segments by "
        f"{len(inputs_meta['testsets'][ts]['systems'])} systems (MQM)" for ts in testsets))
    md += ["", "## A · Agreement with expert judgements", "",
           "*Within-source accuracy*: of all pairs of translations of the **same source** where the "
           "experts preferred one, the share the reward ranks the same way (50% = chance). "
           "This is how best-of-N, DPO and GRPO use a reward. 95% bootstrap CIs over source segments.", ""]
    hdr = ["Reward", "Needs ref", "Within-source acc. (pooled)"] + [f"Within-source {ts}" for ts in testsets] + \
          [f"acc_eq ({' / '.join(testsets)})", f"Kendall τ_b ({' / '.join(testsets)})",
           f"System acc. ({' / '.join(testsets)})", "Speed (inputs/s)"]
    rows = []
    for n in order:
        r = results[n]
        pooled = r.get("pooled")
        per = r["per_testset"]
        rows.append([
            f"**{n}**" if n in (choice.get("r_train"), choice.get("r_eval")) else n,
            "yes" if REGISTRY.get(n, {}).get("needs_reference") else "no",
            f"{pct(pooled['within_acc'])}% [{pct(pooled['within_lo'])}, {pct(pooled['within_hi'])}]"
            if pooled else "–",
            *[f"{pct(per[ts]['within_acc'])}%" if ts in per else "–" for ts in testsets],
            " / ".join(f(per[ts]["acc_eq"]) if ts in per else "–" for ts in testsets),
            " / ".join(f(per[ts]["kendall_tau_b"]) if ts in per else "–" for ts in testsets),
            " / ".join(f(per[ts]["sys_acc"], 2) if ts in per else "–" for ts in testsets),
            f((metas.get(n) or {}).get("inputs_per_second"), 1),
        ])
    md += [md_table(hdr, rows), ""]
    gpu = next((m.get("gpu") for m in metas.values() if m and m.get("gpu")), None)
    if gpu:
        md += [f"Speed measured on {gpu}.", ""]

    sig = [c for c in comparisons]
    if sig:
        md += ["### Pairwise differences (pooled within-source accuracy, paired bootstrap)", "",
               md_table(["A", "B", "A − B (points)", "95% CI", "p"],
                        [[c["a"], c["b"], f"{100 * c['diff']:+.1f}",
                          f"[{100 * c['lo']:+.1f}, {100 * c['hi']:+.1f}]", f(c["p"], 3)] for c in sig]), ""]

    if probe_rows:
        cats = [c for c in analysis.PROBE_ORDER if any(p["category"] == c and p["n"] for p in probe_rows)]
        n_by_cat = {c: max((p["n"] for p in probe_rows if p["category"] == c), default=0) for c in cats}
        md += ["## B · Blind spots", "",
               "Detection rate (%): how often the reward scores the intact translation above a damaged copy. "
               "50 = chance; below 50 = the reward *prefers* the damaged version. "
               "Low values are the predicted exploits for RL.", ""]
        prow = []
        for n in order:
            by = {p["category"]: p for p in probe_rows if p["reward"] == n}
            prow.append([n] + [pct(by[c]["detection"]) if c in by else "–" for c in cats])
        md += [md_table(["Reward"] + [f"{c.replace('_', ' ')} (n={n_by_cat[c]})" for c in cats], prow), "",
               "![blind spots](blind_spots.png)", ""]

    md += ["## C · Sanity checks", ""]
    if sanity:
        srows = []
        for s in sanity:
            status = ("✅" if s["pearson"] >= 0.99 else "⚠️ check") if s["expected_identical"] else "info"
            srows.append([s["reward"], s["testset"], s["stored"], str(s["n"]), f(s["pearson"], 4),
                          f(s["max_abs_diff"], 4), status])
        md += ["Our scores vs. the scores stored with the WMT data "
               "(✅ = same public model, should correlate ≈ 1):", "",
               md_table(["Reward", "Test set", "Stored metric", "n", "Pearson", "max abs diff", "Status"], srows), ""]
    else:
        md += ["No stored WMT scores matched our rewards.", ""]
    good = [c for c in crosscheck if "abs_diff" in c]
    if good:
        worst = max(c["abs_diff"] for c in good)
        md += [f"acc_eq recomputed with mt-metrics-eval's own implementation: max difference "
               f"{worst:.2e} over {len(good)} reward × test-set cells "
               f"({'✅ identical' if worst < 1e-6 else '⚠️ mismatch'}).", ""]
    elif crosscheck:
        md += [f"acc_eq cross-check skipped: {crosscheck[0].get('error')}", ""]

    ctx_rows = []
    for k in shipped_names:
        per = context[f"stored:{k}"]["per_testset"]
        if per:
            ctx_rows.append([k] + [f"{pct(per[ts]['within_acc'])}%" if ts in per else "–" for ts in testsets])
    if ctx_rows:
        md += ["### Context: stored WMT metric scores (not run here)", "",
               "Within-source accuracy of metrics whose scores ship with the WMT data, "
               "including larger models that are too expensive to run as an RL reward on one GPU:", "",
               md_table(["Stored metric"] + testsets, ctx_rows), ""]

    md += ["## D · Suggested choice (rule from docs/workflow.md §4.3)", ""]
    if "r_train" in choice:
        md.append(f"- **R_train (training reward): `{choice['r_train']}`**")
        weak = [p for p in probe_rows if p["reward"] == choice["r_train"] and p["n"] and p["detection"] < 0.8]
        if weak:
            md.append("  - blind spots to watch in Phase 3: " + ", ".join(
                f"{p['category'].replace('_', ' ')} ({pct(p['detection'])}%)"
                for p in sorted(weak, key=lambda p: p["detection"])))
    if "r_eval" in choice:
        md.append(f"- **R_eval (held-out evaluator): `{choice['r_eval']}`**")
    md += [f"- {note}" for note in choice.get("notes", [])]
    md.append("")

    summary = "\n".join(md)
    with open(os.path.join(out_dir, "summary.md"), "w", encoding="utf-8") as fh:
        fh.write(summary)
    print(summary)
    print(f"\nReport written to {out_dir}/")


if __name__ == "__main__":
    main()
