#!/usr/bin/env python
"""Phase 3: analyse the Phase 2 results (CPU only; uses the scores cached in the outputs).

For each model writes, to results/p3/:
  report.md            the readable report (also printed)
  results.json         every number in the report
  <model>_overoptimization.png, <model>_training.png, predicted_vs_observed.png
  <model>_suspicious_wins.jsonl   segments for the human error audit

Sections: A methods vs. SFT on the held-out metric; B over-optimization (GRPO
checkpoints, best-of-k, training logs); C exploits (rates, suspicious wins);
D predicted vs. observed (Phase 1 blind spots); E guards; F forgetting;
G the human check (once scripts/p3_human_eval.py has analysed the sheets);
and a cross-model summary. Missing pieces are skipped, so it can run at any time.

Example:
  python scripts/p3_analyze.py
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from mtrlhf import evaluation as ev  # noqa: E402
from mtrlhf.io import read_json, write_jsonl  # noqa: E402

PRIMARY, SECONDARY, DEV = "flores_devtest.en-de", "wmt24.en-de", "flores_dev.en-de"
FORGETTING = ["flores_devtest.de-en", "flores_devtest.en-zh"]


def f(x, nd=3, signed=False):
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "–"
    return f"{x:+.{nd}f}" if signed else f"{x:.{nd}f}"


def delta_cell(d: dict, nd: int = 3) -> str:
    if not d or math.isnan(d["diff"]):
        return "–"
    star = " *" if d["p"] < 0.05 else ""
    return f"{d['diff']:+.{nd}f} [{d['lo']:+.{nd}f}, {d['hi']:+.{nd}f}]{star}"


def table(header: list[str], rows: list[list[str]]) -> str:
    return "\n".join(["| " + " | ".join(header) + " |", "|" + "---|" * len(header)] +
                     ["| " + " | ".join(r) + " |" for r in rows])


def methods_vs_sft(outputs: dict, ts: str, n_boot: int) -> list[dict]:
    if "sft" not in outputs or ts not in outputs["sft"]:
        return []
    sft = outputs["sft"][ts]
    sd = {m: float(np.std(ev.scores(sft, m))) if ev.scores(sft, m) else float("nan") for m in (ev.TRAIN, ev.HELD_OUT)}
    res = []
    for system in ev.ordered(outputs):
        if ts not in outputs[system]:
            continue
        rows = outputs[system][ts]
        a, b = ev.aligned(rows, sft)
        r = {"system": system, "n": len(rows), **ev.corpus_scores(rows)}
        for m in (ev.TRAIN, ev.HELD_OUT):
            s = ev.scores(rows, m)
            r[m] = ev.mean(s) if s else float("nan")
            sa, sb = ev.scores(a, m), ev.scores(b, m)
            r[f"delta_{m}"] = ev.paired_delta(sa, sb, n_boot) if system != "sft" and sa and sb else None
        dt, dh = r[f"delta_{ev.TRAIN}"], r[f"delta_{ev.HELD_OUT}"]
        r["gap_sd"] = (dt["diff"] / sd[ev.TRAIN] - dh["diff"] / sd[ev.HELD_OUT]) if dt and dh else None
        r["rates"] = ev.exploit_rates(rows)
        res.append(r)
    return res


def model_section(model: str, outputs: dict, logs: dict, detection: dict, out_dir: str, n_boot: int,
                  human_dir: str = "results/p3/human_eval") -> tuple[str, dict]:
    md, data = [f"## {model}", ""], {}
    prim = methods_vs_sft(outputs, PRIMARY, n_boot)
    data["methods_primary"] = prim
    if prim:
        md += [f"### A · Methods vs. SFT on {PRIMARY} ({prim[0]['n']} segments)", "",
               "COMET-22 never guided training (held out); MetricX-QE is the training reward. Δ = paired mean "
               "difference to `sft` with a 95% bootstrap CI (* = p < 0.05). *Gap* = training-reward gain minus "
               "held-out gain, both in units of SFT's segment SD: positive means the training reward improved "
               "more than held-out quality (over-optimization).", ""]
        rows = [[ev.name(r["system"]), f(r[ev.HELD_OUT]), delta_cell(r[f"delta_{ev.HELD_OUT}"]), f(r[ev.TRAIN], 2),
                 delta_cell(r[f"delta_{ev.TRAIN}"], 2), f(r["gap_sd"], 2, signed=True), f(r["chrf"], 1),
                 f(r["bleu"], 1), ev.COST.get(r["system"], "1×")] for r in prim]
        md += [table(["System", "COMET-22 ↑ (held out)", "Δ vs SFT", "MetricX-QE ↑ (trained on)", "Δ vs SFT",
                      "Gap (SD)", "chrF", "BLEU", "Inference cost"], rows), ""]
    sec = methods_vs_sft(outputs, SECONDARY, n_boot)
    data["methods_secondary"] = sec
    if sec and any(r[ev.HELD_OUT] == r[ev.HELD_OUT] for r in sec):  # some system has held-out scores
        md += [f"{SECONDARY} (paragraph-level): " + "; ".join(
            f"{ev.name(r['system'])} {f(r[ev.HELD_OUT])} ({delta_cell(r[f'delta_{ev.HELD_OUT}'])})"
            if r["system"] != "sft" else f"sft {f(r[ev.HELD_OUT])}" for r in sec) + ".", ""]

    # B · over-optimization
    curves = {ev.name(run): ev.checkpoint_curve(outputs, run, DEV)
              for run in ("grpo", "grpo_guarded", "grpo_cometkiwi")}
    curves = {k: v for k, v in curves.items() if len(v) > 1}
    data["checkpoint_curves"] = curves
    train_logs = {run: ev.training_log_curve(log) for run, log in logs.items()}
    data["training_logs"] = train_logs
    bok = ev.best_of_k(outputs.get("sft+bo8", {}).get(DEV, []))
    data["best_of_k"] = bok
    if curves or bok or any(train_logs.values()):
        md += ["### B · Over-optimization", ""]
        if curves:
            fig = f"{model}_overoptimization.png"
            if ev.plot_overoptimization(curves, os.path.join(out_dir, fig), title=f"{model}: GRPO checkpoints on {DEV}"):
                md += [f"![over-optimization]({fig})", ""]
            for run, pts in curves.items():
                md += [f"{run}: " + ", ".join(f"step {p['step']}: MetricX-QE {p['train']:.2f} / COMET-22 "
                                               f"{p['held_out']:.3f}" for p in pts), ""]
        if any(train_logs.values()):
            fig = f"{model}_training.png"
            if ev.plot_training_guards({k: v for k, v in train_logs.items()}, os.path.join(out_dir, fig),
                                       title=f"{model}: during GRPO training (8 samples per prompt)"):
                md += [f"![training]({fig})", ""]
        if bok:
            md += ["Best-of-k on FLORES dev (chosen by the training reward): " + ", ".join(
                f"k={p['k']}: MetricX-QE {p['train']:.2f} / COMET-22 {p['held_out']:.3f}" for p in bok) + ".", ""]

    # C · exploits
    if prim:
        md += ["### C · Exploits", "", "Share of outputs with each signal (%). *short* / *long* = under 70% / over 140% "
               "of the reference length (omission / addition proxies).", ""]
        md += [table(["System"] + ev.SIGNALS, [[ev.name(r["system"])] + [f(r["rates"][k], 1) for k in ev.SIGNALS]
                                               for r in prim]), ""]
        sft_rows = outputs["sft"][PRIMARY]
        wins_by = {}
        for system in [s for s in ev.RL_SYSTEMS if PRIMARY in outputs.get(s, {})]:
            w = ev.suspicious_wins(outputs[system][PRIMARY], sft_rows)
            if w is not None:  # None = not scored yet
                wins_by[system] = w
        if wins_by:
            data["suspicious_wins"] = {s: len(w) for s, w in wins_by.items()}
            md += ["**Suspicious wins** (training reward ≥ 1 point better than SFT, COMET-22 ≥ 0.02 worse), "
                   "with the signals the RL output introduced:", ""]
            for s, w in wins_by.items():
                cats = {}
                for x in w:
                    for c in x["introduced"]:
                        cats[c] = cats.get(c, 0) + 1
                md.append(f"- {ev.name(s)}: {len(w)} of {len(sft_rows)} segments"
                          + (f" — {', '.join(f'{k} {v}' for k, v in sorted(cats.items(), key=lambda kv: -kv[1]))}"
                             if w else ""))
                write_jsonl(os.path.join(out_dir, f"{model}_suspicious_wins_{ev.name(s)}.jsonl"), w)
            md.append("")
            main = wins_by.get("grpo-step625", [])
            for x in main[:2]:
                md += [f"> **SRC** {x['src'][:200]}  ", f"> **SFT** {x['sft'][:200]}  ",
                       f"> **GRPO** {x['rl'][:200]}  ",
                       f"> MetricX-QE {x['train_delta']:+.1f}, COMET-22 {x['held_delta']:+.3f}, "
                       f"introduced: {', '.join(x['introduced'])}", ""]

        # D · predicted vs observed
        if "grpo-step625" in wins_by and detection:
            rl = next(r for r in prim if r["system"] == "grpo-step625")
            sftr = next(r for r in prim if r["system"] == "sft")
            pvo = ev.predicted_vs_observed(detection, sftr["rates"], rl["rates"], wins_by["grpo-step625"])
            data["predicted_vs_observed"] = pvo
            md += ["### D · Predicted vs. observed", "",
                   "Phase 1 measured how often the training reward misses each kind of damage; here is how often "
                   "that kind of error appears in SFT vs. GRPO outputs, and among GRPO's suspicious wins.", ""]
            md += [table(["Phase 1 probe", "Missed by reward (%)", "Signal in outputs", "SFT %", "GRPO %", "Δ (pp)",
                          "Share of suspicious wins %"],
                         [[p["probe"], f(p["missed_pct"], 1), p["signal"], f(p["sft_rate"], 1), f(p["rl_rate"], 1),
                           f(p["delta_pp"], 1, signed=True), f(p["share_of_suspicious_wins"], 1)] for p in pvo]), ""]

        # E · guards
        g = {r["system"]: r for r in prim}
        if "grpo-step625" in g and "grpo_guarded-step625" in g:
            a, b = ev.aligned(outputs["grpo_guarded-step625"][PRIMARY], outputs["grpo-step625"][PRIMARY])
            d = ev.paired_delta(ev.scores(a, ev.HELD_OUT), ev.scores(b, ev.HELD_OUT), n_boot) \
                if ev.scores(a, ev.HELD_OUT) and ev.scores(b, ev.HELD_OUT) else None
            data["guards_effect"] = {"delta_held_out": d, "rates_grpo": g["grpo-step625"]["rates"],
                                     "rates_guarded": g["grpo_guarded-step625"]["rates"]}
            md += ["### E · Do the guards help?", "",
                   f"guarded − unguarded GRPO on COMET-22: {delta_cell(d)}. Error rates (%) unguarded → guarded: "
                   + ", ".join(f"{k} {f(g['grpo-step625']['rates'][k], 1)} → {f(g['grpo_guarded-step625']['rates'][k], 1)}"
                               for k in ev.SIGNALS) + ".", ""]

    # F · forgetting
    fr = []
    for ts in FORGETTING:
        for system in ev.ordered(outputs):
            rows = outputs[system].get(ts)
            if rows:
                h = ev.scores(rows, ev.HELD_OUT)
                rates = ev.exploit_rates(rows)
                exact = 100 * ev.mean([r["hyp"].strip() == r["src"].strip() for r in rows])
                fr.append({"testset": ts, "system": system, ev.HELD_OUT: ev.mean(h) if h else float("nan"),
                           **ev.corpus_scores(rows), "wrong_language": rates["wrong_language"], "copy": rates["copy"],
                           "exact_copy": exact})
    data["forgetting"] = fr
    if fr:
        md += ["### F · Forgetting on untrained directions", "",
               "RSFT, DPO and GRPO train on English→German only; SFT (v2) also on German→English, so "
               "German→English measures forgetting of a trained direction and English→Chinese an untrained one. "
               "*Wrong language* = the output is not in the target language; "
               "*near copy* = most character 4-grams of the output occur in the source (includes rewrites of the "
               "source in its own language); *exact copy* = the output is the source, verbatim.", ""]
        md += [table(["Test set", "System", "COMET-22", "chrF", "BLEU", "Wrong language %", "Near copy %",
                      "Exact copy %"],
                     [[r["testset"], ev.name(r["system"]), f(r[ev.HELD_OUT]), f(r["chrf"], 1), f(r["bleu"], 1),
                       f(r["wrong_language"], 1), f(r["copy"], 1), f(r["exact_copy"], 1)] for r in fr]), ""]
        if os.path.exists(os.path.join(os.path.dirname(out_dir.rstrip("/")), "p2", model, "diagnostics",
                                       "direction.md")):
            md += [f"Why: see the direction diagnostic in `results/p2/{model}/diagnostics/direction.md`.", ""]

    # G · human check (scripts/p3_human_eval.py)
    human = os.path.join(human_dir, model, "summary.md")
    if os.path.exists(human):
        data["human"] = read_json(os.path.join(human_dir, model, "summary.json"))
        md += [open(human, encoding="utf-8").read().strip(), ""]
    return "\n".join(md), data


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", nargs="+", default=["qwen3-1.7b", "gemma3-1b"])
    ap.add_argument("--results", default="results/p2")
    ap.add_argument("--blind-spots", default="results/p1/report/blind_spots.csv")
    ap.add_argument("--out", default="results/p3")
    ap.add_argument("--bootstrap", type=int, default=1000)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    detection = ev.load_blind_spots(args.blind_spots)
    sections, data, pvo = [], {}, {}
    for model in args.models:
        outputs = ev.load_outputs(model, args.results)
        if not outputs:
            continue
        logs = {}
        for run in ("grpo", "grpo_guarded", "grpo_cometkiwi"):
            p = os.path.join(args.results, model, "training", run, "log_history.json")
            if os.path.exists(p):
                logs[run] = read_json(p)
        md, d = model_section(model, outputs, logs, detection, args.out, args.bootstrap,
                              os.path.join(args.out, "human_eval"))
        sections.append(md)
        data[model] = d
        if d.get("predicted_vs_observed"):
            pvo[model] = d["predicted_vs_observed"]

    head = ["# Phase 3 results · held-out evaluation, over-optimization and exploits", "",
            f"Training reward: MetricX-24-QE. Held-out evaluator: COMET-22. Generated by `scripts/p3_analyze.py` "
            f"from `{args.results}/`; sections appear as their inputs exist.", ""]
    if pvo and ev.plot_predicted_vs_observed(pvo, os.path.join(args.out, "predicted_vs_observed.png")):
        head += ["![predicted vs observed](predicted_vs_observed.png)", ""]
    both = [m for m in data if data[m].get("methods_primary")]
    if len(both) > 1:
        head += ["## Does it hold for both model families?", ""]
        rows = []
        for system in ev.RL_SYSTEMS:
            cells = []
            for m in both:
                r = next((x for x in data[m]["methods_primary"] if x["system"] == system), None)
                d = r and r[f"delta_{ev.HELD_OUT}"]
                cells.append(delta_cell(d) if d else "–")
            if any(c != "–" for c in cells):
                rows.append([ev.name(system)] + cells)
        head += [table(["System (COMET-22 Δ vs SFT)"] + both, rows), ""]
    report = "\n".join(head + sections) if sections else "\n".join(head + ["No Phase 2 outputs yet."])
    with open(os.path.join(args.out, "report.md"), "w", encoding="utf-8") as fh:
        fh.write(report)
    ev.dump(os.path.join(args.out, "results.json"), data)
    print(report)


if __name__ == "__main__":
    main()
