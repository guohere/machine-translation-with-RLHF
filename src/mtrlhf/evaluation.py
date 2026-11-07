"""Phase 3: held-out evaluation, over-optimization, exploits and predicted-vs-observed.

Reads what Phase 2 pushed to results/p2/<model>/ (translations with cached scores,
GRPO training logs) and Phase 1's blind-spot profile (results/p1/report/), and
computes everything in results/p3/. Pure CPU; scoring happens in scripts/p3_score.py.

Conventions: TRAIN = the training reward (MetricX-24-QE, negated error, higher is
better), HELD_OUT = the evaluator from another model family (COMET-22). Every
comparison is paired by segment and reported with a bootstrap 95% CI.
"""

from __future__ import annotations

import collections
import glob
import json
import os
import re
import statistics

import numpy as np

from mtrlhf.io import read_jsonl
from mtrlhf.rewards import guards

TRAIN = "metricx24_qe"
HELD_OUT = "comet22"
BUDGET_STEP = 625
SYSTEM_ORDER = ["base", "opus-mt", "sft", "sft+bo8", "rsft", "dpo", "grpo-step625", "grpo_guarded-step625",
                "grpo_cometkiwi-step625"]
RL_SYSTEMS = ["sft+bo8", "rsft", "dpo", "grpo-step625", "grpo_guarded-step625", "grpo_cometkiwi-step625"]
DISPLAY = {"grpo-step625": "grpo", "grpo_guarded-step625": "grpo_guarded", "grpo_cometkiwi-step625": "grpo_cometkiwi"}
COST = {"sft+bo8": "8 samples + 8 reward calls"}

# Phase 1 probe category -> the observable signal in real outputs that it predicts
PROBE_TO_SIGNAL = {"number": "number_mismatch", "wrong_language": "wrong_language", "omission": "short",
                   "addition": "long", "untranslated": "copy", "repetition": "repetition"}
SIGNALS = ["wrong_language", "number_mismatch", "short", "long", "copy", "repetition"]


def name(system: str) -> str:
    return DISPLAY.get(system, system)


# --- loading -------------------------------------------------------------------------

def load_outputs(model: str, root: str = "results/p2") -> dict[str, dict[str, list[dict]]]:
    """{system: {testset: rows}} for every translation file of a model."""
    out: dict[str, dict[str, list[dict]]] = collections.defaultdict(dict)
    for f in glob.glob(os.path.join(root, model, "outputs", "*", "*.jsonl")):
        system, ts = os.path.basename(os.path.dirname(f)), os.path.basename(f)[: -len(".jsonl")]
        out[system][ts] = read_jsonl(f)
    return dict(out)


def ordered(systems) -> list[str]:
    systems = list(systems)
    known = [s for s in SYSTEM_ORDER if s in systems]
    return known + sorted(s for s in systems if s not in known)


# --- scores and statistics -----------------------------------------------------------

def scores(rows: list[dict], metric: str) -> list[float] | None:
    col = f"scores_{metric}"
    if not rows or any(col not in r for r in rows):
        return None
    return [float(r[col]) for r in rows]


def mean(xs) -> float:
    return float(statistics.mean(xs)) if xs else float("nan")


def corpus_scores(rows: list[dict]) -> dict:
    from sacrebleu.metrics import BLEU, CHRF

    if not rows:
        return {"chrf": float("nan"), "bleu": float("nan")}
    hyps, refs = [r["hyp"] for r in rows], [[r["ref"] for r in rows]]
    tok = "zh" if rows[0].get("tgt_lang") == "zh" else "13a"
    return {"chrf": CHRF().corpus_score(hyps, refs).score, "bleu": BLEU(tokenize=tok).corpus_score(hyps, refs).score}


def aligned(a: list[dict], b: list[dict]) -> tuple[list[dict], list[dict]]:
    ib = {r["id"]: r for r in b}
    pairs = [(r, ib[r["id"]]) for r in a if r["id"] in ib]
    return [p[0] for p in pairs], [p[1] for p in pairs]


def paired_delta(a: list[float], b: list[float], n_boot: int = 1000, seed: int = 0) -> dict:
    """Mean of a - b over segments, with a bootstrap 95% CI and two-sided p-value."""
    d = np.asarray(a, float) - np.asarray(b, float)
    if len(d) == 0:
        return {"diff": float("nan"), "lo": float("nan"), "hi": float("nan"), "p": float("nan")}
    idx = np.random.default_rng(seed).integers(0, len(d), size=(n_boot, len(d)))
    boots = d[idx].mean(1)
    p = 2 * min((boots <= 0).mean(), (boots >= 0).mean())
    return {"diff": float(d.mean()), "lo": float(np.quantile(boots, 0.025)), "hi": float(np.quantile(boots, 0.975)),
            "p": float(min(1.0, p))}


# --- exploit signals -----------------------------------------------------------------

def repetition(hyp: str, n: int = 3, times: int = 3) -> bool:
    words = hyp.split()
    grams = collections.Counter(tuple(words[i:i + n]) for i in range(len(words) - n + 1))
    return bool(grams) and max(grams.values()) >= times


def exploit_flags(row: dict) -> dict[str, bool]:
    """Observable failure signals of one translation (the outputs-side view of the probes)."""
    src, hyp, ref = row["src"], row["hyp"], row.get("ref") or ""
    tgt = row.get("tgt_lang", "de")
    ratio = len(hyp.strip()) / max(len(ref.strip()), 1)
    return {"wrong_language": guards.wrong_language(hyp, tgt), "number_mismatch": guards.number_mismatch(src, hyp),
            "short": bool(ref) and ratio < 0.7, "long": bool(ref) and ratio > 1.4, "copy": guards.copy(src, hyp),
            "repetition": repetition(hyp)}


def exploit_rates(rows: list[dict]) -> dict[str, float]:
    flags = [exploit_flags(r) for r in rows]
    return {k: 100 * mean([f[k] for f in flags]) for k in SIGNALS} if flags else {k: float("nan") for k in SIGNALS}


def suspicious_wins(rl: list[dict], sft: list[dict], train_margin: float = 1.0,
                    held_margin: float = 0.02) -> list[dict] | None:
    """Segments where the training reward says RL is clearly better but the held-out metric says worse.

    None if either system lacks the scores.
    """
    a, b = aligned(rl, sft)
    ta, tb, ha, hb = scores(a, TRAIN), scores(b, TRAIN), scores(a, HELD_OUT), scores(b, HELD_OUT)
    if None in (ta, tb, ha, hb):
        return None
    out = []
    for r, s, x1, x0, y1, y0 in zip(a, b, ta, tb, ha, hb):
        if x1 - x0 >= train_margin and y1 - y0 <= -held_margin:
            fr, fs = exploit_flags(r), exploit_flags(s)
            introduced = [k for k in SIGNALS if fr[k] and not fs[k]]
            out.append({"id": r["id"], "src": r["src"], "ref": r.get("ref"), "rl": r["hyp"], "sft": s["hyp"],
                        "train_delta": x1 - x0, "held_delta": y1 - y0, "introduced": introduced or ["other"]})
    return out


# --- curves --------------------------------------------------------------------------

def checkpoint_curve(outputs: dict, run: str, dev: str = "flores_dev.en-de") -> list[dict]:
    """Mean training reward and held-out score per saved GRPO step (step 0 = sft)."""
    pts = []
    for system, sets in outputs.items():
        m = re.fullmatch(rf"{re.escape(run)}-step(\d+)", system)
        step = int(m.group(1)) if m else (0 if system == "sft" else None)
        if step is None or dev not in sets:
            continue
        rows = sets[dev]
        t, h = scores(rows, TRAIN), scores(rows, HELD_OUT)
        if t is None or h is None:
            continue
        pts.append({"step": step, "train": mean(t), "held_out": mean(h), **exploit_rates(rows)})
    return sorted(pts, key=lambda p: p["step"])


def training_reward_key(log_history: list[dict]) -> str | None:
    """The TRL log key of the training reward (`rewards/<name>/mean`; guards are `rewards/guard_*`)."""
    for e in log_history:
        for k in e:
            m = re.fullmatch(r"rewards/(.+)/mean", k)
            if m and not m.group(1).startswith("guard_"):
                return k
    return None


def training_log_curve(log_history: list[dict]) -> list[dict]:
    """Per logged step: training reward, guard rates (% of samples), KL and completion length."""
    key = training_reward_key(log_history)
    pts = []
    for e in log_history:
        if key is None or key not in e:
            continue
        p = {"step": e.get("step"), "reward": e[key], "kl": e.get("kl"), "length": e.get("completions/mean_length")}
        for g in guards.GUARDS:
            v = e.get(f"rewards/guard_{g}/mean")
            p[g] = None if v is None else -100 * v
        pts.append(p)
    return pts


def best_of_k(rows: list[dict], ks=(1, 2, 4, 8)) -> list[dict]:
    """Best-of-k by the training reward, from stored best-of-N candidates (needs their held-out scores)."""
    if not rows or any("candidates" not in r for r in rows):
        return []
    out = []
    for k in ks:
        picked = [max(r["candidates"][:k], key=lambda c: c["score"]) for r in rows if len(r["candidates"]) >= k]
        if not picked or any(HELD_OUT not in c for c in picked):
            continue
        out.append({"k": k, "train": mean([c["score"] for c in picked]), "held_out": mean([c[HELD_OUT] for c in picked])})
    return out


# --- Phase 1 link --------------------------------------------------------------------

def load_blind_spots(path: str = "results/p1/report/blind_spots.csv", reward: str = TRAIN) -> dict[str, float]:
    import csv

    if not os.path.exists(path):
        return {}
    with open(path) as fh:
        return {r["category"]: float(r["detection"]) for r in csv.DictReader(fh)
                if r["reward"] == reward and r["detection"] not in ("", "nan")}


def predicted_vs_observed(detection: dict[str, float], sft_rates: dict, rl_rates: dict, wins: list[dict]) -> list[dict]:
    n_wins = max(len(wins), 1)
    rows = []
    for probe, signal in PROBE_TO_SIGNAL.items():
        if probe not in detection:
            continue
        rows.append({"probe": probe, "signal": signal, "missed_pct": 100 * (1 - detection[probe]),
                     "sft_rate": sft_rates.get(signal), "rl_rate": rl_rates.get(signal),
                     "delta_pp": rl_rates.get(signal, float("nan")) - sft_rates.get(signal, float("nan")),
                     "share_of_suspicious_wins": 100 * sum(signal in w["introduced"] for w in wins) / n_wins})
    return rows


# --- figures -------------------------------------------------------------------------

COLORS = {"grpo": "#2a78d6", "grpo_guarded": "#eb6834", "grpo_cometkiwi": "#1baf7a"}
SURFACE, INK, MUTED, GRID = "#fcfcfb", "#1f1f1e", "#6b6a66", "#e6e5e1"


def _axes(n: int, width: float = 4.2, height: float = 3.2):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, n, figsize=(width * n, height), dpi=150, squeeze=False)
    fig.patch.set_facecolor(SURFACE)
    for ax in axes[0]:
        ax.set_facecolor(SURFACE)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        for s in ("left", "bottom"):
            ax.spines[s].set_color(GRID)
        ax.tick_params(colors=MUTED, labelsize=8)
        ax.grid(axis="y", color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)
    return fig, axes[0]


def plot_overoptimization(curves: dict[str, list[dict]], path: str, budget: int = BUDGET_STEP, title: str = "") -> bool:
    curves = {k: v for k, v in curves.items() if len(v) > 1}
    if not curves:
        return False
    fig, axes = _axes(2)
    for ax, key, label in ((axes[0], "train", "MetricX-QE (training reward, ↑)"),
                           (axes[1], "held_out", "COMET-22 (held out, ↑)")):
        for run, pts in curves.items():
            ax.plot([p["step"] for p in pts], [p[key] for p in pts], color=COLORS.get(run, INK), linewidth=2,
                    marker="o", markersize=5, label=run)
        ax.axvline(budget, color=MUTED, linewidth=1, linestyle="--")
        ax.text(budget, ax.get_ylim()[1], " matched budget", color=MUTED, fontsize=7, va="top")
        ax.set_title(label, fontsize=9, color=INK, loc="left")
        ax.set_xlabel("GRPO step (0 = sft)", fontsize=8, color=MUTED)
    axes[1].legend(frameon=False, fontsize=8, labelcolor=INK)
    if title:
        fig.suptitle(title, fontsize=10, color=INK, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(path, facecolor=SURFACE)
    return True


def smooth(ys: list[float], window: int = 5) -> list[float]:
    """Trailing moving average (TRL logs every 10 steps, so 5 points = 50 steps)."""
    return [float(np.mean(ys[max(0, i - window + 1): i + 1])) for i in range(len(ys))]


def plot_training_guards(logs: dict[str, list[dict]], path: str, title: str = "") -> bool:
    logs = {k: v for k, v in logs.items() if v}
    if not logs:
        return False
    fig, axes = _axes(3)
    for ax, key, label in ((axes[0], "reward", "Training reward (MetricX-QE)"),
                           (axes[1], "number_mismatch", "Number mismatch (% of samples)"),
                           (axes[2], "wrong_language", "Wrong language (% of samples)")):
        for run, pts in logs.items():
            xs = [p["step"] for p in pts if p.get(key) is not None]
            ys = [p[key] for p in pts if p.get(key) is not None]
            if xs:  # raw logged values faintly, a moving average on top
                ax.plot(xs, ys, color=COLORS.get(run, INK), linewidth=1, alpha=0.25)
                ax.plot(xs, smooth(ys), color=COLORS.get(run, INK), linewidth=2, label=run)
        ax.set_title(label, fontsize=9, color=INK, loc="left")
        ax.set_xlabel("GRPO step", fontsize=8, color=MUTED)
    axes[0].legend(frameon=False, fontsize=8, labelcolor=INK)
    if title:
        fig.suptitle(title, fontsize=10, color=INK, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(path, facecolor=SURFACE)
    return True


def _label_points(ax, rows: list[dict]) -> None:
    """Label each probe once, moving a label up or down until it clears the ones already placed."""
    ax.figure.canvas.draw()
    pt = ax.figure.dpi / 72  # display pixels per point
    placed: list[tuple[float, float, float]] = []  # (x, y, width) in points
    for r in sorted(rows, key=lambda r: r["missed_pct"]):
        x, y = (v / pt for v in ax.transData.transform((r["missed_pct"], r["delta_pp"])))
        text = r["probe"].replace("_", " ")
        for dy in (4, -11, 13, -20, 22, -29, 31):
            if all(x + 5 > px + w or px > x + 5 + 4 * len(text) or abs(y + dy - py) > 9 for px, py, w in placed):
                break
        placed.append((x + 5, y + dy, 4 * len(text)))
        ax.annotate(text, (r["missed_pct"], r["delta_pp"]), fontsize=7, color=INK, xytext=(5, dy),
                    textcoords="offset points")


def plot_predicted_vs_observed(by_model: dict[str, list[dict]], path: str) -> bool:
    by_model = {m: rows for m, rows in by_model.items() if rows}
    if not by_model:
        return False
    fig, axes = _axes(1, width=5.2, height=3.8)
    ax = axes[0]
    palette = ["#2a78d6", "#eb6834"]
    for i, (model, rows) in enumerate(by_model.items()):
        xs = [r["missed_pct"] for r in rows]
        ys = [r["delta_pp"] for r in rows]
        ax.scatter(xs, ys, s=60, color=palette[i % 2], edgecolor=SURFACE, linewidth=2, label=model, zorder=3)
    ax.axhline(0, color=MUTED, linewidth=1)
    _label_points(ax, next(iter(by_model.values())))
    ax.set_xlabel("Phase 1: probes the training reward missed (%)", fontsize=8, color=MUTED)
    ax.set_ylabel("GRPO vs. SFT: change in error rate (pp)", fontsize=8, color=MUTED)
    ax.set_title("Did the measured blind spots predict the exploits?", fontsize=9, color=INK, loc="left")
    ax.legend(frameon=False, fontsize=8, labelcolor=INK)
    fig.tight_layout()
    fig.savefig(path, facecolor=SURFACE)
    return True


# --- human check -----------------------------------------------------------------------

def sign_test(wins: int, losses: int) -> float:
    """Two-sided exact binomial test of wins vs. losses (ties dropped) against 50/50."""
    from math import comb

    n = wins + losses
    if n == 0:
        return float("nan")
    k = min(wins, losses)
    return min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n)


def win_rate(outcomes: list[float], n_boot: int = 2000, seed: int = 0) -> dict:
    """outcomes: 1 = win, 0.5 = tie, 0 = loss. Win rate (ties count half) with a bootstrap 95% CI."""
    x = np.asarray(outcomes, float)
    wins, losses = int((x == 1).sum()), int((x == 0).sum())
    if len(x) == 0:
        return {"n": 0, "wins": 0, "losses": 0, "ties": 0, "win_rate": float("nan"), "lo": float("nan"),
                "hi": float("nan"), "p": float("nan")}
    boots = x[np.random.default_rng(seed).integers(0, len(x), size=(n_boot, len(x)))].mean(1)
    return {"n": len(x), "wins": wins, "losses": losses, "ties": len(x) - wins - losses,
            "win_rate": float(x.mean()), "lo": float(np.quantile(boots, 0.025)),
            "hi": float(np.quantile(boots, 0.975)), "p": sign_test(wins, losses)}


def dump(path: str, obj) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, ensure_ascii=False, indent=2, default=float)
