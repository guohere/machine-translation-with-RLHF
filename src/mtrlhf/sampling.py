"""One sampling setting for every sampling-based method, chosen by a pilot.

Candidate sampling (RSFT / DPO data), best-of-N and GRPO must all draw samples the
same way, for both models. The setting lives in results/p2/sampling.json (written by
scripts/p2_sampling_pilot.py) and every script reads it from there unless a value is
given explicitly.

The pilot trades off two things the methods need:
  * diversity: the methods learn from differences between samples of the same source
    (no spread = no DPO pairs and zero GRPO advantages);
  * quality: broken samples waste the reward budget and make preference pairs trivial.
Rule (fixed before running it): among settings with at most MAX_BROKEN_PCT % broken
samples, take the one with the largest mean within-group reward spread; near-ties go
to top_p = 1.0, which keeps GRPO exactly on-policy (TRL accounts for the temperature in
its log-probabilities but not for top-p truncation), then to fewer broken samples.
"""

from __future__ import annotations

import json
import os
import statistics

DEFAULT_FILE = "results/p2/sampling.json"
FALLBACK = {"temperature": 1.0, "top_p": 1.0}
BROKEN_ERROR = 10.0      # MetricX-24 error above this = broken translation
MAX_BROKEN_PCT = 5.0
TIE_TOLERANCE = 0.05     # spreads closer than this (MetricX points) count as a tie


def sampling_file() -> str:
    return os.environ.get("MTRLHF_SAMPLING_FILE", DEFAULT_FILE)


def load_sampling(path: str | None = None) -> dict | None:
    path = path or sampling_file()
    if not os.path.exists(path):
        return None
    with open(path) as fh:
        return json.load(fh)


def resolve(temperature: float | None, top_p: float | None, path: str | None = None) -> tuple[float, float, str]:
    """Explicit values win; otherwise the pilot's choice; otherwise 1.0 / 1.0 (with a warning)."""
    chosen = load_sampling(path)
    base = chosen or FALLBACK
    t = base["temperature"] if temperature is None else temperature
    p = base["top_p"] if top_p is None else top_p
    if temperature is not None and top_p is not None:
        source = "command line"
    elif chosen:
        source = f"sampling pilot ({path or sampling_file()})"
    else:
        source = "fallback 1.0 / 1.0: no sampling pilot result found"
    return float(t), float(p), source


def group_stats(groups: list[list[tuple[str, float]]], broken_error: float = BROKEN_ERROR) -> dict:
    """Quality and diversity of sampled groups.

    groups: one list per source of (translation, score) pairs; scores are rewards
    where higher is better and -error for MetricX (so broken = score < -broken_error).
    """
    scores = [s for g in groups for _, s in g]
    spreads = [statistics.pstdev([s for _, s in g]) for g in groups if len(g) > 1]
    return {
        "groups": len(groups),
        "samples": len(scores),
        "broken_pct": 100 * sum(s < -broken_error for s in scores) / max(len(scores), 1),
        "mean_score": statistics.mean(scores) if scores else float("nan"),
        "mean_best": statistics.mean(max(s for _, s in g) for g in groups) if groups else float("nan"),
        "mean_worst": statistics.mean(min(s for _, s in g) for g in groups) if groups else float("nan"),
        "spread": statistics.mean(spreads) if spreads else 0.0,
        "identical_groups_pct": 100 * sum(len({h for h, _ in g}) == 1 for g in groups) / max(len(groups), 1),
        "unique_per_group": statistics.mean(len({h for h, _ in g}) for g in groups) if groups else 0.0,
    }


def choose(results: list[dict], max_broken_pct: float = MAX_BROKEN_PCT,
           tie_tolerance: float = TIE_TOLERANCE) -> tuple[dict, str]:
    """Apply the pre-registered rule to [{temperature, top_p, broken_pct, spread, ...}]."""
    eligible = [r for r in results if r["broken_pct"] <= max_broken_pct]
    if not eligible:
        best = min(results, key=lambda r: r["broken_pct"])
        return best, (f"no setting had <= {max_broken_pct}% broken samples; took the one with the fewest "
                      f"({best['broken_pct']:.1f}%)")
    top = max(r["spread"] for r in eligible)
    near = [r for r in eligible if r["spread"] >= top - tie_tolerance]
    best = min(near, key=lambda r: (r["top_p"] != 1.0, r["broken_pct"], -r["spread"]))
    why = (f"largest within-group reward spread ({best['spread']:.2f}) among the {len(eligible)} settings with "
           f"<= {max_broken_pct}% broken samples")
    if len(near) > 1:
        why += f"; {len(near)} settings were within {tie_tolerance} of the best spread, top_p = 1.0 preferred"
    return best, why
