"""Agreement statistics between a reward and expert human scores.

Scores are arranged as matrices H (human) and M (reward) of shape
[n_items, n_systems], with NaN where a score is missing. An "item" is one source
segment, so every row holds competing translations of the same source.

* within_source_accuracy: of all same-source pairs where the experts preferred
  one translation, the share the reward orders the same way (reward ties count
  half). This is how best-of-N, DPO and GRPO use a reward, so it is the primary
  statistic of Phase 1.
* tie_calibrated_accuracy: acc_eq / "acc23" of Deutsch et al. (2023), averaged
  over items with one global tie threshold. Matches mt-metrics-eval's
  KendallWithTiesOpt(variant="acc23", average_by="item", sample_rate=1.0).
* kendall_tau_b: segment-level Kendall tau-b over all scored cells (no grouping).
* system_accuracy: share of system pairs ranked in the human order.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Pairs:
    item: np.ndarray  # item index per pair
    dh: np.ndarray    # human score difference (sys_i - sys_j)
    dm: np.ndarray    # reward score difference
    n_items: int


def item_pairs(H: np.ndarray, M: np.ndarray) -> Pairs:
    """All within-item system pairs where both scores exist for both systems."""
    H = np.asarray(H, dtype=float)
    M = np.asarray(M, dtype=float)
    if H.shape != M.shape:
        raise ValueError("H and M must have the same shape")
    n_items, n_sys = H.shape
    iu, ju = np.triu_indices(n_sys, k=1)
    dh = H[:, iu] - H[:, ju]
    dm = M[:, iu] - M[:, ju]
    valid = ~np.isnan(dh) & ~np.isnan(dm)
    item = np.broadcast_to(np.arange(n_items)[:, None], dh.shape)
    return Pairs(item[valid], dh[valid], dm[valid], n_items)


def within_source_counts(p: Pairs) -> tuple[np.ndarray, np.ndarray]:
    """Per-item (credit, count) over pairs with a human preference."""
    pref = p.dh != 0
    agree = np.sign(p.dm[pref]) == np.sign(p.dh[pref])
    credit = np.where(p.dm[pref] == 0, 0.5, agree.astype(float))
    correct = np.bincount(p.item[pref], weights=credit, minlength=p.n_items)
    count = np.bincount(p.item[pref], minlength=p.n_items).astype(float)
    return correct, count


def within_source_accuracy(H, M) -> tuple[float, int]:
    correct, count = within_source_counts(item_pairs(H, M))
    n = int(count.sum())
    return (float(correct.sum() / n) if n else float("nan")), n


def tie_calibrated_accuracy(H, M) -> tuple[float, float, np.ndarray, np.ndarray]:
    """acc_eq averaged over items, with the tie threshold chosen to maximize it.

    Returns (accuracy, threshold, per_item_agreements, per_item_pairs) where the
    per-item arrays are evaluated at the chosen threshold (for bootstrapping).
    """
    p = item_pairs(H, M)
    if len(p.dh) == 0:
        return float("nan"), 0.0, np.zeros(p.n_items), np.zeros(p.n_items)
    pairs_per_item = np.bincount(p.item, minlength=p.n_items).astype(float)
    n_rows = int((pairs_per_item > 0).sum())
    w = 1.0 / (pairs_per_item[p.item] * n_rows)

    h_tie = p.dh == 0
    concordant = (~h_tie) & (p.dm != 0) & (np.sign(p.dm) == np.sign(p.dh))
    d = np.abs(p.dm)
    order = np.argsort(d, kind="stable")
    d_sorted = d[order]
    cum_tie = np.concatenate([[0.0], np.cumsum((w * h_tie)[order])])
    cum_con = np.concatenate([[0.0], np.cumsum((w * concordant)[order])])
    total_con = cum_con[-1]

    # A pair counts as a reward tie when |dm| <= threshold. Candidates: 0 and
    # every observed |dm|; k = number of pairs tied at that threshold.
    ends = np.r_[np.nonzero(np.diff(d_sorted))[0] + 1, len(d_sorted)]
    thresholds = np.r_[0.0, d_sorted[ends - 1]]
    k = np.r_[np.searchsorted(d_sorted, 0.0, side="right"), ends]
    acc = cum_tie[k] + total_con - cum_con[k]
    best = int(np.argmax(acc))
    eps = float(thresholds[best])

    tied = d <= eps
    agree = (h_tie & tied) | (concordant & ~tied)
    per_item_agree = np.bincount(p.item, weights=agree.astype(float), minlength=p.n_items)
    return float(acc[best]), eps, per_item_agree, pairs_per_item


def kendall_tau_b(H, M) -> float:
    from scipy.stats import kendalltau

    h = np.asarray(H, dtype=float).ravel()
    m = np.asarray(M, dtype=float).ravel()
    ok = ~np.isnan(h) & ~np.isnan(m)
    if ok.sum() < 3:
        return float("nan")
    return float(kendalltau(h[ok], m[ok], variant="b").statistic)


def system_scores(H, M) -> tuple[np.ndarray, np.ndarray]:
    """Mean human and reward score per system over the cells both have."""
    H = np.asarray(H, dtype=float)
    M = np.asarray(M, dtype=float)
    ok = ~np.isnan(H) & ~np.isnan(M)
    with np.errstate(invalid="ignore"):
        hs = np.where(ok, H, 0).sum(0) / ok.sum(0)
        ms = np.where(ok, M, 0).sum(0) / ok.sum(0)
    return hs, ms


def system_accuracy(H, M) -> tuple[float, int]:
    hs, ms = system_scores(H, M)
    keep = ~np.isnan(hs) & ~np.isnan(ms)
    hs, ms = hs[keep], ms[keep]
    iu, ju = np.triu_indices(len(hs), k=1)
    dh, dm = hs[iu] - hs[ju], ms[iu] - ms[ju]
    pref = dh != 0
    if not pref.any():
        return float("nan"), 0
    credit = np.where(dm[pref] == 0, 0.5, (np.sign(dm[pref]) == np.sign(dh[pref])).astype(float))
    return float(credit.mean()), int(pref.sum())


def system_pearson(H, M) -> float:
    hs, ms = system_scores(H, M)
    keep = ~np.isnan(hs) & ~np.isnan(ms)
    if keep.sum() < 3:
        return float("nan")
    return float(np.corrcoef(hs[keep], ms[keep])[0, 1])


# --- bootstrap over items -----------------------------------------------------------

def bootstrap_indices(n_items: int, n_boot: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.integers(0, n_items, size=(n_boot, n_items))


def bootstrap_ratio(numer: np.ndarray, denom: np.ndarray, idx: np.ndarray) -> np.ndarray:
    """Micro-averaged ratio sum(numer)/sum(denom) for each bootstrap resample of items."""
    with np.errstate(invalid="ignore", divide="ignore"):
        return numer[idx].sum(1) / denom[idx].sum(1)


def bootstrap_macro(per_item_agree: np.ndarray, per_item_pairs: np.ndarray, idx: np.ndarray) -> np.ndarray:
    """Macro-averaged (over items with pairs) accuracy for each resample."""
    has = per_item_pairs > 0
    with np.errstate(invalid="ignore", divide="ignore"):
        acc_item = np.where(has, per_item_agree / np.where(has, per_item_pairs, 1), 0.0)
        return acc_item[idx].sum(1) / has[idx].sum(1)


def ci(samples: np.ndarray, level: float = 0.95) -> tuple[float, float]:
    samples = samples[~np.isnan(samples)]
    if len(samples) == 0:
        return float("nan"), float("nan")
    a = (1 - level) / 2
    return float(np.quantile(samples, a)), float(np.quantile(samples, 1 - a))


def paired_difference(a: np.ndarray, b: np.ndarray) -> dict:
    """Bootstrap difference a - b from samples drawn with the same indices."""
    diff = a - b
    diff = diff[~np.isnan(diff)]
    lo, hi = ci(diff)
    # two-sided p-value: how often the resampled difference crosses zero
    p = 2 * min((diff <= 0).mean(), (diff >= 0).mean()) if len(diff) else float("nan")
    return {"diff": float(np.mean(diff)) if len(diff) else float("nan"),
            "lo": lo, "hi": hi, "p": float(min(1.0, p))}
