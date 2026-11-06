"""Phase 1 analysis: turn scores into the reward benchmark, blind-spot profile and choice."""

from __future__ import annotations

import collections
import glob
import os

import numpy as np

from mtrlhf.io import read_json, read_jsonl
from mtrlhf.rewards import REGISTRY

from . import stats

# Stored WMT metric scores that should match our own scoring of the same model.
# exact: True, False, or the set of test sets where the stored scores come from the
# same public checkpoint and settings. WMT22's COMET-22 / COMETKiwi entries are the
# submitted systems (ensembles); the WMT23/24 baselines are the released single models.
SHIPPED_EQUIVALENTS = {
    "cometkiwi22": [("CometKiwi-src", True), ("COMETKiwi-src", False)],
    "comet22": [("COMET-{ref}", True), ("COMET-22-{ref}", {"wmt24"})],
    "chrf": [("chrF-{ref}", True)],
    "bleu": [("BLEU-{ref}", False)],
    "metricx24_qe": [("MetricX-24-Hybrid-QE-src", False)],
    "metricx24_ref": [("MetricX-24-Hybrid-{ref}", False)],
}

PROBE_ORDER = ["omission", "addition", "number", "named_entity", "negation",
               "untranslated", "wrong_language", "word_order", "repetition"]


# --- loading ------------------------------------------------------------------------

def load_scores(run: str, sub: str = "scores") -> tuple[dict[str, dict[str, float]], dict[str, dict]]:
    scores, metas = {}, {}
    for path in sorted(glob.glob(os.path.join(run, sub, "*.jsonl"))):
        name = os.path.basename(path)[: -len(".jsonl")]
        scores[name] = {r["key"]: r["score"] for r in read_jsonl(path)}
        meta_path = os.path.join(run, sub, f"{name}.meta.json")
        metas[name] = read_json(meta_path) if os.path.exists(meta_path) else {}
    return scores, metas


# --- matrices -----------------------------------------------------------------------

def build_matrices(rows: list[dict], value) -> dict[str, tuple[np.ndarray, np.ndarray, list[str], list[str]]]:
    """Per test set: (H, M, items, systems) with NaN for missing cells.

    `value(row)` returns the reward score for a row or None.
    """
    by_ts = collections.defaultdict(list)
    for r in rows:
        by_ts[r["testset"]].append(r)
    out = {}
    for ts, rs in sorted(by_ts.items()):
        items = sorted({r["item"] for r in rs}, key=lambda x: (len(x), x))
        systems = sorted({r["sys"] for r in rs})
        ii = {k: i for i, k in enumerate(items)}
        si = {k: i for i, k in enumerate(systems)}
        H = np.full((len(items), len(systems)), np.nan)
        M = np.full_like(H, np.nan)
        for r in rs:
            H[ii[r["item"]], si[r["sys"]]] = r["human"]
            v = value(r)
            if v is not None:
                M[ii[r["item"]], si[r["sys"]]] = v
        out[ts] = (H, M, items, systems)
    return out


# --- benchmark ----------------------------------------------------------------------

def benchmark(rows: list[dict], values: dict[str, callable], n_boot: int = 1000, seed: int = 0,
              pool: list[str] | None = None):
    """Agreement statistics for each scorer in `values` (name -> row -> score).

    `pool` lists the test sets combined into the headline ("pooled") within-source
    accuracy, which also drives the reward choice. Other test sets are reported
    per test set only. None pools all test sets.
    """
    testsets = sorted({r["testset"] for r in rows})
    pool_sets = [ts for ts in testsets if pool is None or ts in pool]
    results: dict[str, dict] = {}
    pooled_samples: dict[str, np.ndarray] = {}
    idx_by_ts: dict[str, np.ndarray] = {}
    pooled_idx = None

    for name, value in values.items():
        mats = build_matrices(rows, value)
        res = {"per_testset": {}}
        pooled_correct, pooled_count = [], []
        for ts in testsets:
            H, M, items, systems = mats[ts]
            if np.isnan(M).all():
                continue
            if ts not in idx_by_ts:
                idx_by_ts[ts] = stats.bootstrap_indices(len(items), n_boot, seed)
            idx = idx_by_ts[ts]
            p = stats.item_pairs(H, M)
            correct, count = stats.within_source_counts(p)
            acc = correct.sum() / count.sum() if count.sum() else float("nan")
            lo, hi = stats.ci(stats.bootstrap_ratio(correct, count, idx))
            acc_eq, eps, agree_i, pairs_i = stats.tie_calibrated_accuracy(H, M)
            eq_lo, eq_hi = stats.ci(stats.bootstrap_macro(agree_i, pairs_i, idx))
            sys_acc, sys_n = stats.system_accuracy(H, M)
            res["per_testset"][ts] = {
                "within_acc": float(acc), "within_lo": lo, "within_hi": hi, "n_pairs": int(count.sum()),
                "acc_eq": acc_eq, "acc_eq_lo": eq_lo, "acc_eq_hi": eq_hi, "tie_threshold": eps,
                "kendall_tau_b": stats.kendall_tau_b(H, M),
                "sys_acc": sys_acc, "sys_pairs": sys_n, "sys_pearson": stats.system_pearson(H, M),
                "n_items": len(items), "n_systems": len(systems),
            }
            if ts in pool_sets:
                pooled_correct.append(correct)
                pooled_count.append(count)
        if pooled_count and all(ts in res["per_testset"] for ts in pool_sets):
            c, n = np.concatenate(pooled_correct), np.concatenate(pooled_count)
            if pooled_idx is None or pooled_idx.shape[1] != len(c):
                pooled_idx = stats.bootstrap_indices(len(c), n_boot, seed + 1)
            samples = stats.bootstrap_ratio(c, n, pooled_idx)
            lo, hi = stats.ci(samples)
            res["pooled"] = {"within_acc": float(c.sum() / n.sum()), "within_lo": lo, "within_hi": hi,
                             "n_pairs": int(n.sum()), "testsets": pool_sets}
            pooled_samples[name] = samples
        results[name] = res
    return results, pooled_samples


def paired_comparisons(samples: dict[str, np.ndarray], names: list[str]) -> list[dict]:
    out = []
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            if a in samples and b in samples:
                out.append({"a": a, "b": b, **stats.paired_difference(samples[a], samples[b])})
    return out


# --- sanity checks ------------------------------------------------------------------

def sanity_checks(rows: list[dict], scores: dict[str, dict[str, float]], std_refs: dict[str, str]) -> list[dict]:
    """Correlate our scores with the stored WMT scores of the same metric."""
    from scipy.stats import pearsonr, spearmanr

    out = []
    for name, cands in SHIPPED_EQUIVALENTS.items():
        if name not in scores:
            continue
        for ts in sorted({r["testset"] for r in rows}):
            for pattern, exact in cands:
                stored = pattern.format(ref=std_refs.get(ts, "refA"))
                pairs = [(scores[name][r["key"]], r["shipped"][stored]) for r in rows
                         if r["testset"] == ts and r["key"] in scores[name] and stored in r.get("shipped", {})]
                if len(pairs) < 10:
                    continue
                x, y = np.array(pairs).T
                is_exact = exact if isinstance(exact, bool) else ts in exact
                out.append({"reward": name, "testset": ts, "stored": stored, "n": len(pairs),
                            "pearson": float(pearsonr(x, y)[0]), "spearman": float(spearmanr(x, y)[0]),
                            "max_abs_diff": float(np.max(np.abs(x - y))), "expected_identical": is_exact})
    return out


def crosscheck_acc_eq(rows: list[dict], values: dict[str, callable]) -> list[dict]:
    """Recompute acc_eq with mt-metrics-eval's own implementation and compare."""
    try:
        from mtrlhf.meta_eval.wmt import _import_mtme_data

        _import_mtme_data()
        from mt_metrics_eval import stats as mtme_stats
    except Exception as e:  # pragma: no cover - only when mt-metrics-eval is missing
        return [{"error": f"mt-metrics-eval not importable: {e}"}]
    out = []
    for name, value in values.items():
        for ts, (H, M, items, systems) in build_matrices(rows, value).items():
            if np.isnan(M).all():
                continue
            ours, _, _, _ = stats.tie_calibrated_accuracy(H, M)
            gold = [None if np.isnan(v) else float(v) for v in H.T.ravel()]
            metric = [None if np.isnan(v) else float(v) for v in M.T.ravel()]
            theirs = mtme_stats.KendallWithTiesOpt(gold, metric, variant="acc23", num_sys=len(systems),
                                                   average_by="item", sample_rate=1.0)[0]
            out.append({"reward": name, "testset": ts, "ours": ours, "mt_metrics_eval": float(theirs),
                        "abs_diff": abs(ours - float(theirs))})
    return out


# --- probes -------------------------------------------------------------------------

def probe_table(probes: list[dict], scores: dict[str, dict[str, float]], seg_rows: list[dict],
                n_boot: int = 1000, seed: int = 0) -> list[dict]:
    """Detection rate per reward and category: P(score(original) > score(perturbed))."""
    out = []
    by_item = collections.defaultdict(dict)
    for p in probes:
        by_item[p["item"]][p["category"]] = p["key"]
    rng = np.random.default_rng(seed)
    for name, sc in scores.items():
        seg_vals = np.array([sc[r["key"]] for r in seg_rows if r["key"] in sc])
        scale = float(np.std(seg_vals)) if len(seg_vals) > 1 else float("nan")
        for cat in PROBE_ORDER:
            credit, drops = [], []
            for cats in by_item.values():
                if cat not in cats or "original" not in cats:
                    continue
                o, q = sc.get(cats["original"]), sc.get(cats[cat])
                if o is None or q is None:
                    continue
                credit.append(1.0 if o > q else 0.5 if o == q else 0.0)
                drops.append(o - q)
            if not credit:
                out.append({"reward": name, "category": cat, "n": 0, "detection": float("nan"),
                            "lo": float("nan"), "hi": float("nan"), "mean_drop_sd": float("nan")})
                continue
            credit = np.array(credit)
            idx = rng.integers(0, len(credit), size=(n_boot, len(credit)))
            lo, hi = stats.ci(credit[idx].mean(1))
            out.append({"reward": name, "category": cat, "n": len(credit), "detection": float(credit.mean()),
                        "lo": lo, "hi": hi, "mean_drop_sd": float(np.mean(drops) / scale) if scale else float("nan")})
    return out


def plot_blind_spots(table: list[dict], names: list[str], path: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

    cats = [c for c in PROBE_ORDER if any(t["category"] == c and t["n"] for t in table)]
    if not cats or not names:
        return
    V = np.full((len(names), len(cats)), np.nan)
    for t in table:
        if t["reward"] in names and t["category"] in cats and t["n"]:
            V[names.index(t["reward"]), cats.index(t["category"])] = 100 * t["detection"]

    # Diverging around chance (50%): red = prefers the damaged translation, gray = chance,
    # blue = detects the damage. Values are printed in every cell.
    cmap = LinearSegmentedColormap.from_list("div", ["#e34948", "#f0efec", "#2a78d6"])
    cmap.set_bad("#e6e5e1")
    surface, ink, muted = "#fcfcfb", "#1f1f1e", "#6b6a66"
    fig, ax = plt.subplots(figsize=(1.0 * len(cats) + 2.6, 0.55 * len(names) + 1.9), dpi=150)
    fig.patch.set_facecolor(surface)
    ax.set_facecolor(surface)
    mesh = ax.pcolormesh(np.ma.masked_invalid(V), cmap=cmap, norm=TwoSlopeNorm(50, 0, 100),
                         edgecolors=surface, linewidth=2)
    for i in range(len(names)):
        for j in range(len(cats)):
            v = V[i, j]
            rgb = np.array(cmap(mesh.norm(v))[:3]) if not np.isnan(v) else np.array([0.9, 0.9, 0.9])
            lum = 0.2126 * rgb[0] + 0.7152 * rgb[1] + 0.0722 * rgb[2]
            ax.text(j + 0.5, i + 0.5, "–" if np.isnan(v) else f"{v:.0f}", ha="center", va="center",
                    fontsize=9, color="white" if lum < 0.45 else ink)
    ax.set_xticks(np.arange(len(cats)) + 0.5, [c.replace("_", " ") for c in cats],
                  rotation=30, ha="right", fontsize=9, color=ink)
    ax.set_yticks(np.arange(len(names)) + 0.5, names, fontsize=9, color=ink)
    ax.invert_yaxis()
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    cb = fig.colorbar(mesh, ax=ax, fraction=0.04, pad=0.02, ticks=[0, 25, 50, 75, 100])
    cb.outline.set_visible(False)
    cb.ax.tick_params(labelsize=8, colors=muted, length=0)
    cb.set_label("detection rate (%) · 50 = chance", fontsize=8, color=muted)
    ax.set_title("How often each reward scores the intact translation above the damaged one",
                 fontsize=10, color=ink, loc="left", pad=10)
    fig.tight_layout()
    fig.savefig(path, facecolor=surface)
    plt.close(fig)


# --- choice of R_train / R_eval -------------------------------------------------------

def choose_rewards(results: dict, comparisons: list[dict], metas: dict, names: list[str]) -> dict:
    """Apply the pre-registered decision rule from docs/workflow.md §4.3."""

    def pooled(n):
        return results.get(n, {}).get("pooled", {}).get("within_acc", float("nan"))

    def info(n):
        return REGISTRY.get(n, {})

    qe = sorted([n for n in names if n in REGISTRY and not info(n)["needs_reference"]
                 and not np.isnan(pooled(n))], key=pooled, reverse=True)
    out: dict = {"notes": []}
    if not qe:
        out["notes"].append("No reference-free reward was scored, so R_train cannot be chosen yet.")
        return out
    r_train = qe[0]
    if len(qe) > 1:
        runner = qe[1]
        cmp = next((c for c in comparisons if {c["a"], c["b"]} == {r_train, runner}), None)
        if cmp and cmp["p"] >= 0.05:
            speed = {n: (metas.get(n) or {}).get("inputs_per_second") or 0 for n in (r_train, runner)}
            faster = max(speed, key=speed.get)
            out["notes"].append(
                f"{r_train} and {runner} are not significantly different on within-source accuracy "
                f"(p = {cmp['p']:.2f}); choosing the faster one ({faster}).")
            r_train = faster
        elif cmp:
            out["notes"].append(f"{r_train} beats {runner} on within-source accuracy "
                                f"(diff {cmp['diff']:+.3f}, 95% CI [{cmp['lo']:+.3f}, {cmp['hi']:+.3f}]).")
    out["r_train"] = r_train
    fam = info(r_train)["family"]
    evals = sorted([n for n in names if n in REGISTRY and info(n)["needs_reference"]
                    and info(n)["family"] not in (fam, "surface") and not np.isnan(pooled(n))],
                   key=pooled, reverse=True)
    if evals:
        out["r_eval"] = evals[0]
    else:
        same = sorted([n for n in names if n in REGISTRY and info(n)["needs_reference"]
                       and info(n)["family"] != "surface"], key=pooled, reverse=True)
        if same:
            out["r_eval"] = same[0]
            out["notes"].append("No reference-based reward from another family was scored; "
                                f"falling back to {same[0]} (same family as R_train).")
    return out
