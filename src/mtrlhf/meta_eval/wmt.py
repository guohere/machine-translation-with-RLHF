"""Load WMT system outputs and expert MQM scores from Google's mt-metrics-eval.

Each returned row is one (system, segment) translation with its human score:

    {"key", "testset", "lp", "item", "seg_id", "sys", "doc", "domain",
     "src", "hyp", "ref", "alt_ref", "human", "shipped": {metric_name: score}}

`human` is oriented so that higher is better. `item` identifies the source
segment; all rows with the same item are translations of the same source, which
is what the within-source statistics compare. `shipped` holds the metric scores
that ship with mt-metrics-eval (e.g. the WMT23 CometKiwi baseline); they are
used to check that our own scoring reproduces them.
"""

from __future__ import annotations

import os
import sys
import tarfile
import types
import urllib.request

DEFAULT_TGZ = "https://data.statmt.org/wmt26/mt-metrics-eval-v2.tgz"
ARCHIVE_ROOT = "mt-metrics-eval-v2"

# Stored metric scores worth loading (base names as used by WMT22-24).
SHIPPED_BASENAMES = [
    "COMETKiwi", "CometKiwi", "COMET-22", "COMET", "BLEURT-20",
    "MetricX-23", "MetricX-23-QE", "MetricX-24-Hybrid", "MetricX-24-Hybrid-QE",
    "XCOMET-Ensemble", "XCOMET-QE-Ensemble", "XCOMET", "XCOMET-QE",
    "GEMBA-MQM", "chrF", "BLEU",
]


def _import_mtme_data():
    """Import mt_metrics_eval.data without requiring Apache Beam.

    data.py imports apache_beam at module level, but only its Beam-based
    significance helpers use it. Those are not used here, so a placeholder
    module is registered when Beam is not installed.
    """
    try:
        import apache_beam  # noqa: F401
    except ImportError:
        sys.modules.setdefault("apache_beam", types.ModuleType("apache_beam"))
    from mt_metrics_eval import data

    return data


def default_mtme_path() -> str:
    return os.path.join(os.path.expanduser("~"), ".mt-metrics-eval", ARCHIVE_ROOT)


def download_mtme(testsets: list[str], lps: list[str] | None = None, dest_root: str | None = None,
                  url: str | None = None, keep_metric_scores: bool = True) -> str:
    """Stream the mt-metrics-eval archive and extract only the needed test sets.

    The full archive covers every WMT year. Streaming it and keeping only
    `testsets` (and only `lps` for the bulky metric-score folders) saves
    many GB of disk on Kaggle/Colab. Returns the data path to pass to EvalSet.
    """
    if url is None:
        try:
            url = _import_mtme_data().TGZ
        except Exception:
            url = DEFAULT_TGZ
    dest_root = dest_root or os.path.dirname(default_mtme_path())
    os.makedirs(dest_root, exist_ok=True)
    wanted = tuple(f"{ARCHIVE_ROOT}/{t}/" for t in testsets)
    n = 0
    with urllib.request.urlopen(url) as resp, tarfile.open(fileobj=resp, mode="r|gz") as tar:
        for member in tar:
            name = member.name.lstrip("./")
            if not name.startswith(wanted):
                continue
            if name.endswith(".rating"):  # character-level error spans, not needed
                continue
            if "/metric-scores/" in name:
                if not keep_metric_scores:
                    continue
                if lps and not any(f"/metric-scores/{lp}/" in name or name.endswith(f"/metric-scores/{lp}")
                                   or name.endswith("/metric-scores") for lp in lps):
                    continue
            member.name = name
            try:
                tar.extract(member, dest_root, filter="data")  # safe extraction (Python 3.12+)
            except TypeError:
                tar.extract(member, dest_root)
            n += 1
    if n == 0:
        raise RuntimeError(f"No files for {testsets} found in {url}")
    return os.path.join(dest_root, ARCHIVE_ROOT)


def load_eval_set(testset: str, lp: str, mtme_path: str | None = None, shipped: bool = True):
    data = _import_mtme_data()
    return data.EvalSet(testset, lp, read_stored_metric_scores=shipped, path=mtme_path)


def load_mqm_rows(
    testset: str,
    lp: str = "en-de",
    mtme_path: str | None = None,
    include_human_systems: bool = False,
    shipped: bool = True,
) -> tuple[list[dict], dict]:
    """Rows for every (system, segment) with an expert MQM score."""
    evs = load_eval_set(testset, lp, mtme_path, shipped)
    gold_name = evs.StdHumanScoreName("seg")
    gold = evs.Scores("seg", gold_name)
    if gold is None:
        raise ValueError(f"{testset} {lp}: no segment-level {gold_name} scores")

    std_ref = evs.std_ref
    alt_refs = sorted(r for r in evs.ref_names if r != std_ref)
    alt_ref = alt_refs[0] if alt_refs else None
    excluded = set(evs.outlier_sys_names) | {std_ref}
    if not include_human_systems:
        excluded |= set(evs.human_sys_names)
    systems = sorted(s for s in evs.sys_outputs if s in gold and s not in excluded)

    # mt-metrics-eval stores MQM as negative error points (higher is better).
    # If a set stores raw error points instead, flip it.
    values = [v for s in systems for v in gold[s] if v is not None]
    flip = bool(values) and min(values) >= 0 and max(values) > 0

    shipped_scores = {}
    if shipped:
        names = set(evs.metric_names)
        for base in SHIPPED_BASENAMES:
            for suffix in (std_ref, "src"):
                name = f"{base}-{suffix}"
                if name in names:
                    scores = evs.Scores("seg", name)
                    if scores:
                        shipped_scores[name] = scores

    docs = evs.DocsPerSeg()
    domains = evs.DomainsPerSeg()
    src = evs.src
    ref = evs.all_refs[std_ref]
    alt = evs.all_refs[alt_ref] if alt_ref else None

    rows = []
    for s in systems:
        hyps = evs.sys_outputs[s]
        for i, g in enumerate(gold[s]):
            if g is None or hyps[i] is None or src[i] is None or ref[i] is None:
                continue
            row = {
                "key": f"{testset}|{lp}|{s}|{i}",
                "testset": testset,
                "lp": lp,
                "item": f"{testset}|{lp}|{i}",
                "seg_id": i,
                "sys": s,
                "doc": docs[i],
                "domain": domains[i],
                "src": src[i],
                "hyp": hyps[i],
                "ref": ref[i],
                "alt_ref": alt[i] if alt is not None else None,
                "human": -g if flip else g,
                "shipped": {},
            }
            for name, scores in shipped_scores.items():
                if s in scores and scores[s][i] is not None:
                    row["shipped"][name] = scores[s][i]
            rows.append(row)

    meta = {
        "testset": testset,
        "lp": lp,
        "gold": gold_name,
        "gold_flipped": flip,
        "std_ref": std_ref,
        "alt_ref": alt_ref,
        "systems": systems,
        "excluded_systems": sorted(excluded & set(evs.sys_outputs)),
        "n_rows": len(rows),
        "n_items": len({r["item"] for r in rows}),
        "shipped_metrics": sorted(shipped_scores),
    }
    return rows, meta


def other_language_references(testset: str, src_lp: str = "en-de", mtme_path: str | None = None,
                              candidates: tuple[str, ...] = ("en-cs", "en-hr", "en-uk", "en-ru",
                                                             "en-zh", "en-ja")) -> dict[str, tuple[str, str]]:
    """Map English source -> (lp, reference) from other en-X pairs of the same test set.

    Used for the wrong-target-language probe. Pairs that share no sources with
    `src_lp` simply contribute nothing.
    """
    out: dict[str, tuple[str, str]] = {}
    for lp in candidates:
        if lp == src_lp:
            continue
        try:
            evs = load_eval_set(testset, lp, mtme_path, shipped=False)
        except Exception:
            continue
        ref = evs.all_refs[evs.std_ref]
        for s, r in zip(evs.src, ref):
            if s and r and s not in out:
                out[s] = (lp, r)
    return out
