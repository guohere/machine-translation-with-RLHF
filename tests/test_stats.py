import numpy as np
import pytest

from mtrlhf.meta_eval import stats

nan = np.nan


def test_within_source_accuracy_counts_only_human_preferences():
    # 2 items x 3 systems
    H = np.array([[0.0, -1.0, -5.0],
                  [-2.0, -2.0, nan]])
    M = np.array([[0.9, 0.5, 0.7],   # pairs: (0,1) ok, (0,2) ok, (1,2) wrong
                  [0.1, 0.3, 0.2]])  # human tie only -> no pairs
    acc, n = stats.within_source_accuracy(H, M)
    assert n == 3
    assert acc == pytest.approx(2 / 3)


def test_reward_ties_get_half_credit():
    H = np.array([[0.0, -1.0]])
    M = np.array([[0.5, 0.5]])
    assert stats.within_source_accuracy(H, M)[0] == pytest.approx(0.5)


def _brute_force_acc_eq(H, M):
    p = stats.item_pairs(H, M)
    rows = sorted(set(p.item.tolist()))
    best = -1
    for eps in [0.0] + sorted(set(np.abs(p.dm).tolist())):
        accs = []
        for r in rows:
            m = p.item == r
            dh, dm = p.dh[m], p.dm[m]
            tie_m = np.abs(dm) <= eps
            agree = ((dh == 0) & tie_m) | ((dh != 0) & ~tie_m & (np.sign(dh) == np.sign(dm)))
            accs.append(agree.mean())
        best = max(best, float(np.mean(accs)))
    return best


def test_tie_calibrated_accuracy_matches_brute_force():
    rng = np.random.default_rng(1)
    H = rng.integers(-5, 1, size=(30, 6)).astype(float)  # many human ties
    M = np.round(H + rng.normal(0, 1.5, size=H.shape), 1)  # some reward ties
    H[rng.random(H.shape) < 0.1] = nan
    acc, eps, agree, pairs = stats.tie_calibrated_accuracy(H, M)
    assert acc == pytest.approx(_brute_force_acc_eq(H, M))
    # per-item arrays reproduce the macro average at the chosen threshold
    has = pairs > 0
    assert np.mean(agree[has] / pairs[has]) == pytest.approx(acc)


def test_tie_calibrated_accuracy_matches_mt_metrics_eval():
    pytest.importorskip("mt_metrics_eval")
    from mtrlhf.meta_eval.wmt import _import_mtme_data

    _import_mtme_data()
    from mt_metrics_eval import stats as mtme_stats

    rng = np.random.default_rng(7)
    n_items, n_sys = 40, 5
    H = rng.integers(-6, 1, size=(n_items, n_sys)).astype(float)
    M = H + rng.normal(0, 2.0, size=H.shape)
    H[rng.random(H.shape) < 0.15] = nan
    ours = stats.tie_calibrated_accuracy(H, M)[0]
    gold = [None if np.isnan(v) else float(v) for v in H.T.ravel()]
    metric = [float(v) for v in M.T.ravel()]
    theirs = mtme_stats.KendallWithTiesOpt(gold, metric, variant="acc23", num_sys=n_sys,
                                           average_by="item", sample_rate=1.0)[0]
    assert ours == pytest.approx(theirs, abs=1e-12)


def test_system_accuracy_and_kendall():
    H = np.array([[0.0, -1.0, -2.0], [0.0, -2.0, -3.0]])
    M = np.array([[3.0, 2.0, 1.0], [3.0, 1.0, 2.0]])
    # system means: human (0, -1.5, -2.5), reward (3, 1.5, 1.5): the last pair is a reward tie
    acc, n = stats.system_accuracy(H, M)
    assert n == 3 and acc == pytest.approx(2.5 / 3)
    assert -1 <= stats.kendall_tau_b(H, M) <= 1


def test_bootstrap_ratio_and_paired_difference():
    correct = np.array([3.0, 1.0, 2.0, 0.0])
    count = np.array([3.0, 2.0, 2.0, 1.0])
    idx = stats.bootstrap_indices(4, 500, seed=3)
    a = stats.bootstrap_ratio(correct, count, idx)
    b = stats.bootstrap_ratio(count * 0.5, count, idx)
    lo, hi = stats.ci(a)
    assert 0 <= lo <= correct.sum() / count.sum() <= hi <= 1
    d = stats.paired_difference(a, b)
    assert d["diff"] > 0 and 0 <= d["p"] <= 1
