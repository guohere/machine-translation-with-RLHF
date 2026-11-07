import numpy as np
import pytest

from mtrlhf.meta_eval import analysis


def rows_for(testset, n_items, flip_every=None):
    """Two systems per item; human prefers sysA. The reward agrees except on every k-th item."""
    rows = []
    for i in range(n_items):
        for sys, h in (("sysA", 0.0), ("sysB", -1.0)):
            rows.append({"key": f"{testset}|{sys}|{i}", "testset": testset, "item": f"{testset}|{i}",
                         "sys": sys, "human": h, "shipped": {}})
    scores = {}
    for r in rows:
        i = int(r["item"].split("|")[1])
        good = r["sys"] == "sysA"
        if flip_every and i % flip_every == 0:
            good = not good
        scores[r["key"]] = 1.0 if good else 0.0
    return rows, scores


def test_pooled_accuracy_uses_only_pool_testsets():
    a_rows, a_scores = rows_for("A", 40)                  # reward always right
    b_rows, b_scores = rows_for("B", 40, flip_every=2)    # right half the time
    rows = a_rows + b_rows
    sc = {**a_scores, **b_scores}
    values = {"r": lambda r: sc.get(r["key"])}

    res, _ = analysis.benchmark(rows, values, n_boot=50, pool=["A"])
    assert res["r"]["pooled"]["within_acc"] == pytest.approx(1.0)
    assert res["r"]["pooled"]["testsets"] == ["A"]
    assert res["r"]["per_testset"]["B"]["within_acc"] == pytest.approx(0.5)

    res_all, _ = analysis.benchmark(rows, values, n_boot=50)
    assert res_all["r"]["pooled"]["within_acc"] == pytest.approx(0.75)


def test_choose_rewards_prefers_other_family_for_eval():
    res = {"cometkiwi22": {"pooled": {"within_acc": 0.62}}, "metricx24_qe": {"pooled": {"within_acc": 0.58}},
           "comet22": {"pooled": {"within_acc": 0.66}}, "metricx24_ref": {"pooled": {"within_acc": 0.64}},
           "chrf": {"pooled": {"within_acc": 0.70}}}
    cmp = [{"a": "cometkiwi22", "b": "metricx24_qe", "diff": 0.04, "lo": 0.01, "hi": 0.07, "p": 0.01}]
    choice = analysis.choose_rewards(res, cmp, {}, list(res))
    assert choice["r_train"] == "cometkiwi22"
    assert choice["r_eval"] == "metricx24_ref"  # not comet22 (same family), not chrf (surface)


def test_choose_rewards_ties_go_to_faster_reward():
    res = {"cometkiwi22": {"pooled": {"within_acc": 0.61}}, "metricx24_qe": {"pooled": {"within_acc": 0.60}}}
    cmp = [{"a": "cometkiwi22", "b": "metricx24_qe", "diff": 0.01, "lo": -0.02, "hi": 0.04, "p": 0.4}]
    metas = {"cometkiwi22": {"inputs_per_second": 20}, "metricx24_qe": {"inputs_per_second": 60}}
    assert analysis.choose_rewards(res, cmp, metas, list(res))["r_train"] == "metricx24_qe"


def test_choice_note_reports_positive_margin_for_winner():
    # the comparison is stored as (loser - winner), as in the real Phase 1 run
    res = {"cometkiwi22": {"pooled": {"within_acc": 0.62}}, "metricx24_qe": {"pooled": {"within_acc": 0.64}}}
    cmp = [{"a": "cometkiwi22", "b": "metricx24_qe", "diff": -0.022, "lo": -0.034, "hi": -0.012, "p": 0.0}]
    choice = analysis.choose_rewards(res, cmp, {}, list(res))
    assert choice["r_train"] == "metricx24_qe"
    assert "by 2.2 points (95% CI [1.2, 3.4])" in choice["notes"][0]
