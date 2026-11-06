import pytest

pytest.importorskip("mt_metrics_eval")

from mtrlhf.meta_eval import wmt  # noqa: E402

from .conftest import MQM  # noqa: E402


def test_load_mqm_rows(fake_mtme):
    rows, meta = wmt.load_mqm_rows("wmt22", "en-de", mtme_path=fake_mtme)
    # outlier M2M100 and the human system refB are excluded; one sysB score is None
    assert meta["systems"] == ["sysA", "sysB", "sysC"]
    assert set(meta["excluded_systems"]) == {"M2M100_1.2B-B4", "refB"}
    assert len(rows) == 8 * 3 - 1
    assert meta["std_ref"] == "refA" and meta["alt_ref"] == "refB"
    assert not meta["gold_flipped"]
    r = next(r for r in rows if r["sys"] == "sysC" and r["seg_id"] == 2)
    assert r["human"] == MQM["sysC"][2]
    assert r["item"] == "wmt22|en-de|2" and r["domain"] == "news"
    assert set(r["shipped"]) == {"COMETKiwi-src", "chrF-refA"}


def test_human_systems_can_be_included(fake_mtme):
    rows, meta = wmt.load_mqm_rows("wmt22", "en-de", mtme_path=fake_mtme, include_human_systems=True)
    assert "refB" in meta["systems"]
