import os
import tarfile

import pytest

pytest.importorskip("mt_metrics_eval")

from mtrlhf.meta_eval import wmt  # noqa: E402


def test_download_extracts_only_requested_test_sets(fake_mtme, tmp_path):
    # Package the fake data like the real archive, plus an unrelated test set.
    os.makedirs(os.path.join(fake_mtme, "wmt99", "sources"))
    open(os.path.join(fake_mtme, "wmt99", "sources", "en-de.txt"), "w").write("x\n")
    tgz = tmp_path / "mt-metrics-eval-v2.tgz"
    with tarfile.open(tgz, "w:gz") as tar:
        tar.add(fake_mtme, arcname=wmt.ARCHIVE_ROOT)

    path = wmt.download_mtme(["wmt22"], lps=["en-de"], dest_root=str(tmp_path / "dl"), url=tgz.as_uri())
    assert os.path.isdir(os.path.join(path, "wmt22", "metric-scores", "en-de"))
    assert not os.path.exists(os.path.join(path, "wmt99"))
    rows, meta = wmt.load_mqm_rows("wmt22", "en-de", mtme_path=path)
    assert meta["n_rows"] == 23
