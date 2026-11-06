"""End-to-end Phase 1 pipeline on the fake dataset, using chrF/BLEU (no GPU or downloads)."""

import json
import os
import subprocess
import sys

import pytest

pytest.importorskip("mt_metrics_eval")

ROOT = os.path.join(os.path.dirname(__file__), "..")


def run(*args):
    proc = subprocess.run([sys.executable, *args], cwd=ROOT, capture_output=True, text=True)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    return proc.stdout


def test_phase1_pipeline(fake_mtme, tmp_path):
    run_dir = str(tmp_path / "run")
    out = run("scripts/p1_build_inputs.py", "--run", run_dir, "--testsets", "wmt22",
              "--mtme-path", fake_mtme, "--probe-items", "8")
    assert "23 scored translations" in out

    run("scripts/p1_score.py", "--run", run_dir, "--rewards", "chrf", "bleu")
    # resuming scores nothing new
    again = run("scripts/p1_score.py", "--run", run_dir, "--rewards", "chrf")
    assert "0 rows to score" in again

    summary = run("scripts/p1_analyze.py", "--run", run_dir, "--bootstrap", "200")
    assert "Within-source accuracy" in summary
    report = os.path.join(run_dir, "report")
    for name in ("summary.md", "reward_benchmark.csv", "blind_spots.csv", "blind_spots.png", "results.json"):
        assert os.path.exists(os.path.join(report, name)), name

    results = json.load(open(os.path.join(report, "results.json")))
    pooled = results["benchmark"]["chrf"]["pooled"]["within_acc"]
    assert 0.5 < pooled <= 1.0  # the fake systems degrade monotonically, chrF should mostly agree
    # our chrF reproduces the stored chrF scores exactly
    chk = [s for s in results["sanity"] if s["reward"] == "chrf"]
    assert chk and chk[0]["pearson"] > 0.999 and chk[0]["max_abs_diff"] < 1e-6
    # acc_eq agrees with mt-metrics-eval's implementation on real loader output
    assert all(c["abs_diff"] < 1e-9 for c in results["acc_eq_crosscheck"])
    # untranslated output should never beat the reference under chrF
    unt = [p for p in results["probes"] if p["reward"] == "chrf" and p["category"] == "untranslated"]
    assert unt[0]["detection"] == 1.0
