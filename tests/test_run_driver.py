"""The Phase 2 driver decides what to run from what is already saved (dry runs, no GPU)."""

import json
import os
import re
import subprocess
import sys

import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SAMPLING = {"temperature": 0.8, "top_p": 1.0}
SETS = ["flores_devtest.en-de", "wmt24.en-de", "flores_devtest.de-en", "flores_devtest.en-zh", "flores_dev.en-de"]


def dry_run(cwd, *sets):
    env = {k: v for k, v in os.environ.items() if k not in ("MTRLHF_SAMPLING_FILE", "MTRLHF_HUB_REPO", "GH_TOKEN")}
    proc = subprocess.run([sys.executable, f"{ROOT}/scripts/p2_run.py", "--plan", f"{ROOT}/configs/p2/run.yaml",
                           "--dry-run", "--set", "hub_repo=none", *sets],
                          cwd=cwd, capture_output=True, text=True, env=env)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    steps = re.findall(r"^\[dry-run\] ([\w+.-]+):", proc.stdout, re.M)
    return [s for s in steps if s != "save"], proc.stdout


def write(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as fh:
        if isinstance(obj, list):
            fh.write("".join(json.dumps(o) + "\n" for o in obj))
        else:
            json.dump(obj, fh)


def adapter(cwd, model, name, **meta):
    write(f"{cwd}/ckpt/{model}/{name}/mtrlhf_meta.json", {"base_model": "x", "parent": None, **meta})


def outputs(cwd, model, system, sets=SETS):
    for ts in sets:
        write(f"{cwd}/results/p2/{model}/outputs/{system}/{ts}.jsonl", [{"id": 1}])


def part_a_done(cwd, model="qwen3-1.7b", rsft_sampling=None):
    cand_meta = {"system": "sft", "n": 8, **SAMPLING, "reward": "metricx24_qe", "seed": 0}
    write(f"{cwd}/data/p2/prompts.jsonl", [{"id": i} for i in range(3)])
    write(f"{cwd}/results/p2/sampling.json", SAMPLING)
    write(f"{cwd}/data/p2/{model}/candidates.jsonl", [{"id": i} for i in range(3)])
    write(f"{cwd}/data/p2/{model}/candidates.meta.json", cand_meta)
    write(f"{cwd}/data/p2/{model}/pairs_stats.json", {"sampling": cand_meta})
    adapter(cwd, model, "sft")
    adapter(cwd, model, "rsft", data_sampling=rsft_sampling or cand_meta)
    adapter(cwd, model, "dpo", data_sampling=cand_meta)
    for s in ("base", "sft", "rsft", "dpo"):
        outputs(cwd, model, s)
    outputs(cwd, model, "opus-mt", [s for s in SETS if s.endswith(".en-de")])  # OPUS-MT is en-de only
    outputs(cwd, model, "sft+bo8", ["flores_devtest.en-de", "wmt24.en-de", "flores_dev.en-de"])
    write(f"{cwd}/results/p2/{model}/outputs/sft+bo8/_sampling.json", {**SAMPLING, "reranker": "metricx24_qe"})


def part_b_done(cwd, model="qwen3-1.7b"):
    for run in ("grpo", "grpo_guarded"):
        adapter(cwd, model, f"{run}-step625")
        outputs(cwd, model, f"{run}-step625")


def test_fresh_start_runs_part_a_only(tmp_path):
    steps, out = dry_run(tmp_path)
    assert "model qwen3-1.7b" in out
    assert steps == ["prepare_data", "smoke_sft", "smoke_sample", "smoke_grpo", "smoke_translate", "sft",
                     "sampling_pilot", "sample", "pairs", "rsft", "dpo", "translate_A", "translate_bo8", "quick_eval"]
    assert "Part B (GRPO) starts in the next run" in out


def test_part_b_runs_once_part_a_has_nothing_left(tmp_path):
    part_a_done(tmp_path)
    steps, out = dry_run(tmp_path)
    assert steps == ["grpo", "grpo_guarded", "quick_eval"] and "Part A complete." in out
    assert "sampling for every method and model: temperature 0.8, top-p 1.0" in out


def test_outdated_rsft_is_retrained_and_retranslated(tmp_path):
    part_a_done(tmp_path, rsft_sampling={"temperature": 1.0, "top_p": 1.0})
    steps, out = dry_run(tmp_path)
    assert steps[:2] == ["rsft", "translate_A"] and "grpo" not in steps
    assert "rsft: trained on samples from another sampling setting" in out
    assert not os.path.exists(tmp_path / "results/p2/qwen3-1.7b/outputs/rsft/flores_devtest.en-de.jsonl")


def test_current_kaggle_state_resamples_and_retrains(tmp_path):
    """The real situation: sft, rsft and dpo exist (rsft/dpo trained on temperature-1.0 samples
    before sampling settings were recorded), no pilot yet, old candidates without settings."""
    write(f"{tmp_path}/data/p2/prompts.jsonl", [{"id": i} for i in range(3)])
    for name in ("sft", "rsft", "dpo"):
        adapter(tmp_path, "qwen3-1.7b", name)
    outputs(tmp_path, "qwen3-1.7b", "base")
    steps, out = dry_run(tmp_path)
    assert steps == ["sampling_pilot", "sample", "pairs", "rsft", "dpo", "translate_A", "translate_bo8", "quick_eval"]
    assert "sampling: chosen by the pilot in this run" in out
    assert "--systems opus-mt sft rsft dpo " in out  # base is complete already


def test_finished_main_model_moves_on_to_the_replication(tmp_path):
    part_a_done(tmp_path)
    part_b_done(tmp_path)
    steps, out = dry_run(tmp_path)
    assert "model gemma3-1b" in out
    # new model: smoke test first; replication trains no rsft and no best-of-8
    assert steps[:4] == ["smoke_sft", "smoke_sample", "smoke_grpo", "smoke_translate"]
    assert "rsft" not in steps and "translate_bo8" not in steps and "dpo" in steps


def test_everything_finished(tmp_path):
    for m in ("qwen3-1.7b", "gemma3-1b"):
        part_a_done(tmp_path, m)
        part_b_done(tmp_path, m)
    steps, out = dry_run(tmp_path)
    assert steps == [] and "All planned models are finished" in out


def test_explicit_overrides(tmp_path):
    part_a_done(tmp_path)
    steps, _ = dry_run(tmp_path, "part=A", "smoke_test=true")
    assert steps[:4] == ["smoke_sft", "smoke_sample", "smoke_grpo", "smoke_translate"] and "grpo" not in steps


def test_opus_mt_is_only_expected_for_english_german(tmp_path, monkeypatch):
    sys.path.insert(0, os.path.join(ROOT, "src"))
    from mtrlhf.pipeline import Runner

    monkeypatch.chdir(tmp_path)
    r = Runner("qwen3-1.7b", "main", hub_repo="none")
    assert r.missing_outputs("opus-mt", SETS) == ["flores_devtest.en-de", "wmt24.en-de", "flores_dev.en-de"]
    assert len(r.missing_outputs("sft", SETS)) == len(SETS)


def test_failed_step_pushes_its_log_and_stops(tmp_path, monkeypatch):
    sys.path.insert(0, os.path.join(ROOT, "src"))
    from mtrlhf.pipeline import Runner, StepFailed

    monkeypatch.chdir(tmp_path)
    saved = []
    r = Runner("qwen3-1.7b", "main", hub_repo="none")
    r.save = lambda message: saved.append(message)
    r.sh("echo fine", "ok_step")
    with pytest.raises(StepFailed, match="bad_step failed"):
        r.sh("echo 'ValueError: boom' >&2; exit 2", "bad_step")
    assert "ValueError: boom" in open("results/p2/qwen3-1.7b/logs/bad_step.log").read()
    assert saved and "bad_step FAILED" in saved[-1] and r.ran == ["ok_step", "bad_step"]
