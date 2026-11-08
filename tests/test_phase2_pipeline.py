"""End-to-end Phase 2 on CPU with a tiny random model: SFT -> sample -> pairs -> RSFT, DPO,
GRPO -> translate (greedy and best-of-N) -> quick summary. chrF stands in for MetricX so
nothing is downloaded; the code paths are the ones used on Kaggle."""

import json
import os
import subprocess
import sys

import pytest

pytest.importorskip("trl")
pytest.importorskip("peft")

from mtrlhf.io import read_jsonl, write_jsonl  # noqa: E402

from .conftest import REF_A, SRC  # noqa: E402
from .tiny_lm import build_tiny_lm  # noqa: E402

ROOT = os.path.join(os.path.dirname(__file__), "..")


ENV: dict = {}  # MTRLHF_SAMPLING_FILE points into the test's temp folder (set by the fixture)


def run(*args, env=None, expect_ok=True):
    proc = subprocess.run([sys.executable, *args], cwd=ROOT, capture_output=True, text=True,
                          env={**os.environ, **ENV, **(env or {})})
    if expect_ok:
        assert proc.returncode == 0, proc.stdout[-3000:] + proc.stderr[-6000:]
    else:
        assert proc.returncode != 0, proc.stdout[-3000:]
    return proc.stdout + proc.stderr


@pytest.fixture(scope="module")
def setup(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("p2")
    base = build_tiny_lm(str(tmp / "tiny"), SRC + REF_A)
    data = tmp / "data"
    pairs = [{"id": f"sft-{i}", "src": s, "ref": r} for i, (s, r) in enumerate(zip(SRC * 4, REF_A * 4))]
    write_jsonl(str(data / "sft_train.jsonl"), pairs)
    write_jsonl(str(data / "prompts.jsonl"), [{"id": f"p-{i}", "src": s, "ref": r}
                                              for i, (s, r) in enumerate(zip(SRC, REF_A))])
    write_jsonl(str(data / "sft_val.jsonl"), [{"id": f"v-{i}", "src": s, "ref": r}
                                              for i, (s, r) in enumerate(zip(SRC[:3], REF_A[:3]))])
    ENV["MTRLHF_SAMPLING_FILE"] = str(tmp / "sampling.json")
    for name, (sl, tl) in {"flores_devtest.en-de": ("en", "de"), "flores_devtest.de-en": ("de", "en")}.items():
        src, ref = (SRC, REF_A) if sl == "en" else (REF_A, SRC)
        write_jsonl(str(data / "testsets" / f"{name}.jsonl"),
                    [{"id": f"{name}-{i}", "src": s, "ref": r, "src_lang": sl, "tgt_lang": tl}
                     for i, (s, r) in enumerate(zip(src[:3], ref[:3]))])
    return {"tmp": tmp, "base": base, "data": data, "ckpt": str(tmp / "ckpt"), "runs": str(tmp / "runs")}


def train(s, config, *sets):
    common = [f"base_model={s['base']}", f"out_root={s['runs']}", "max_steps=2", "batch_size=4",
              "grad_accum=1", "logging_steps=1", "max_length=128"]
    return run("scripts/p2_train.py", "--config", config, "--ckpt", s["ckpt"], "--set", *common, *sets)


def test_phase2_end_to_end(setup):
    s = setup
    d = s["data"]
    train(s, "configs/p2/sft.yaml", f"data={d}/sft_train.jsonl")
    meta = json.load(open(os.path.join(s["ckpt"], "sft", "mtrlhf_meta.json")))
    assert meta["parent"] is None and meta["base_model"] == s["base"]
    # an existing adapter is not retrained
    assert "already exists" in train(s, "configs/p2/sft.yaml", f"data={d}/sft_train.jsonl")

    # sampling pilot: chooses the setting every sampling-based method then uses
    out = run("scripts/p2_sampling_pilot.py", "--system", "sft", "--prompts", f"{d}/sft_val.jsonl", "--limit", "3",
              "--n", "3", "--temperatures", "0.7", "1.0", "--top-ps", "1.0", "--reward", "chrf",
              "--max-new-tokens", "16", "--ckpt", s["ckpt"], "--out", str(s["tmp"] / "pilot"),
              "--write", ENV["MTRLHF_SAMPLING_FILE"])
    chosen = json.load(open(ENV["MTRLHF_SAMPLING_FILE"]))
    assert chosen["temperature"] in (0.7, 1.0) and "Chosen: temperature" in out
    assert os.path.exists(s["tmp"] / "pilot" / "report.md")

    sample = ["scripts/p2_sample.py", "--system", "sft", "--prompts", f"{d}/prompts.jsonl", "--n", "4",
              "--reward", "chrf", "--out", f"{d}/candidates.jsonl", "--ckpt", s["ckpt"], "--max-new-tokens", "24",
              "--chunk", "3"]
    assert "(sampling pilot" in run(*sample)
    meta = json.load(open(f"{d}/candidates.meta.json"))
    assert (meta["temperature"], meta["top_p"]) == (chosen["temperature"], chosen["top_p"])
    # samples drawn with another setting are never mixed into the same file
    assert "holds samples drawn with" in run(*sample, "--temperature", "0.5", "--top-p", "1.0", expect_ok=False)
    cands = read_jsonl(f"{d}/candidates.jsonl")
    assert len(cands) == len(SRC) and all(len(c["samples"]) == 4 for c in cands)
    assert set(cands[0]["samples"][0]["flags"]) == {"wrong_language", "number_mismatch", "length_ratio", "copy"}

    out = run("scripts/p2_build_pairs.py", "--candidates", f"{d}/candidates.jsonl", "--out", str(d),
              "--rsft-min-score", "-1000", "--dpo-margin", "0")
    assert read_jsonl(f"{d}/rsft.jsonl") and read_jsonl(f"{d}/dpo_pairs.jsonl"), out

    assert json.load(open(f"{d}/pairs_stats.json"))["sampling"]["temperature"] == chosen["temperature"]
    train(s, "configs/p2/rsft.yaml", f"data={d}/rsft.jsonl")
    train(s, "configs/p2/dpo.yaml", f"data={d}/dpo_pairs.jsonl")
    assert "already exists" in train(s, "configs/p2/rsft.yaml", f"data={d}/rsft.jsonl")
    # new samples with another setting -> RSFT is retrained on them, not silently kept
    run(*sample, "--temperature", "0.5", "--top-p", "1.0", "--on-mismatch", "restart")
    run("scripts/p2_build_pairs.py", "--candidates", f"{d}/candidates.jsonl", "--out", str(d),
        "--rsft-min-score", "-1000", "--dpo-margin", "0")
    assert "retraining" in train(s, "configs/p2/rsft.yaml", f"data={d}/rsft.jsonl")
    assert json.load(open(os.path.join(s["ckpt"], "rsft", "mtrlhf_meta.json")))["data_sampling"]["temperature"] == 0.5
    train(s, "configs/p2/grpo_guarded.yaml", f"data={d}/prompts.jsonl", "reward=chrf", "num_generations=2",
          "batch_size=2", "save_steps=1", "max_completion_length=16", "sample_log_every=1")
    for name in ("rsft", "dpo", "grpo_guarded", "grpo_guarded-step1", "grpo_guarded-step2"):
        meta = json.load(open(os.path.join(s["ckpt"], name, "mtrlhf_meta.json")))
        assert meta["parent"] == "sft", name
    log = json.load(open(os.path.join(s["runs"], "grpo_guarded", "log_history.json")))
    logged = {k for entry in log for k in entry}
    assert any("guard_number_mismatch" in k for k in logged) and any("chrf" in k for k in logged)
    samples = read_jsonl(os.path.join(s["runs"], "grpo_guarded", "samples.jsonl"))
    assert samples and "flag_wrong_language" in samples[0]

    outs = str(s["tmp"] / "outputs")
    common = ["--data", str(d), "--out", outs, "--ckpt", s["ckpt"], "--base-model", s["base"],
              "--max-new-tokens", "16"]
    run("scripts/p2_translate.py", "--systems", "base", "sft", "dpo", "grpo_guarded-step1",
        "--testsets", "flores_devtest.en-de", "flores_devtest.de-en", *common)
    run("scripts/p2_translate.py", "--systems", "sft", "--best-of", "3", "--rerank-reward", "chrf",
        "--testsets", "flores_devtest.en-de", *common)
    summary = run("scripts/p2_translate.py", "--summary-only", "--score", "chrf", *common)
    for system in ("base", "sft", "dpo", "grpo_guarded-step1", "sft+bo3"):
        assert f"| {system} | flores_devtest.en-de |" in summary
    bo = read_jsonl(os.path.join(outs, "sft+bo3", "flores_devtest.en-de.jsonl"))
    assert len(bo[0]["candidates"]) == 3
    assert bo[0]["hyp"] == max(bo[0]["candidates"], key=lambda c: c["score"])["hyp"]


def test_direction_diagnostic(setup):
    """Runs after the end-to-end test (needs its sft adapter): adapter off, scaled and on."""
    s = setup
    texts = s["tmp"] / "texts.json"
    json.dump({"en": SRC[:3], "de": REF_A[:3], "fr": SRC[:3], "zh": REF_A[:3]}, open(texts, "w"))
    out = s["tmp"] / "diag"
    log = run("scripts/p2_diagnose_direction.py", "--ckpt", s["ckpt"], "--base-model", s["base"], "--texts",
              str(texts), "--n", "3", "--max-new-tokens", "8", "--out", str(out))
    res = json.load(open(out / "direction.json"))["results"]
    assert [r["scale"] for r in res[:5]] == [None, 0.25, 0.5, 0.75, 1.0] and len(res) == 12
    assert all(r["n"] == 3 and 0 <= r["target_lang_pct"] <= 100 for r in res)
    assert "| de→en, adapter off |" in open(out / "direction.md").read() and "chrF" in log


def test_dpo_epoch_sweep_keeps_an_adapter_per_epoch(setup):
    """Runs after the end-to-end test (needs its sft adapter and DPO pairs)."""
    s = setup
    d = s["data"]
    train(s, "configs/p2/dpo_sweep.yaml", f"data={d}/dpo_pairs.jsonl", "max_steps=-1", "epochs=2")
    for k in (1, 2):
        meta = json.load(open(os.path.join(s["ckpt"], f"dpo_sweep-ep{k}", "mtrlhf_meta.json")))
        assert meta["parent"] == "sft" and meta["epoch"] == k and meta["epochs_planned"] == 2
    outs = str(s["tmp"] / "outputs_sweep")
    run("scripts/p2_translate.py", "--systems", "dpo_sweep-ep1", "dpo_sweep-ep2", "--testsets", "flores_devtest.en-de",
        "--data", str(d), "--out", outs, "--ckpt", s["ckpt"], "--base-model", s["base"], "--max-new-tokens", "8")
    assert all(read_jsonl(os.path.join(outs, f"dpo_sweep-ep{k}", "flores_devtest.en-de.jsonl")) for k in (1, 2))
