"""Phase 3 analysis on a synthetic results/p2 tree shaped like the real one (CPU only).

The fixture plants a known exploit: GRPO drops numbers on a third of the segments,
which the training reward (MetricX-QE) rewards and the held-out metric (COMET-22)
punishes. The analysis has to find it in every section.
"""

import csv
import json
import os
import subprocess
import sys

import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))

from mtrlhf import evaluation as ev  # noqa: E402

MODEL = "qwen3-1.7b"
N = 45


def write_jsonl(path, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))


def segment(i, ts):
    src = f"The city council met {i + 3} times in {2000 + i} and approved the new plan for the harbour."
    ref = f"Der Stadtrat traf sich {i + 3} Mal im Jahr {2000 + i} und billigte den neuen Plan für den Hafen."
    return {"id": f"{ts}-{i:04d}", "src": src, "ref": ref, "src_lang": "en", "tgt_lang": "de"}


def system_rows(ts, kind, step=None):
    rows = []
    for i in range(N):
        r = segment(i, ts)
        noise = ((i * 7) % 5 - 2) / 100
        hyp, train, held = r["ref"].replace("billigte", "genehmigte"), -5.0 + noise, 0.80 + noise / 10
        if kind == "rsft":
            train, held = train + 0.3, held + 0.005
        elif kind == "dpo":
            train, held = train + 0.6, held + 0.01
        elif kind == "grpo":
            frac = (step or 625) / 625
            train += 1.5 * frac
            if i % 3 == 0:  # the planted exploit: drop the numbers, MetricX-QE likes it, COMET-22 does not
                hyp = f"Der Stadtrat traf sich mehrmals und billigte den neuen Plan für den Hafen."
                train += 1.5 * frac
                held -= 0.06 * frac
            else:
                held += 0.01 * min(frac, 1) - 0.01 * max(frac - 1, 0)
        elif kind == "grpo_guarded":
            train, held = train + 1.0, held + 0.012
            hyp = hyp.replace("neuen Plan", "neuen Bebauungsplan") if i % 2 else hyp
        rows.append({**r, "hyp": hyp, "scores_metricx24_qe": train, "scores_comet22": held,
                     "scores_chrf": 60.0 + 10 * noise})
    return rows


def other_direction(ts, kind):
    rows = []
    for i in range(N):
        de = f"Der Stadtrat traf sich {i + 3} Mal und billigte den neuen Plan für den Hafen der Stadt."
        en = f"The city council met {i + 3} times and approved the new plan for the harbour of the city."
        if ts.endswith("de-en"):
            hyp = de if kind == "sft" else en  # sft forgot the direction and copies the German source
            rows.append({"id": f"{ts}-{i}", "src": de, "ref": en, "src_lang": "de", "tgt_lang": "en", "hyp": hyp,
                         "scores_comet22": 0.6 if kind == "sft" else 0.85})
        else:
            rows.append({"id": f"{ts}-{i}", "src": en, "ref": "市议会开会批准了港口的新计划。", "src_lang": "en",
                         "tgt_lang": "zh", "hyp": "市议会开会批准了新计划。", "scores_comet22": 0.8})
    return rows


@pytest.fixture
def tree(tmp_path):
    res = tmp_path / "results"
    out = res / "p2" / MODEL / "outputs"
    for system, kind in (("sft", "sft"), ("rsft", "rsft"), ("dpo", "dpo"), ("grpo-step625", "grpo"),
                         ("grpo_guarded-step625", "grpo_guarded")):
        for ts in ("flores_devtest.en-de", "wmt24.en-de", "flores_dev.en-de"):
            write_jsonl(out / system / f"{ts}.jsonl", system_rows(ts, kind))
        if system in ("sft", "grpo-step625"):
            for ts in ("flores_devtest.de-en", "flores_devtest.en-zh"):
                write_jsonl(out / system / f"{ts}.jsonl", other_direction(ts, kind))
    for step in (125, 250, 375, 500, 750, 1000, 1250):
        write_jsonl(out / f"grpo-step{step}" / "flores_dev.en-de.jsonl", system_rows("flores_dev.en-de", "grpo", step))
    bo = system_rows("flores_dev.en-de", "sft")
    for r in bo:
        r["candidates"] = [{"hyp": r["hyp"], "score": r["scores_metricx24_qe"] + k / 4, "comet22": 0.80 - k / 200}
                           for k in range(8)]
    write_jsonl(out / "sft+bo8" / "flores_dev.en-de.jsonl", bo)
    for run, guard in (("grpo", -0.05), ("grpo_guarded", -0.01)):
        log = [{"step": s, "loss": 0.1, "kl": s / 1e4, "completions/mean_length": 40.0,
                "rewards/metricx24_qe/mean": -5 + s / 400, "rewards/guard_number_mismatch/mean": guard * s / 625,
                "rewards/guard_wrong_language/mean": 0.0} for s in range(10, 1251, 10)]
        os.makedirs(res / "p2" / MODEL / "training" / run)
        json.dump(log, open(res / "p2" / MODEL / "training" / run / "log_history.json", "w"))
    os.makedirs(res / "p1" / "report")
    with open(res / "p1" / "report" / "blind_spots.csv", "w") as fh:
        fh.write("reward,category,n,detection,lo,hi,mean_drop_sd\n")
        for cat, det in (("number", 0.84), ("omission", 0.95), ("wrong_language", 0.9), ("addition", 0.97),
                         ("untranslated", 0.99), ("repetition", 0.98), ("negation", 0.9)):
            fh.write(f"metricx24_qe,{cat},100,{det},0,1,0.5\n")
            fh.write(f"cometkiwi22,{cat},100,0.5,0,1,0.5\n")
    return res


def analyze(res):
    proc = subprocess.run([sys.executable, f"{ROOT}/scripts/p3_analyze.py", "--models", MODEL, "gemma3-1b",
                           "--results", str(res / "p2"), "--blind-spots", str(res / "p1/report/blind_spots.csv"),
                           "--out", str(res / "p3"), "--bootstrap", "200"], capture_output=True, text=True)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    return open(res / "p3" / "report.md").read(), json.load(open(res / "p3" / "results.json"))[MODEL]


def test_report_finds_the_planted_exploit(tree):
    report, data = analyze(tree)
    for section in ("### A ·", "### B ·", "### C ·", "### D ·", "### E ·", "### F ·"):
        assert section in report
    prim = {r["system"]: r for r in data["methods_primary"]}
    grpo = prim["grpo-step625"]
    assert grpo["delta_metricx24_qe"]["lo"] > 0 and grpo["delta_comet22"]["hi"] < 0  # reward up, held-out down
    assert grpo["gap_sd"] > 0 and prim["dpo"]["delta_comet22"]["lo"] > 0
    assert "### A2 ·" in report and grpo["shift"]["worse_pct"] == pytest.approx(100 * 15 / N)
    assert prim["dpo"]["shift"]["worse_pct"] == 0 and prim["sft"]["shift"] is None
    assert abs(sum(grpo["tails"]["bands"].values()) - 100) < 1e-9
    worst = [json.loads(x) for x in open(tree / "p3" / f"{MODEL}_worst_regressions_grpo.jsonl")]
    assert len(worst) == 20 and worst[0]["held_delta"] <= worst[-1]["held_delta"]
    assert "number_mismatch" in worst[0]["introduced"]
    assert grpo["rates"]["number_mismatch"] == pytest.approx(100 * 15 / N)
    assert data["suspicious_wins"]["grpo-step625"] == 15 and data["suspicious_wins"]["dpo"] == 0
    wins = [json.loads(x) for x in open(tree / "p3" / f"{MODEL}_suspicious_wins_grpo.jsonl")]
    assert all("number_mismatch" in w["introduced"] for w in wins)
    pvo = {p["probe"]: p for p in data["predicted_vs_observed"]}
    assert pvo["number"]["delta_pp"] > 30 and pvo["number"]["share_of_suspicious_wins"] == 100
    assert pvo["number"]["missed_pct"] == pytest.approx(16)
    curve = data["checkpoint_curves"]["grpo"]
    assert [p["step"] for p in curve] == [0, 125, 250, 375, 500, 625, 750, 1000, 1250]
    assert curve[-1]["train"] > curve[5]["train"] and curve[-1]["held_out"] < curve[5]["held_out"]
    assert [p["k"] for p in data["best_of_k"]] == [1, 2, 4, 8]
    assert data["best_of_k"][-1]["train"] > data["best_of_k"][0]["train"]
    log = data["training_logs"]["grpo"]
    assert log[-1]["number_mismatch"] == pytest.approx(10.0) and log[-1]["reward"] > log[0]["reward"]
    fg = {(r["testset"], r["system"]): r for r in data["forgetting"]}
    assert fg[("flores_devtest.de-en", "sft")]["wrong_language"] == 100
    assert fg[("flores_devtest.de-en", "grpo-step625")]["wrong_language"] == 0
    assert data["guards_effect"]["delta_held_out"]["lo"] > 0
    for fig in ("predicted_vs_observed.png", f"{MODEL}_overoptimization.png", f"{MODEL}_training.png",
                f"{MODEL}_tails.png"):
        assert os.path.getsize(tree / "p3" / fig) > 10_000
    assert "gemma3-1b" not in report  # a model without outputs is skipped


def test_report_runs_on_partial_results(tree):
    import shutil

    for d in os.listdir(tree / "p2" / MODEL / "outputs"):
        if d not in ("sft", "rsft"):
            shutil.rmtree(tree / "p2" / MODEL / "outputs" / d)
    shutil.rmtree(tree / "p2" / MODEL / "training")
    report, data = analyze(tree)
    assert "### A ·" in report and "### D ·" not in report and "### B ·" not in report
    assert data["suspicious_wins"] == {"rsft": 0}
    for f in os.listdir(tree / "p2" / MODEL / "outputs" / "rsft"):  # unscored: no claim of "0 suspicious wins"
        p = tree / "p2" / MODEL / "outputs" / "rsft" / f
        rows = [{k: v for k, v in json.loads(x).items() if k != "scores_comet22"} for x in open(p)]
        write_jsonl(p, rows)
    report, data = analyze(tree)
    assert "Suspicious wins" not in report and "suspicious_wins" not in data and "wmt24.en-de (paragraph" in report


def sheet(path):
    with open(path, newline="", encoding="utf-8-sig") as fh:
        return list(csv.DictReader(fh))


def fill(path, rows):
    with open(path, "w", newline="", encoding="utf-8-sig") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def human(res, *args):
    proc = subprocess.run([sys.executable, f"{ROOT}/scripts/p3_human_eval.py", *args, "--model", MODEL,
                           "--results", str(res / "p2"), "--out", str(res / "p3" / "human_eval")],
                          capture_output=True, text=True)
    return proc


def test_human_check_is_blinded_and_analysed(tree):
    proc = human(tree, "prepare", "--n", "20")
    assert proc.returncode == 0, proc.stdout + proc.stderr
    d = tree / "p3" / "human_eval" / MODEL
    key = json.load(open(d / "ab_key.json"))
    assert key["system"] == "grpo_guarded-step625"  # best on dev by COMET-22, not the exploiting grpo
    ab = sheet(d / "ab_sheet.csv")
    assert len(ab) == 20 and set(ab[0]) == {"item", "source", "reference", "A", "B", "better", "comment"}
    sides = [key["items"][r["item"]]["rl_side"] for r in ab]
    assert 0 < sides.count("A") < 20  # randomised
    audit = sheet(d / "audit_sheet.csv")
    akey = json.load(open(d / "audit_key.json"))
    assert akey["system"] == "grpo-step625" and len(audit) == 15
    assert all(akey["items"][r["item"]]["signals"] == ["number_mismatch"] for r in audit)
    assert "error_type" in audit[0] and "number_mismatch" not in open(d / "audit_sheet.csv").read()

    # annotate: the RL system wins 15 of 20 A/B items (5 ties); every suspicious win is a real exploit
    for i, r in enumerate(ab):
        rl = key["items"][r["item"]]["rl_side"]
        r["better"] = "tie" if i % 4 == 0 else rl
    fill(d / "ab_sheet.csv", ab)
    for r in audit:
        rl = akey["items"][r["item"]]["rl_side"]
        r["better"], r["error_type"] = ("B" if rl == "A" else "a"), "Number"
    fill(d / "audit_sheet.csv", audit)
    proc = human(tree, "analyze")
    assert proc.returncode == 0, proc.stdout + proc.stderr
    s = json.load(open(d / "summary.json"))
    assert (s["ab"]["wins"], s["ab"]["ties"], s["ab"]["losses"]) == (15, 5, 0)
    assert s["ab"]["win_rate"] == pytest.approx(17.5 / 20) and s["ab"]["p"] < 0.001
    assert s["audit"] == {"system": "grpo-step625", "n": 15, "held_out": 15, "train": 0, "tie": 0,
                          "error_types": {"number": 15}}
    report, data = analyze(tree)
    assert "### G · Human check" in report and data["human"]["audit"]["held_out"] == 15

    # a second `prepare` must not wipe the annotations
    proc = human(tree, "prepare")
    assert proc.returncode != 0 and "already has annotations" in proc.stdout + proc.stderr


def test_p3_score_fills_missing_scores(tree):
    out = tree / "p2" / MODEL / "outputs"
    rows = [json.loads(x) for x in open(out / "sft" / "flores_dev.en-de.jsonl")]
    for r in rows:
        del r["scores_chrf"]
    write_jsonl(out / "sft" / "flores_dev.en-de.jsonl", rows)
    proc = subprocess.run([sys.executable, f"{ROOT}/scripts/p3_score.py", "--model", MODEL, "--results",
                           str(tree / "p2"), "--metrics", "chrf", "--held-out", "chrf"], capture_output=True, text=True)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "sft/flores_dev.en-de.jsonl" in proc.stdout and "sft/flores_devtest.de-en.jsonl" in proc.stdout
    assert "rsft/" not in proc.stdout and "sft/flores_devtest.en-de.jsonl" not in proc.stdout  # already scored
    rows = [json.loads(x) for x in open(out / "sft" / "flores_dev.en-de.jsonl")]
    assert all(r["scores_chrf"] > 50 for r in rows)
    bo = [json.loads(x) for x in open(out / "sft+bo8" / "flores_dev.en-de.jsonl")]
    assert all("chrf" in c for r in bo for c in r["candidates"])
    proc = subprocess.run([sys.executable, f"{ROOT}/scripts/p3_score.py", "--model", MODEL, "--results",
                           str(tree / "p2"), "--metrics", "chrf", "--held-out", "chrf"], capture_output=True, text=True)
    assert "scored" not in proc.stdout  # nothing left to do


def test_statistics():
    d = ev.paired_delta([1.0] * 50 + [0.0] * 50, [0.0] * 100, n_boot=500)
    assert d["diff"] == pytest.approx(0.5) and d["lo"] < 0.5 < d["hi"] and d["p"] < 0.01
    assert ev.paired_delta([], [])["diff"] != ev.paired_delta([], [])["diff"]  # nan
    assert ev.sign_test(9, 1) == pytest.approx(2 * 11 / 1024)
    assert ev.sign_test(5, 5) == 1.0
    w = ev.win_rate([1, 1, 0.5, 0])
    assert (w["wins"], w["ties"], w["losses"], w["win_rate"]) == (2, 1, 1, 0.625)


def test_exploit_flags():
    row = {"src": "In 2019, 35 people came.", "ref": "Im Jahr 2019 kamen 35 Menschen.", "tgt_lang": "de"}
    assert ev.exploit_flags({**row, "hyp": "Im Jahr 2019 kamen 35 Menschen."}) == {k: False for k in ev.SIGNALS}
    assert ev.exploit_flags({**row, "hyp": "Im Jahr 2019 kamen viele Menschen."})["number_mismatch"]
    assert ev.exploit_flags({**row, "hyp": "Es kamen."})["short"]
    assert ev.repetition("das ist gut das ist gut das ist gut") and not ev.repetition("das ist gut")
    assert ev.training_reward_key([{"rewards/guard_copy/mean": 0, "rewards/cometkiwi22/mean": 0.7}]) == \
        "rewards/cometkiwi22/mean"
