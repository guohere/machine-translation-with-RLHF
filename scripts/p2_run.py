#!/usr/bin/env python
"""Phase 2 driver: do whatever is left, following configs/p2/run.yaml.

The Kaggle notebook only pulls the repository and runs this script, so changes to
the pipeline and to the plan never require editing or re-importing the notebook.
Each step checks what is already saved (adapters on the Hub, results on GitHub) and
skips it; a failed step pushes its log to GitHub and stops the run.

Order for a model: data -> (smoke test) -> Part A: sft, sampling pilot, samples,
pairs, rsft + dpo, translations -> Part B: grpo + grpo_guarded, translations ->
quick evaluation -> Phase 3 scores and report (results/p3/). With part=auto, Part B starts only in a run where Part A had
nothing left to do, so each part gets a full 12-hour Kaggle session.

Examples:
  python scripts/p2_run.py                       # the planned run
  python scripts/p2_run.py --set part=B          # override the plan
  python scripts/p2_run.py --dry-run             # print what would run
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

DEFAULT_BRANCH = "claude/compassionate-ritchie-2nga86"


def load_plan(path: str, sets: list[str]) -> dict:
    import yaml

    with open(path) as fh:
        plan = yaml.safe_load(fh)
    for kv in sets:
        k, v = kv.split("=", 1)
        plan[k] = yaml.safe_load(v)
    return plan


def grpo_runs(plan: dict) -> list[str]:
    return ["grpo", "grpo_guarded"] + (["grpo_cometkiwi"] if plan.get("grpo_cometkiwi") else [])


def model_finished(r, plan: dict) -> bool:
    budget = plan["budget_step"]
    return all(r.have(f"{run}-step{budget}") and not r.missing_outputs(f"{run}-step{budget}", plan["test_sets"])
               for run in grpo_runs(plan))


def smoke_test(r, plan: dict) -> None:
    from mtrlhf.data.prompts import build_messages
    from mtrlhf.models import load_tokenizer

    m, dbg, druns = r.model, f"{r.P['ckpt']}_debug", f"{r.P['runs']}_debug"
    if not r.dry_run:
        tok = load_tokenizer(r.P["model_id"])
        print("prompt:", repr(tok.apply_chat_template(build_messages("The weather is nice today."), tokenize=False,
                                                       add_generation_prompt=True)))
    r.sh(f"python scripts/p2_train.py --model {m} --config configs/p2/sft.yaml --debug --force", "smoke_sft")
    r.sh(f"python scripts/p2_sample.py --model {m} --ckpt {dbg} --system sft --limit 8 --n 4 "
         f"--out {druns}/candidates.jsonl --on-mismatch restart", "smoke_sample")
    if not r.dry_run:
        with open(f"{druns}/candidates.jsonl") as fh:
            print("sample translations:", *[s["hyp"] for s in json.loads(fh.readline())["samples"]], sep="\n  ")
    r.sh(f"python scripts/p2_train.py --model {m} --config configs/p2/grpo_guarded.yaml --debug --force", "smoke_grpo")
    r.sh(f"python scripts/p2_translate.py --model {m} --ckpt {dbg} --systems sft --testsets flores_devtest.en-de "
         f"--limit 8 --out {druns}/outputs --score chrf", "smoke_translate")
    print("smoke test passed", flush=True)
    r.save(f"Phase 2 ({m}): smoke test passed")


def part_a(r, plan: dict) -> None:
    from mtrlhf.io import read_jsonl
    from mtrlhf.sampling import load_sampling

    m, systems = r.model, r.P["systems"]
    sets = plan["test_sets"] + [plan["dev_set"]]
    if not r.have("sft"):
        r.sh(f"python scripts/p2_train.py --model {m} --config configs/p2/sft.yaml", "sft")
        r.save(f"Phase 2 ({m}): sft")
    if load_sampling() is None:
        r.sh(f"python scripts/p2_sampling_pilot.py --model {m} --write results/p2/sampling.json", "sampling_pilot")
        r.save(f"Phase 2 ({m}): sampling pilot")
    chosen = load_sampling()
    if chosen:
        print(f"sampling for every method and model: temperature {chosen['temperature']}, top-p {chosen['top_p']}",
              flush=True)
    else:  # only in a dry run: a real run has just written it
        chosen = {"temperature": "<pilot's choice>", "top_p": "<pilot's choice>"}
        print("sampling: chosen by the pilot in this run", flush=True)
    # what p2_sample.py records for samples drawn now (RSFT/DPO adapters must match it)
    expected = {"system": "sft", "n": plan["n_samples"], "temperature": chosen["temperature"],
                "top_p": chosen["top_p"], "reward": plan["train_reward"], "seed": 0}

    r.restore_candidates()
    n_prompts = len(read_jsonl("data/p2/prompts.jsonl"))
    if not r.candidates_done(chosen, plan["n_samples"], plan["train_reward"], n_prompts):
        r.sh(f"python scripts/p2_sample.py --model {m} --system sft --n {plan['n_samples']} "
             f"--reward {plan['train_reward']} --on-mismatch restart", "sample")
        r.save(f"Phase 2 ({m}): sampled candidates")
    cand_meta = os.path.join(r.P["model_data"], "candidates.meta.json")
    current = json.load(open(cand_meta)) if os.path.exists(cand_meta) else None
    data_sampling = current or expected
    retrain = [s for s in ("rsft", "dpo") if s in systems and r.needs_training(s, data_sampling)]
    # Pairs are only needed to (re)train RSFT/DPO. data/ does not survive a Kaggle session,
    # so a fresh session has no pairs on disk; building them there anyway would count as
    # Part A work and postpone Part B forever.
    if retrain and (r.pairs_sampling() != current or current is None):
        r.sh(f"python scripts/p2_build_pairs.py --model {m}", "pairs")
        r.save(f"Phase 2 ({m}): preference pairs")
    for s in retrain:
        if r.have(s):
            print(f"{s}: trained on samples from another sampling setting -> retraining and translating again")
        r.drop_outputs(s)
    r.run_jobs([(s, f"python scripts/p2_train.py --model {m} --config configs/p2/{s}.yaml") for s in retrain])
    if retrain:
        r.save(f"Phase 2 ({m}): {' and '.join(retrain)}")

    todo = [s for s in ("base", "opus-mt", "sft", "rsft", "dpo") if s in systems and r.missing_outputs(s, sets)]
    if todo:
        r.sh(f"python scripts/p2_translate.py --model {m} --systems {' '.join(todo)} --testsets {' '.join(sets)}",
             "translate_A")
        r.save(f"Phase 2 ({m}): translations ({', '.join(todo)})")
    if "sft+bo8" in systems:
        bo_sets = ["flores_devtest.en-de", "wmt24.en-de", plan["dev_set"]]
        meta = os.path.join(r.P["outputs"], "sft+bo8", "_sampling.json")
        stale = os.path.exists(meta) and {k: json.load(open(meta)).get(k) for k in ("temperature", "top_p")} != \
            {"temperature": chosen["temperature"], "top_p": chosen["top_p"]}
        if stale or r.missing_outputs("sft+bo8", bo_sets):
            r.sh(f"python scripts/p2_translate.py --model {m} --systems sft --best-of 8 --testsets {' '.join(bo_sets)}",
                 "translate_bo8")
            r.save(f"Phase 2 ({m}): translations (sft + best-of-8)")
    # Why does SFT answer de->en in German? Bug or learned behaviour (a few minutes, once per model).
    if "sft" in systems and plan.get("diagnose_direction", True) and \
            not os.path.exists(os.path.join(r.res, "diagnostics", "direction.json")):
        r.sh(f"python scripts/p2_diagnose_direction.py --model {m}", "diagnose_direction")
        r.save(f"Phase 2 ({m}): direction diagnostic")


def part_b(r, plan: dict) -> None:
    m, budget = r.model, plan["budget_step"]
    jobs = []
    for run in grpo_runs(plan):
        if r.have(run) or r.have(f"{run}-step{budget}"):
            print(f"{run}: matched-budget checkpoint already saved, skipping training")
        else:
            jobs.append((run, f"python scripts/p2_train.py --model {m} --config configs/p2/{run}.yaml --force"))
    r.run_jobs(jobs)
    if jobs:
        r.save(f"Phase 2 ({m}): GRPO runs")
    for run in grpo_runs(plan):
        main = f"{run}-step{budget}"
        if r.have(main) and r.missing_outputs(main, plan["test_sets"] + [plan["dev_set"]]):
            r.sh(f"python scripts/p2_translate.py --model {m} --systems {main} "
                 f"--testsets {' '.join(plan['test_sets'] + [plan['dev_set']])}", f"translate_{run}")
        curve = [f"{run}-step{k}" for k in range(125, 2 * budget + 1, 125)
                 if k != budget and r.have(f"{run}-step{k}") and r.missing_outputs(f"{run}-step{k}", [plan["dev_set"]])]
        if curve:
            r.sh(f"python scripts/p2_translate.py --model {m} --systems {' '.join(curve)} --testsets {plan['dev_set']}",
                 f"translate_{run}_curve")
        r.save(f"Phase 2 ({m}): translations ({run})")


def phase3(runners: list, plan: dict) -> None:
    """Fill in the scores Phase 3 needs and refresh results/p3/report.md.

    Runs at the end of every run, so the analysis grows with the results. It never
    stops Phase 2: a failure here is reported and the run carries on.
    """
    from mtrlhf.pipeline import StepFailed

    try:
        for r in runners:
            r.sh(f"python scripts/p3_score.py --model {r.model} --metrics {' '.join(plan['quick_scores'])} "
                 f"--bon-testsets {plan['dev_set']}", "p3_score")
        runners[0].sh(f"python scripts/p3_analyze.py --models {' '.join(plan['models'])}", "p3_analyze")
        runners[0].save("Phase 3: analysis updated")
    except StepFailed as e:
        print(f"Phase 3 analysis skipped ({e}). Phase 2 results are saved; this does not affect them.", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--plan", default="configs/p2/run.yaml")
    ap.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE", help="override plan values")
    ap.add_argument("--branch", default=os.environ.get("MTRLHF_BRANCH", DEFAULT_BRANCH))
    ap.add_argument("--dry-run", action="store_true", help="print the steps instead of running them")
    args = ap.parse_args()

    from mtrlhf.pipeline import Runner, StepFailed

    plan = load_plan(args.plan, args.set)
    hub = plan.get("hub_repo") or "none"
    os.environ.setdefault("MTRLHF_HUB_REPO", hub)
    gh = os.environ.get("GH_TOKEN")

    def runner(model: str) -> Runner:
        return Runner(model, args.branch, gh, hub, dry_run=args.dry_run)

    model = plan["model"]
    if model == "auto":
        model = next((m for m in plan["models"] if not model_finished(runner(m), plan)), None)
        if model is None:
            print("All planned models are finished. Phase 3: scoring and analysis.", flush=True)
            phase3([runner(m) for m in plan["models"]], plan)
            print(f"\nPhase 2 is complete; the analysis is in results/p3/report.md. Next: the human check "
                  f"(python scripts/p3_human_eval.py prepare --model {plan['models'][0]}).", flush=True)
            return
    r = runner(model)
    print(f"=== Phase 2 run: model {model} ({r.P['model_id']}), plan part={plan['part']}, "
          f"smoke_test={plan['smoke_test']} ===", flush=True)
    try:
        if not os.path.exists("data/p2/prompts.jsonl"):
            r.sh("python scripts/p2_prepare_data.py --out data/p2", "prepare_data")
            r.save("Phase 2: data statistics")
        smoke = plan["smoke_test"]
        if smoke is True or (smoke == "auto" and not r.have("sft")):
            smoke_test(r, plan)
        part = str(plan["part"]).upper()
        did_a = False
        if part in ("A", "ALL", "AUTO"):
            before = len(r.ran)
            part_a(r, plan)
            did_a = len(r.ran) > before
            print("Part A complete." if not did_a else "Part A done in this run.", flush=True)
        if part in ("B", "ALL") or (part == "AUTO" and not did_a):
            part_b(r, plan)
            print("Part B complete.", flush=True)
        elif part == "AUTO":
            print("Part B (GRPO) starts in the next run, in a fresh 12-hour session.", flush=True)
        r.sh(f"python scripts/p2_translate.py --model {model} --summary-only --score {' '.join(plan['quick_scores'])}",
             "quick_eval")
        r.save(f"Phase 2 ({model}): quick evaluation")
        phase3([r], plan)
    except StepFailed as e:
        print(f"\nSTOPPED: {e}", flush=True)
        sys.exit(1)
    nxt = model if not model_finished(r, plan) else next(
        (m for m in plan["models"] if m != model and not model_finished(runner(m), plan)), None)
    print(f"\nNext run: {'model ' + nxt if nxt else 'nothing left in Phase 2'}. "
          f"Just 'Save & Run All' again.", flush=True)


if __name__ == "__main__":
    main()
