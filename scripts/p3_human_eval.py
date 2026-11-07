#!/usr/bin/env python
"""Phase 3: a small blinded human check of the automatic results (CPU only).

Two annotation sheets per model, both blinded (the two translations are shown as A
and B in random order; which one is which is kept in a separate key file):

  ab_sheet.csv     60 random FLORES devtest segments, SFT vs. the best preference-tuned
                   system (chosen on FLORES dev by COMET-22, so the test set is not used
                   to pick it). Question: which translation is better? -> a win rate
                   with a 95% CI, and whether humans side with the training reward or
                   with the held-out metric where the two disagree.
  audit_sheet.csv  up to 50 of GRPO's "suspicious wins" (the training reward says GRPO
                   is better, COMET-22 says worse). Question: who is right, and what is
                   wrong with the worse translation? -> how many are real reward
                   exploits, and of which kind.

Usage:
  python scripts/p3_human_eval.py prepare --model qwen3-1.7b
  # fill in the `better` (and `error_type`) columns, save as CSV, then:
  python scripts/p3_human_eval.py analyze --model qwen3-1.7b
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from mtrlhf import evaluation as ev  # noqa: E402

TESTSET, DEV = "flores_devtest.en-de", "flores_dev.en-de"
CANDIDATES = ["rsft", "dpo", "grpo-step625", "grpo_guarded-step625", "grpo_cometkiwi-step625"]
AUDIT_SYSTEM = "grpo-step625"
ERROR_TYPES = ["omission", "addition", "mistranslation", "number", "named_entity", "wrong_language",
               "untranslated", "grammar", "repetition", "other"]
CHOICES = {"a": "A", "b": "B", "tie": "tie", "=": "tie", "same": "tie"}

INSTRUCTIONS = """# Human check · {model}

Two short annotation jobs. Open each CSV in a spreadsheet (Google Sheets, Excel,
LibreOffice), fill in the empty columns, and save it again as CSV (UTF-8) under the
same name. Then run `python scripts/p3_human_eval.py analyze --model {model}`.

The two translations are shown as **A** and **B** in random order. You do not know
which system wrote which; please do not look at `*_key.json` before you finish.

## ab_sheet.csv ({n_ab} segments)

`better`: `A`, `B` or `tie` — which German translation is better overall
(meaning first, then fluency). The reference is there to help; it is not the answer.
`comment`: optional.

## audit_sheet.csv ({n_audit} segments)

Segments where the two automatic metrics disagree strongly.
`better`: `A`, `B` or `tie`.
`error_type`: the main problem of the **worse** translation, one of
{types}
(leave empty for a tie). `comment`: optional.

Tip: about 1 minute per segment; take a break in the middle.
"""


def pick_system(outputs: dict) -> str:
    """The candidate with the best mean COMET-22 on FLORES dev."""
    best, best_score = None, float("-inf")
    for s in CANDIDATES:
        h = ev.scores(outputs.get(s, {}).get(DEV, []), ev.HELD_OUT)
        if h and ev.mean(h) > best_score:
            best, best_score = s, ev.mean(h)
    if best is None:
        raise SystemExit(f"No candidate system has {ev.HELD_OUT} scores on {DEV} yet; run Phase 2 first.")
    return best


def write_csv(path: str, header: list[str], rows: list[list]) -> None:
    with open(path, "w", newline="", encoding="utf-8-sig") as fh:  # BOM: Excel shows umlauts correctly
        w = csv.writer(fh)
        w.writerow(header)
        w.writerows(rows)


def read_csv(path: str) -> list[dict]:
    with open(path, newline="", encoding="utf-8-sig") as fh:
        return [{(k or "").strip().lower(): (v or "").strip() for k, v in r.items()} for r in csv.DictReader(fh)]


def blinded(items: list[tuple[dict, dict]], rng: random.Random) -> tuple[list[list], dict]:
    """items: (rl_row, sft_row) pairs -> sheet rows and the key {item: {id, rl_side, ...}}."""
    sheet, key = [], {}
    for i, (r, s) in enumerate(items, 1):
        rl_first = rng.random() < 0.5
        a, b = (r, s) if rl_first else (s, r)
        sheet.append([i, r["src"], r.get("ref", ""), a["hyp"], b["hyp"]])
        key[str(i)] = {"id": r["id"], "rl_side": "A" if rl_first else "B"}
        for m in (ev.TRAIN, ev.HELD_OUT):
            if f"scores_{m}" in r and f"scores_{m}" in s:
                key[str(i)][f"delta_{m}"] = r[f"scores_{m}"] - s[f"scores_{m}"]
    return sheet, key


def prepare(args) -> None:
    outputs = ev.load_outputs(args.model, args.results)
    if TESTSET not in outputs.get("sft", {}):
        raise SystemExit(f"No sft outputs on {TESTSET} for {args.model}.")
    system = args.system or pick_system(outputs)
    out = os.path.join(args.out, args.model)
    started = [f for f in ("ab_sheet.csv", "audit_sheet.csv") if os.path.exists(os.path.join(out, f))
               and any(r.get("better") for r in read_csv(os.path.join(out, f)))]
    if started and not args.force:
        raise SystemExit(f"{', '.join(started)} in {out}/ already has annotations; use --force to start over.")
    os.makedirs(out, exist_ok=True)
    rng = random.Random(args.seed)
    sft = outputs["sft"][TESTSET]

    rl, base = ev.aligned(outputs[system][TESTSET], sft)
    differ = [(r, s) for r, s in zip(rl, base) if r["hyp"].strip() != s["hyp"].strip()]
    ab = rng.sample(differ, min(args.n, len(differ)))
    sheet, key = blinded(ab, rng)
    write_csv(os.path.join(out, "ab_sheet.csv"), ["item", "source", "reference", "A", "B", "better", "comment"],
              [row + ["", ""] for row in sheet])
    ev.dump(os.path.join(out, "ab_key.json"), {"system": system, "testset": TESTSET, "seed": args.seed,
                                               "identical_share": 1 - len(differ) / max(len(rl), 1),
                                               "items": key})

    audit_system = args.audit_system or (AUDIT_SYSTEM if TESTSET in outputs.get(AUDIT_SYSTEM, {}) else system)
    wins = ev.suspicious_wins(outputs[audit_system][TESTSET], sft) or []
    wins = rng.sample(wins, min(args.audit, len(wins)))
    by_id = {r["id"]: r for r in outputs[audit_system][TESTSET]}
    sft_by_id = {r["id"]: r for r in sft}
    sheet, key = blinded([(by_id[w["id"]], sft_by_id[w["id"]]) for w in wins], rng)
    for k, w in zip(sorted(key, key=int), wins):
        key[k]["signals"] = w["introduced"]
    write_csv(os.path.join(out, "audit_sheet.csv"),
              ["item", "source", "reference", "A", "B", "better", "error_type", "comment"],
              [row + ["", "", ""] for row in sheet])
    ev.dump(os.path.join(out, "audit_key.json"), {"system": audit_system, "testset": TESTSET, "seed": args.seed,
                                                  "items": key})
    with open(os.path.join(out, "README.md"), "w", encoding="utf-8") as fh:
        fh.write(INSTRUCTIONS.format(model=args.model, n_ab=len(ab), n_audit=len(wins),
                                     types=", ".join(f"`{t}`" for t in ERROR_TYPES)))
    print(f"{args.model}: {len(ab)} A/B items (sft vs {ev.name(system)}, {len(differ)} of {len(rl)} segments differ) "
          f"and {len(wins)} audit items ({ev.name(audit_system)}) -> {out}/")


def choice(value: str) -> str | None:
    return CHOICES.get(value.strip().lower()) if value.strip() else None


def analyze(args) -> None:
    out = os.path.join(args.out, args.model)
    summary: dict = {"model": args.model}
    md = [f"### G · Human check ({args.model})", ""]

    ab_path, ab_key_path = os.path.join(out, "ab_sheet.csv"), os.path.join(out, "ab_key.json")
    if os.path.exists(ab_path) and os.path.exists(ab_key_path):
        key = json.load(open(ab_key_path))
        outcomes, agree = [], {ev.TRAIN: [], ev.HELD_OUT: []}
        for row in read_csv(ab_path):
            c, k = choice(row.get("better", "")), key["items"].get(row.get("item", ""))
            if c is None or k is None:
                continue
            o = 0.5 if c == "tie" else float(c == k["rl_side"])
            outcomes.append(o)
            for m in agree:
                d = k.get(f"delta_{m}")
                if o != 0.5 and d is not None and d != 0:
                    agree[m].append(float((d > 0) == (o == 1)))
        wr = ev.win_rate(outcomes)
        summary["ab"] = {"system": key["system"], **wr,
                         "agreement": {m: {"n": len(v), "pct": 100 * ev.mean(v)} for m, v in agree.items()}}
        if wr["n"]:
            md += [f"Blinded A/B on {wr['n']} random {key['testset']} segments where the outputs differ: "
                   f"**{ev.name(key['system'])}** vs. sft — {wr['wins']} wins, {wr['losses']} losses, {wr['ties']} ties; "
                   f"win rate {100 * wr['win_rate']:.0f}% [{100 * wr['lo']:.0f}, {100 * wr['hi']:.0f}] "
                   f"(sign test p = {wr['p']:.3f}). Where the annotator had a preference, it matched the sign of the "
                   f"MetricX-QE difference in {summary['ab']['agreement'][ev.TRAIN]['pct']:.0f}% of segments and "
                   f"COMET-22 in {summary['ab']['agreement'][ev.HELD_OUT]['pct']:.0f}%.", ""]

    au_path, au_key_path = os.path.join(out, "audit_sheet.csv"), os.path.join(out, "audit_key.json")
    if os.path.exists(au_path) and os.path.exists(au_key_path):
        key = json.load(open(au_key_path))
        sides = {"held_out": 0, "train": 0, "tie": 0}
        errors: dict[str, int] = {}
        n = 0
        for row in read_csv(au_path):
            c, k = choice(row.get("better", "")), key["items"].get(row.get("item", ""))
            if c is None or k is None:
                continue
            n += 1
            if c == "tie":
                sides["tie"] += 1
            elif c == k["rl_side"]:
                sides["train"] += 1  # RL really is better: the held-out metric was wrong here
            else:
                sides["held_out"] += 1  # RL is worse: the training reward was exploited
                t = row.get("error_type", "").strip().lower().replace(" ", "_") or "unspecified"
                errors[t] = errors.get(t, 0) + 1
        summary["audit"] = {"system": key["system"], "n": n, **sides, "error_types": errors}
        if n:
            md += [f"Audit of {n} suspicious wins of {ev.name(key['system'])}: the RL output was really worse in "
                   f"{sides['held_out']} ({100 * sides['held_out'] / n:.0f}%, a reward exploit), really better in "
                   f"{sides['train']} (COMET-22 was wrong), tie in {sides['tie']}. Errors in the exploited cases: "
                   + (", ".join(f"{k} {v}" for k, v in sorted(errors.items(), key=lambda kv: -kv[1])) or "–") + ".", ""]

    if len(md) == 2:
        raise SystemExit(f"No filled-in sheets in {out}/ (run `prepare` first, then annotate).")
    ev.dump(os.path.join(out, "summary.json"), summary)
    with open(os.path.join(out, "summary.md"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(md))
    print("\n".join(md))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("action", choices=["prepare", "analyze"])
    ap.add_argument("--model", required=True)
    ap.add_argument("--results", default="results/p2")
    ap.add_argument("--out", default="results/p3/human_eval")
    ap.add_argument("--system", help="system to compare with sft (default: best on FLORES dev by COMET-22)")
    ap.add_argument("--audit-system", help=f"system whose suspicious wins are audited (default: {AUDIT_SYSTEM})")
    ap.add_argument("--n", type=int, default=60)
    ap.add_argument("--audit", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--force", action="store_true", help="overwrite sheets that already have annotations")
    args = ap.parse_args()
    prepare(args) if args.action == "prepare" else analyze(args)


if __name__ == "__main__":
    main()
