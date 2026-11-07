#!/usr/bin/env python
"""Diagnose why the SFT model answers German->English requests in German (GPU, ~15 min).

After SFT on English->German only, Qwen3's de->en outputs on FLORES are German (chrF
64.0 for the base model, 25.1 after SFT). This is either a bug in how adapters are
loaded or applied at inference, or a real effect of training on one direction with
one fixed instruction. These conditions separate the two, on the first --n FLORES
devtest sentences, with the same generation code as p2_translate.py (greedy):

  1. bug check      SFT loaded, adapter switched off -> must reproduce the base model
                    (compared with the stored base outputs); adapter on, unmerged ->
                    must reproduce the stored (merged) SFT outputs
  2. dose-response  the adapter's update scaled by 0, 0.25, 0.5, 0.75, 1: a gradual
                    slide into German means learned behaviour, not a loading error
  3. which part     fr->en (non-German input, English target) and de->zh (German
                    input, non-English target): is the target "English" ignored, or
                    does German input trigger it?
  4. prompt         the instruction after the text instead of before it
  5. controls       en->zh and fr->de with the adapter on

Writes results/p2/<model>/diagnostics/direction.{json,md} and the outputs.

Example:
  python scripts/p2_diagnose_direction.py --model qwen3-1.7b
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from mtrlhf.io import read_jsonl, write_json, write_jsonl  # noqa: E402

# (label, source, target, adapter scale or None = adapter off, prompt variant)
CONDITIONS = [
    ("de→en, adapter off", "de", "en", None, "standard"),
    ("de→en, adapter ×0.25", "de", "en", 0.25, "standard"),
    ("de→en, adapter ×0.5", "de", "en", 0.5, "standard"),
    ("de→en, adapter ×0.75", "de", "en", 0.75, "standard"),
    ("de→en, adapter ×1 (SFT)", "de", "en", 1.0, "standard"),
    ("de→en, SFT, instruction after the text", "de", "en", 1.0, "after"),
    ("fr→en, adapter off", "fr", "en", None, "standard"),
    ("fr→en, SFT", "fr", "en", 1.0, "standard"),
    ("de→zh, adapter off", "de", "zh", None, "standard"),
    ("de→zh, SFT", "de", "zh", 1.0, "standard"),
    ("en→zh, SFT (control)", "en", "zh", 1.0, "standard"),
    ("fr→de, SFT (control)", "fr", "de", 1.0, "standard"),
]


def messages(src: str, s: str, t: str, variant: str) -> list[dict]:
    from mtrlhf.data.prompts import LANG_NAMES, build_messages

    if variant == "standard":
        return build_messages(src, s, t)
    a, b = LANG_NAMES[s], LANG_NAMES[t]
    return [{"role": "user", "content": f"{src}\n\nTranslate the {a} text above into {b}. "
                                        f"Reply with the {b} translation only."}]


def lora_scaler(model):
    """Return set_scale(s): multiply every LoRA layer's update by s (1.0 = as trained)."""
    from peft.tuners.lora import LoraLayer

    layers = [(m, dict(m.scaling)) for m in model.modules() if isinstance(m, LoraLayer)]
    if not layers:
        raise RuntimeError("no LoRA layers found")

    def set_scale(s: float) -> None:
        for m, orig in layers:
            for k, v in orig.items():
                m.scaling[k] = v * s

    return set_scale


def load(system: str, store, base_model: str | None, device: str):
    """Base model with the system's adapter chain: all but the last merged, the last kept as LoRA."""
    from peft import PeftModel
    from transformers import AutoModelForCausalLM

    from mtrlhf.compat import ignore_incompatible_torchao
    from mtrlhf.models import adapter_chain, load_tokenizer, neutral_generation_config, pick_dtype

    b, chain = adapter_chain(system, store)
    base_model = b or base_model
    tok = load_tokenizer(base_model)
    model = AutoModelForCausalLM.from_pretrained(base_model, dtype=pick_dtype(device, base_model))
    ignore_incompatible_torchao()
    for name in chain[:-1]:
        model = PeftModel.from_pretrained(model, store.ensure(name)).merge_and_unload()
    model = PeftModel.from_pretrained(model, store.ensure(chain[-1]))
    neutral_generation_config(model, tok)
    return model.to(device).eval(), tok


def summarize(rows: list[dict], tgt: str) -> dict:
    from sacrebleu.metrics import CHRF

    from mtrlhf.rewards.guards import _lid

    langs = collections.Counter(_lid().classify(r["hyp"])[0] if r["hyp"].strip() else "empty" for r in rows)
    n = len(rows)
    return {"n": n, "target_lang_pct": 100 * langs.get(tgt, 0) / n,
            "languages": dict(langs.most_common(4)),
            "exact_copy_pct": 100 * sum(r["hyp"].strip() == r["src"].strip() for r in rows) / n,
            "chrf": CHRF().corpus_score([r["hyp"] for r in rows], [[r["ref"] for r in rows]]).score}


def same_as(rows: list[dict], stored_path: str) -> float | None:
    if not os.path.exists(stored_path):
        return None
    stored = {r["src"]: r["hyp"].strip() for r in read_jsonl(stored_path)}
    hits = [stored[r["src"]] == r["hyp"].strip() for r in rows if r["src"] in stored]
    return 100 * sum(hits) / len(hits) if hits else None


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default=None, help="model tag, e.g. qwen3-1.7b")
    ap.add_argument("--system", default="sft")
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--base-model", default=None)
    ap.add_argument("--flores-dir", default="data/p2/raw")
    ap.add_argument("--texts", default=None, help="JSON {lang: [sentences]} instead of FLORES devtest (tests)")
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--batch-size", type=int, default=32)  # as in p2_translate.py
    ap.add_argument("--max-new-tokens", type=int, default=200)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    from mtrlhf.models import CheckpointStore, Translator, default_device, model_paths
    from mtrlhf.data.prompts import extract_translation

    P = model_paths(args.model)
    store = CheckpointStore.from_env(args.ckpt or P["ckpt"], None, P["hub_prefix"])
    out_dir = args.out or os.path.join(os.path.dirname(P["outputs"]), "diagnostics")
    langs = sorted({c[1] for c in CONDITIONS} | {c[2] for c in CONDITIONS})
    if args.texts:
        texts = json.load(open(args.texts))
    else:
        from mtrlhf.data.testsets import load_flores

        flores = load_flores(args.flores_dir, langs)
        texts = {lang: flores[("devtest", lang)] for lang in langs}
    texts = {k: v[: args.n] for k, v in texts.items()}

    device = default_device(0)
    model, tok = load(args.system, store, args.base_model or P["model_id"], device)
    set_scale = lora_scaler(model)
    tr = Translator(model, tok, kind="causal", batch_size=args.batch_size)

    results, all_rows = [], []
    for label, s, t, scale, variant in CONDITIONS:
        srcs, refs = texts[s], texts[t]
        tr._inputs = lambda sources, s_, t_, v=variant: [  # same template call as Translator._inputs
            tok.apply_chat_template(messages(x, s_, t_, v), tokenize=False, add_generation_prompt=True)
            for x in sources]
        if scale is None:
            with model.disable_adapter():
                raw = tr.generate(srcs, s, t, max_new_tokens=args.max_new_tokens)
        else:
            set_scale(scale)
            raw = tr.generate(srcs, s, t, max_new_tokens=args.max_new_tokens)
            set_scale(1.0)
        rows = [{"condition": label, "src": x, "ref": y, "hyp": extract_translation(o[0], x)[0]}
                for x, y, o in zip(srcs, refs, raw)]
        r = {"condition": label, "src": s, "tgt": t, "scale": scale, "prompt": variant, **summarize(rows, t)}
        if (s, t, variant) == ("de", "en", "standard") and scale in (None, 1.0):
            stored = "base" if scale is None else args.system
            r["same_as_stored_pct"] = same_as(rows, os.path.join(P["outputs"], stored, "flores_devtest.de-en.jsonl"))
            r["stored_system"] = stored
        results.append(r)
        all_rows += rows
        print(f"{label:42s} in {t}: {r['target_lang_pct']:5.1f}%  exact copy: {r['exact_copy_pct']:5.1f}%  "
              f"chrF {r['chrf']:5.1f}  {r['languages']}"
              + (f"  same as stored {r['stored_system']}: {r['same_as_stored_pct']:.0f}%"
                 if r.get("same_as_stored_pct") is not None else ""), flush=True)

    os.makedirs(out_dir, exist_ok=True)
    write_json(os.path.join(out_dir, "direction.json"), {"system": args.system, "n": args.n, "results": results})
    write_jsonl(os.path.join(out_dir, "direction_outputs.jsonl"), all_rows)
    md = [f"# Direction diagnostic · {args.model or 'default'} · {args.system}", "",
          f"First {args.n} FLORES devtest sentences, greedy decoding, same generation code as the evaluation. "
          "*Same as stored* compares with the outputs saved by `p2_translate.py` (base model / merged adapter); "
          "a few differences are expected from different batching (padding) in half precision.", "",
          "| Condition | In target language % | Exact copy of source % | chrF | Languages | Same as stored % |",
          "|---|---|---|---|---|---|"]
    for r in results:
        md.append(f"| {r['condition']} | {r['target_lang_pct']:.0f} | {r['exact_copy_pct']:.0f} | {r['chrf']:.1f} | "
                  + ", ".join(f"{k} {v}" for k, v in r["languages"].items()) + " | "
                  + (f"{r['same_as_stored_pct']:.0f} ({r['stored_system']})"
                     if r.get("same_as_stored_pct") is not None else "–") + " |")
    md += ["", "Examples (first 3 sentences):", ""]
    for label in ("de→en, adapter off", "de→en, adapter ×0.5", "de→en, adapter ×1 (SFT)", "fr→en, SFT", "de→zh, SFT"):
        for row in [x for x in all_rows if x["condition"] == label][:3]:
            md.append(f"- *{label}*: {row['hyp'][:160]}")
    with open(os.path.join(out_dir, "direction.md"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(md) + "\n")
    print(f"-> {out_dir}/direction.md")


if __name__ == "__main__":
    main()
