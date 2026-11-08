"""SFT / RSFT, DPO and GRPO from one config dict.

Common keys: name, method (sft | dpo | grpo), parent ("base" or a trained system),
base_model, data, epochs, learning_rate, batch_size, grad_accum, max_length, lora,
seed, max_steps, limit. See configs/p2/*.yaml for the values used in the project.
"""

from __future__ import annotations

import json
import os

from mtrlhf.data.prompts import build_messages
from mtrlhf.io import read_jsonl, write_json
from mtrlhf.models import DEFAULT_BASE, CheckpointStore, load_causal

DEFAULTS = {
    "parent": "base",
    "base_model": DEFAULT_BASE,
    "epochs": 1,
    "learning_rate": 2e-4,
    "batch_size": 16,
    "grad_accum": 1,
    "max_length": 384,
    "warmup_ratio": 0.03,
    "lr_scheduler": "cosine",
    "seed": 0,
    "max_steps": -1,
    "limit": 0,
    "logging_steps": 10,
    "gradient_checkpointing": True,
    "lora": {"r": 16, "alpha": 32, "dropout": 0.05, "target_modules": "all-linear"},
    "out_root": "runs/p2",
}


def precision_kwargs(model_id: str) -> dict:
    """bf16 where the GPU has it; fp16 mixed precision on older GPUs (T4), except for
    models that overflow in fp16 (Gemma), which then train in plain fp32."""
    import torch

    from mtrlhf.models import bf16_supported, fp16_safe

    if not torch.cuda.is_available():
        return {"bf16": False, "fp16": False, "use_cpu": True}
    if bf16_supported():
        return {"bf16": True, "fp16": False}
    return {"bf16": False, "fp16": fp16_safe(model_id)}


def lora_config(cfg: dict):
    from peft import LoraConfig

    lc = cfg["lora"]
    return LoraConfig(r=lc["r"], lora_alpha=lc["alpha"], lora_dropout=lc["dropout"],
                      target_modules=lc["target_modules"], task_type="CAUSAL_LM")


def common_args(cfg: dict, out_dir: str, model_id: str) -> dict:
    return dict(
        output_dir=out_dir,
        num_train_epochs=cfg["epochs"],
        max_steps=cfg["max_steps"],
        learning_rate=cfg["learning_rate"],
        per_device_train_batch_size=cfg["batch_size"],
        gradient_accumulation_steps=cfg["grad_accum"],
        warmup_ratio=cfg["warmup_ratio"],
        lr_scheduler_type=cfg["lr_scheduler"],
        logging_steps=cfg["logging_steps"],
        gradient_checkpointing=cfg["gradient_checkpointing"],
        seed=cfg["seed"],
        report_to="none",
        **precision_kwargs(model_id),
    )


def _rows(path: str, limit: int) -> list[dict]:
    rows = read_jsonl(path)
    if not rows:
        raise FileNotFoundError(f"No data in {path}")
    return rows[:limit] if limit else rows


def _messages(r: dict, variants: bool = False) -> list[dict]:
    from mtrlhf.data.prompts import prompt_variant

    v = prompt_variant(str(r.get("id", r["src"]))) if variants else 0
    return build_messages(r["src"], r.get("src_lang", "en"), r.get("tgt_lang", "de"), v)


def _finish(trainer, cfg: dict, store: CheckpointStore, base_model: str, out_dir: str, extra: dict) -> str:
    trainer.save_model(out_dir)
    write_json(os.path.join(out_dir, "log_history.json"), trainer.state.log_history)
    path = store.save(cfg["name"], out_dir, base_model, cfg["parent"] if cfg["parent"] != "base" else None,
                      extra={"method": cfg["method"], "data_sampling": cfg.get("data_sampling"),
                             "recipe": cfg.get("recipe"), **extra})
    print(f"saved adapter '{cfg['name']}' -> {path}")
    return path


def train_sft(cfg: dict, store: CheckpointStore) -> str:
    """Supervised fine-tuning on (src, ref|tgt) pairs; loss on the translation only.
    RSFT is the same method trained on the best sampled translations, with parent "sft"."""
    from datasets import Dataset
    from trl import SFTConfig, SFTTrainer

    def example(r):
        return {"prompt": _messages(r, cfg.get("prompt_variants", False)),
                "completion": [{"role": "assistant", "content": r.get("tgt") or r["ref"]}]}

    train = Dataset.from_list([example(r) for r in _rows(cfg["data"], cfg["limit"])])
    model, tok, base = load_causal(cfg["parent"], store, cfg["base_model"])
    out_dir = os.path.join(cfg["out_root"], cfg["name"])
    args = SFTConfig(**common_args(cfg, out_dir, base), max_length=cfg["max_length"], save_strategy="no",
                     completion_only_loss=True)
    trainer = SFTTrainer(model=model, args=args, train_dataset=train, processing_class=tok,
                         peft_config=lora_config(cfg))
    trainer.train()
    return _finish(trainer, cfg, store, base, out_dir, {"n_train": len(train)})


def train_dpo(cfg: dict, store: CheckpointStore) -> str:
    """DPO on (src, chosen, rejected) pairs. The reference policy is the parent system
    (the LoRA adapter is disabled to compute reference log-probabilities)."""
    from datasets import Dataset
    from trl import DPOConfig, DPOTrainer

    def example(r):
        return {"prompt": _messages(r),
                "chosen": [{"role": "assistant", "content": r["chosen"]}],
                "rejected": [{"role": "assistant", "content": r["rejected"]}]}

    train = Dataset.from_list([example(r) for r in _rows(cfg["data"], cfg["limit"])])
    model, tok, base = load_causal(cfg["parent"], store, cfg["base_model"])
    out_dir = os.path.join(cfg["out_root"], cfg["name"])
    args = DPOConfig(**common_args(cfg, out_dir, base), beta=cfg.get("beta", 0.1), max_length=cfg["max_length"],
                     save_strategy="no")
    trainer = DPOTrainer(model=model, args=args, train_dataset=train, processing_class=tok,
                         peft_config=lora_config(cfg))
    trainer.train()
    return _finish(trainer, cfg, store, base, out_dir, {"n_pairs": len(train), "beta": args.beta})


def train_grpo(cfg: dict, store: CheckpointStore) -> str:
    """Online GRPO against the Phase 1 training reward, with guard checks logged
    (weight 0) or applied as penalties (guard_weights > 0). Intermediate adapters are
    exported every save_steps as '<name>-step<k>' for the over-optimization curve."""
    from datasets import Dataset
    from transformers import TrainerCallback
    from trl import GRPOConfig, GRPOTrainer

    from mtrlhf.models import default_device
    from mtrlhf.rewards import get_reward
    from mtrlhf.rewards.guards import GUARDS

    from .rewards import make_guard, make_quality_reward

    rows = _rows(cfg["data"], cfg["limit"])
    train = Dataset.from_list([{"prompt": _messages(r), "source": r["src"], "ref": r.get("ref") or ""}
                               for r in rows])
    model, tok, base = load_causal(cfg["parent"], store, cfg["base_model"])
    out_dir = os.path.join(cfg["out_root"], cfg["name"])
    n_gen = cfg.get("num_generations", 8)

    reward_kw = {"device": cfg.get("reward_device") or default_device()} if "metricx" in cfg["reward"] else {}
    reward = get_reward(cfg["reward"], **reward_kw)
    funcs = [make_quality_reward(reward, os.path.join(out_dir, "samples.jsonl"),
                                 log_every=cfg.get("sample_log_every", 25), num_generations=n_gen)]
    weights = [1.0]
    gw = cfg.get("guard_weights") or {}
    for g in GUARDS:
        funcs.append(make_guard(g))
        weights.append(float(gw.get(g, 0.0)))

    args = GRPOConfig(
        **common_args(cfg, out_dir, base),
        num_generations=n_gen,
        max_completion_length=cfg.get("max_completion_length", 160),
        temperature=cfg.get("temperature", 1.0),
        top_p=cfg.get("top_p", 1.0),
        beta=cfg.get("beta", 0.0),
        reward_weights=weights,
        save_strategy="steps",
        save_steps=cfg.get("save_steps", 125),
        save_only_model=True,
    )

    class ExportAdapters(TrainerCallback):
        def on_save(self, a, state, control, **kw):
            ckpt = os.path.join(a.output_dir, f"checkpoint-{state.global_step}")
            if os.path.isdir(ckpt):
                store.save(f"{cfg['name']}-step{state.global_step}", ckpt, base, cfg["parent"],
                           extra={"method": "grpo", "step": state.global_step,
                                  "temperature": a.temperature, "top_p": a.top_p})

    prompts_per_step = args.per_device_train_batch_size * args.gradient_accumulation_steps // n_gen
    print(f"GRPO: {len(train)} prompts, {prompts_per_step} prompts x {n_gen} samples per step, "
          f"guard weights {dict(zip(GUARDS, weights[1:]))}")
    trainer = GRPOTrainer(model=model, reward_funcs=funcs, args=args, train_dataset=train, processing_class=tok,
                          peft_config=lora_config(cfg), callbacks=[ExportAdapters()])
    trainer.train()
    path = _finish(trainer, cfg, store, base, out_dir,
                   {"n_prompts": len(train), "num_generations": n_gen, "beta": args.beta,
                    "temperature": args.temperature, "top_p": args.top_p,
                    "reward": cfg["reward"], "guard_weights": gw, "prompts_per_step": prompts_per_step})
    reward.close()
    return path


METHODS = {"sft": train_sft, "dpo": train_dpo, "grpo": train_grpo}

# --debug keeps the real model (so GPU memory and chat templates are tested) but uses
# a few examples and steps, and a separate local store that never touches the Hub.
DEBUG = {"limit": 64, "max_steps": 4, "logging_steps": 1, "save_steps": 2, "batch_size": 4, "grad_accum": 1,
         "num_generations": 4}


def build_config(path: str, model: str | None = None, debug: bool = False, sets: list[str] = (),
                 ckpt: str | None = None) -> tuple[dict, str, str]:
    """YAML config + per-model overrides + debug settings + KEY=VALUE overrides, with the
    {shared} / {model_data} data placeholders resolved. Returns (cfg, ckpt_root, hub_prefix)."""
    import yaml

    from mtrlhf.models import model_paths

    paths = model_paths(model)
    with open(path) as fh:
        cfg = yaml.safe_load(fh)
    cfg.update(paths["overrides"].get(cfg["method"], {}))
    if cfg["method"] == "grpo":  # same sampling setting as candidate sampling and best-of-N
        from mtrlhf.sampling import load_sampling

        chosen = load_sampling()
        if chosen:
            cfg["temperature"], cfg["top_p"] = chosen["temperature"], chosen["top_p"]
        else:
            print("(no sampling pilot result: GRPO samples with the config's temperature/top_p)")
    if model:
        cfg["base_model"] = paths["model_id"]
    cfg["out_root"] = paths["runs"]
    ckpt = ckpt or paths["ckpt"]
    if debug:
        cfg.update({k: v for k, v in DEBUG.items() if not (k == "num_generations" and cfg["method"] != "grpo")})
        cfg["out_root"] = paths["runs"] + "_debug"
        ckpt = ckpt + "_debug"
    for kv in sets:  # explicit overrides win
        k, v = kv.split("=", 1)
        cfg[k] = yaml.safe_load(v)
    for k, v in cfg.items():
        if isinstance(v, str):
            cfg[k] = v.replace("{shared}", "data/p2").replace("{model_data}", paths["model_data"])
    cfg["data_sampling"] = data_sampling(cfg.get("data", ""))
    return cfg, ckpt, paths["hub_prefix"]


def data_sampling(data_path: str) -> dict | None:
    """Sampling setting behind sampled training data (RSFT targets, DPO pairs), if any."""
    import json

    stats = os.path.join(os.path.dirname(data_path), "pairs_stats.json")
    if os.path.basename(data_path) in ("rsft.jsonl", "dpo_pairs.jsonl") and os.path.exists(stats):
        with open(stats) as fh:
            return json.load(fh).get("sampling")
    return None


def run(cfg: dict, store: CheckpointStore) -> str:
    from mtrlhf.compat import ignore_incompatible_torchao

    ignore_incompatible_torchao()
    full = {**DEFAULTS, **cfg, "lora": {**DEFAULTS["lora"], **(cfg.get("lora") or {})}}
    print("config:", json.dumps({k: v for k, v in full.items() if k != "lora"}, default=str))
    return METHODS[full["method"]](full, store)
