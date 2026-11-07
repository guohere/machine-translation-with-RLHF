#!/usr/bin/env python
"""Phase 2: train one system (SFT, RSFT, DPO or GRPO) from a YAML config.

The adapter is saved to ckpt/<name>/ (and to the Hugging Face Hub mirror if
--hub-repo is set), the training log to runs/p2/<name>/.

Examples:
  python scripts/p2_train.py --config configs/p2/sft.yaml
  python scripts/p2_train.py --config configs/p2/grpo.yaml --set beta=0.0 name=grpo_beta0
  python scripts/p2_train.py --config configs/p2/sft.yaml --debug     # tiny run to test the setup
"""

from __future__ import annotations

import argparse
import os
import sys

# Train on one GPU. With two visible GPUs the HF Trainer would wrap the model in
# DataParallel; to use both, run two trainings with CUDA_VISIBLE_DEVICES=0 / =1.
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

DEBUG = {"base_model": "Qwen/Qwen2.5-0.5B-Instruct", "limit": 64, "max_steps": 4, "logging_steps": 1,
         "save_steps": 2, "batch_size": 4, "grad_accum": 1, "num_generations": 4}


def parse_value(v: str):
    import yaml

    return yaml.safe_load(v)


def main() -> None:
    import yaml

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE", help="override config values")
    ap.add_argument("--ckpt", default="ckpt", help="local adapter store")
    ap.add_argument("--hub-repo", default=None,
                    help='private HF repo mirroring the adapters ("auto" = <user>/mtrlhf-en-de-adapters, "none" = off; default: $MTRLHF_HUB_REPO)')
    ap.add_argument("--debug", action="store_true", help="tiny model and data, a few steps, separate store")
    ap.add_argument("--force", action="store_true", help="retrain even if the adapter already exists")
    args = ap.parse_args()

    from mtrlhf.models import CheckpointStore
    from mtrlhf.train.methods import run

    with open(args.config) as fh:
        cfg = yaml.safe_load(fh)
    ckpt = args.ckpt
    if args.debug:
        cfg.update({k: v for k, v in DEBUG.items() if not (k == "num_generations" and cfg.get("method") != "grpo")})
        cfg["out_root"] = "runs/p2_debug"
        ckpt = args.ckpt + "_debug"
    for kv in args.set:  # explicit overrides win over --debug
        k, v = kv.split("=", 1)
        cfg[k] = parse_value(v)
    store = CheckpointStore(ckpt) if args.debug else CheckpointStore.from_env(ckpt, args.hub_repo)
    if store.has(cfg["name"]) and not args.force:
        print(f"adapter '{cfg['name']}' already exists in {store.root}"
              f"{' / ' + store.hub_repo if store.hub_repo else ''}; skipping (use --force to retrain)")
        return
    run(cfg, store)


if __name__ == "__main__":
    main()
