#!/usr/bin/env python
"""Phase 2: train one system (SFT, RSFT, DPO or GRPO) from a YAML config.

The adapter is saved to ckpt/<model>/<name>/ (and to the Hugging Face Hub mirror),
the training log to runs/p2/<model>/<name>/.

Examples:
  python scripts/p2_train.py --model qwen3-1.7b --config configs/p2/sft.yaml
  python scripts/p2_train.py --model gemma3-1b --config configs/p2/grpo.yaml
  python scripts/p2_train.py --model qwen3-1.7b --config configs/p2/grpo.yaml --set beta=0.0 name=grpo_beta0
  python scripts/p2_train.py --model qwen3-1.7b --config configs/p2/sft.yaml --debug   # few-step test run
"""

from __future__ import annotations

import argparse
import os
import sys

# Train on one GPU. With two visible GPUs the HF Trainer would wrap the model in
# DataParallel; to use both, run two trainings with CUDA_VISIBLE_DEVICES=0 / =1.
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))



def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE", help="override config values")
    ap.add_argument("--model", default=None, help="policy model: qwen3-1.7b | gemma3-1b (sets the model id and per-model folders)")
    ap.add_argument("--ckpt", default=None, help="local adapter store (default: ckpt/<model>)")
    ap.add_argument("--hub-repo", default=None,
                    help='private HF repo mirroring the adapters ("auto" = <user>/mtrlhf-en-de-adapters, "none" = off; default: $MTRLHF_HUB_REPO)')
    ap.add_argument("--debug", action="store_true", help="tiny model and data, a few steps, separate store")
    ap.add_argument("--force", action="store_true", help="retrain even if the adapter already exists")
    args = ap.parse_args()

    from mtrlhf.models import CheckpointStore
    from mtrlhf.train.methods import build_config, run

    cfg, ckpt, hub_prefix = build_config(args.config, args.model, args.debug, args.set, args.ckpt)
    store = CheckpointStore(ckpt) if args.debug else CheckpointStore.from_env(ckpt, args.hub_repo, hub_prefix)
    if store.has(cfg["name"]) and not args.force:
        trained_on = store.meta(cfg["name"]).get("data_sampling")
        if cfg.get("data_sampling") and trained_on != cfg["data_sampling"]:
            print(f"adapter '{cfg['name']}' was trained on samples drawn with {trained_on}; the current samples "
                  f"use {cfg['data_sampling']}: retraining")
        else:
            print(f"adapter '{cfg['name']}' already exists in {store.root}"
                  f"{' / ' + store.hub_repo if store.hub_repo else ''}; skipping (use --force to retrain)")
            return
    run(cfg, store)


if __name__ == "__main__":
    main()
