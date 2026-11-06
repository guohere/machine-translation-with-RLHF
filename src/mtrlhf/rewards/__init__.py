"""Reward registry. `get_reward(name)` builds a reward; `reward_info(name)` describes it
without loading any model."""

from __future__ import annotations

from .base import Reward

# family: model family; the held-out evaluator must come from a different one.
# stretch: optional candidates that are slower or larger than the core set.
REGISTRY: dict[str, dict] = {
    "cometkiwi22": dict(kind="comet", model_id="Unbabel/wmt22-cometkiwi-da",
                        needs_reference=False, family="comet", gated=True),
    "comet22": dict(kind="comet", model_id="Unbabel/wmt22-comet-da",
                    needs_reference=True, family="comet"),
    "metricx24_qe": dict(kind="metricx", model_id="google/metricx-24-hybrid-large-v2p6",
                         needs_reference=False, family="metricx"),
    "metricx24_ref": dict(kind="metricx", model_id="google/metricx-24-hybrid-large-v2p6",
                          needs_reference=True, family="metricx"),
    "chrf": dict(kind="chrf", needs_reference=True, family="surface"),
    "bleu": dict(kind="bleu", needs_reference=True, family="surface"),
    # Stretch candidates (3.5B parameters; slow on a T4).
    "cometkiwi23_xl": dict(kind="comet", model_id="Unbabel/wmt23-cometkiwi-da-xl",
                           needs_reference=False, family="comet", gated=True, stretch=True),
    "xcomet_xl_qe": dict(kind="comet", model_id="Unbabel/XCOMET-XL",
                         needs_reference=False, family="comet", gated=True, stretch=True),
}


def reward_info(name: str) -> dict:
    if name not in REGISTRY:
        raise KeyError(f"Unknown reward {name!r}. Known: {', '.join(sorted(REGISTRY))}")
    return dict(REGISTRY[name], name=name)


def get_reward(name: str, **overrides) -> Reward:
    info = reward_info(name)
    kind = info["kind"]
    if kind == "comet":
        from .comet import CometReward

        kw = {k: overrides[k] for k in ("batch_size", "gpus", "progress_bar") if k in overrides}
        return CometReward(name, info["model_id"], info["needs_reference"], **kw)
    if kind == "metricx":
        from .metricx import MetricX24Reward

        kw = {k: overrides[k] for k in ("max_tokens_per_batch", "dtype", "device", "max_input_length")
              if k in overrides}
        return MetricX24Reward(name, info["model_id"], qe=not info["needs_reference"], **kw)
    if kind == "chrf":
        from .surface import ChrFReward

        return ChrFReward(name)
    if kind == "bleu":
        from .surface import BleuReward

        return BleuReward(name)
    raise ValueError(f"Unknown reward kind {kind!r}")


__all__ = ["Reward", "REGISTRY", "get_reward", "reward_info"]
