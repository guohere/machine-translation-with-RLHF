"""Reward functions in the form TRL's GRPOTrainer expects.

TRL calls each function as f(prompts=..., completions=..., **dataset_columns) and
logs its mean under the function's name. The main reward is the Phase 1 choice
(R_train); the guards are separate functions so that their rates are logged
during every run. Plain GRPO gives them weight 0 (logged only), guarded GRPO a
negative weight per violation.
"""

from __future__ import annotations

import json
import os
import random

from mtrlhf.data.prompts import extract_translation
from mtrlhf.rewards import guards


def completion_text(completion) -> str:
    """Conversational completions are [{"role": "assistant", "content": ...}]."""
    if isinstance(completion, list):
        return completion[-1]["content"] if completion else ""
    return str(completion)


def make_quality_reward(reward, sample_log: str | None = None, log_every: int = 25, log_groups: int = 2,
                        num_generations: int = 8):
    """Wrap a Reward (e.g. MetricX-24-QE) as a TRL reward function.

    Every `log_every` steps, the completions of `log_groups` prompts are appended to
    `sample_log` (JSONL) with their scores and guard flags, so Phase 3 can see how
    outputs change while the training reward rises. TRL places the completions of
    one prompt next to each other, so the first log_groups * num_generations rows
    are whole groups.
    """
    logged_steps: set[int] = set()

    def quality(prompts, completions, source, ref=None, trainer_state=None, **kwargs):
        hyps = [extract_translation(completion_text(c), s)[0] for c, s in zip(completions, source)]
        refs = ref if reward.needs_reference else None
        scores = reward.score(list(source), hyps, list(refs) if refs is not None else None)
        step = getattr(trainer_state, "global_step", 0) or 0
        if sample_log and step % log_every == 0 and step not in logged_steps:
            logged_steps.add(step)
            rows = []
            for i in range(min(len(hyps), log_groups * num_generations)):
                rows.append({"step": step, "source": source[i], "hyp": hyps[i], "score": scores[i],
                             **{f"flag_{k}": v for k, v in guards.guard_flags(source[i], hyps[i]).items()}})
            os.makedirs(os.path.dirname(sample_log) or ".", exist_ok=True)
            with open(sample_log, "a", encoding="utf-8") as fh:
                for r in rows:
                    fh.write(json.dumps(r, ensure_ascii=False) + "\n")
        return scores

    quality.__name__ = reward.name
    return quality


def make_guard(name: str, tgt_lang: str = "de"):
    """A reward function returning -1 for a flagged output and 0 otherwise."""
    check = {
        "wrong_language": lambda s, h: guards.wrong_language(h, tgt_lang),
        "number_mismatch": guards.number_mismatch,
        "length_ratio": guards.length_ratio,
        "copy": guards.copy,
    }[name]

    def guard(prompts, completions, source, **kwargs):
        out = []
        for c, s in zip(completions, source):
            h = extract_translation(completion_text(c), s)[0]
            out.append(-1.0 if check(s, h) else 0.0)
        return out

    guard.__name__ = f"guard_{name}"
    return guard


def subsample(rows: list, n: int, seed: int = 0) -> list:
    if n <= 0 or n >= len(rows):
        return rows
    return random.Random(seed).sample(rows, n)
