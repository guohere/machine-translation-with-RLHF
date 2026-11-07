"""Surface-overlap metrics (sentence-level chrF and BLEU) as reference-based rewards."""

from __future__ import annotations

from typing import Sequence

from .base import Reward


class ChrFReward(Reward):
    name = "chrf"
    family = "surface"
    needs_reference = True

    def __init__(self, name: str = "chrf"):
        from sacrebleu.metrics import CHRF

        self.name = name
        self._metric = CHRF()

    def score(self, sources, hypotheses, references=None) -> list[float]:
        self._check_inputs(sources, hypotheses, references)
        return [self._metric.sentence_score(h, [r]).score for h, r in zip(hypotheses, references)]


class BleuReward(Reward):
    name = "bleu"
    family = "surface"
    needs_reference = True

    def __init__(self, name: str = "bleu"):
        from sacrebleu.metrics import BLEU

        self.name = name
        self._metric = BLEU(effective_order=True)

    def score(self, sources: Sequence[str], hypotheses, references=None) -> list[float]:
        self._check_inputs(sources, hypotheses, references)
        return [self._metric.sentence_score(h, [r]).score for h, r in zip(hypotheses, references)]
