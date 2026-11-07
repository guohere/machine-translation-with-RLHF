"""The single reward interface used by every phase of the project.

The same `Reward` object scores WMT outputs in the Phase 1 meta-evaluation,
labels preference pairs and provides the GRPO reward in Phase 2, and evaluates
systems in Phase 3. That way the reward that was validated is exactly the one
that gets optimized.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Sequence


class Reward(ABC):
    """Scores translations. Higher is always better."""

    #: Registry name, e.g. "cometkiwi22".
    name: str = "reward"
    #: Model family ("comet", "metricx", "surface"). The held-out evaluator is
    #: chosen from a different family than the training reward.
    family: str = "none"
    #: Whether a reference translation is required.
    needs_reference: bool = False

    @abstractmethod
    def score(
        self,
        sources: Sequence[str],
        hypotheses: Sequence[str],
        references: Sequence[str] | None = None,
    ) -> list[float]:
        """Return one score per hypothesis (higher is better)."""

    def close(self) -> None:
        """Release model memory. Safe to call more than once."""

    def _check_inputs(self, sources, hypotheses, references) -> None:
        if len(sources) != len(hypotheses):
            raise ValueError(
                f"{self.name}: {len(sources)} sources but {len(hypotheses)} hypotheses"
            )
        if self.needs_reference:
            if references is None or len(references) != len(hypotheses):
                raise ValueError(f"{self.name} needs one reference per hypothesis")
            if any(r is None for r in references):
                raise ValueError(f"{self.name}: missing reference")

    def __repr__(self) -> str:
        return f"{type(self).__name__}(name={self.name!r})"
