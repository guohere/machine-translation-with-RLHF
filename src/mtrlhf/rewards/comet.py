"""COMET-family rewards (CometKiwi, COMET-22, xCOMET) via `unbabel-comet`."""

from __future__ import annotations

import gc
from typing import Sequence

from .base import Reward


class CometReward(Reward):
    family = "comet"

    def __init__(
        self,
        name: str,
        model_id: str,
        needs_reference: bool,
        batch_size: int = 32,
        gpus: int | None = None,
        progress_bar: bool = True,
    ):
        import torch
        import transformers
        from comet import download_model, load_from_checkpoint

        if int(transformers.__version__.split(".")[0]) >= 5:
            raise RuntimeError(
                f"unbabel-comet 2.2.7 does not work with transformers {transformers.__version__}: "
                "its XLM-R encoder expects the transformers 4 output layout. "
                'Install "transformers>=4.40,<5" (pip install -e ".[phase1]" does this).'
            )
        self.name = name
        self.model_id = model_id
        self.needs_reference = needs_reference
        self.batch_size = batch_size
        self.gpus = (1 if torch.cuda.is_available() else 0) if gpus is None else gpus
        self.progress_bar = progress_bar
        # Gated models (CometKiwi, xCOMET) need HF_TOKEN in the environment and the
        # license accepted on the model page.
        self._model = load_from_checkpoint(download_model(model_id))

    def score(
        self,
        sources: Sequence[str],
        hypotheses: Sequence[str],
        references: Sequence[str] | None = None,
    ) -> list[float]:
        self._check_inputs(sources, hypotheses, references)
        if not hypotheses:
            return []
        samples = [{"src": s, "mt": h} for s, h in zip(sources, hypotheses)]
        if self.needs_reference:
            for sample, ref in zip(samples, references):
                sample["ref"] = ref
        out = self._model.predict(
            samples,
            batch_size=self.batch_size,
            gpus=self.gpus,
            progress_bar=self.progress_bar,
        )
        return [float(x) for x in out.scores]

    def close(self) -> None:
        self._model = None
        gc.collect()
        try:
            import torch

            torch.cuda.empty_cache()
        except Exception:
            pass
