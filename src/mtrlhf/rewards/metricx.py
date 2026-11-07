"""MetricX-24 (hybrid) reward, in QE or reference-based mode.

Mirrors the official inference code in google-research/metricx
(`metricx24/predict.py` and `metricx24/models.py`) but uses the standard
`MT5ForConditionalGeneration` class, so it does not depend on MetricX's pinned
library versions:

* input text: "source: {src} candidate: {hyp}" (+ " reference: {ref}" when not QE)
* tokenize with the mT5-XL tokenizer, truncate to 1536 tokens, drop the final EOS
* one decoder step with decoder_input_ids = 0; the score is the logit of
  token 250089 (<extra_id_10>), clipped to [0, 25]

MetricX predicts an error score (0 = perfect, 25 = worst). `score()` returns
the negated value so that, like every Reward, higher is better.
"""

from __future__ import annotations

import gc
from typing import Sequence

from .base import Reward

SCORE_TOKEN_ID = 250089  # <extra_id_10>
MAX_ERROR = 25.0


def make_input(source: str, hypothesis: str, reference: str | None, qe: bool) -> str:
    """Build the MetricX-24 input string exactly as metricx24/predict.py does."""
    if qe:
        return "source: " + source + " candidate: " + hypothesis
    return "source: " + source + " candidate: " + hypothesis + " reference: " + reference


def resolve_dtype(dtype: str, device: str):
    """bf16 on GPUs that support it natively (Ampere+), otherwise fp32.

    fp16 is never used: T5-family models overflow in fp16. T4 GPUs (Kaggle,
    Colab) have no native bf16, so they run in fp32.
    """
    import torch

    if dtype == "float32":
        return torch.float32
    if dtype == "bfloat16":
        return torch.bfloat16
    if dtype != "auto":
        raise ValueError(f"Unsupported dtype {dtype!r} (use auto, float32 or bfloat16)")
    if str(device).startswith("cuda") and torch.cuda.is_available():
        major, _ = torch.cuda.get_device_capability(torch.device(device))
        if major >= 8:
            return torch.bfloat16
    return torch.float32


def make_batches(lengths: Sequence[int], max_tokens: int, max_batch_size: int) -> list[list[int]]:
    """Group indices (longest first) so that batch_size * max_len <= max_tokens."""
    order = sorted(range(len(lengths)), key=lambda i: -lengths[i])
    batches: list[list[int]] = []
    current: list[int] = []
    current_max = 0
    for i in order:
        new_max = max(current_max, lengths[i])
        if current and (
            new_max * (len(current) + 1) > max_tokens or len(current) >= max_batch_size
        ):
            batches.append(current)
            current, new_max = [], lengths[i]
        current.append(i)
        current_max = new_max
    if current:
        batches.append(current)
    return batches


class MetricX24Reward(Reward):
    family = "metricx"

    def __init__(
        self,
        name: str,
        model_id: str = "google/metricx-24-hybrid-large-v2p6",
        qe: bool = True,
        tokenizer_id: str = "google/mt5-xl",
        max_input_length: int = 1536,
        max_tokens_per_batch: int = 16384,
        max_batch_size: int = 64,
        dtype: str = "auto",
        device: str | None = None,
        model=None,
        tokenizer=None,
    ):
        import torch

        self.name = name
        self.model_id = model_id
        self.qe = qe
        self.needs_reference = not qe
        self.max_input_length = max_input_length
        self.max_tokens_per_batch = max_tokens_per_batch
        self.max_batch_size = max_batch_size
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = resolve_dtype(dtype, self.device)

        if tokenizer is None:
            from transformers import AutoTokenizer

            tokenizer = AutoTokenizer.from_pretrained(tokenizer_id)
        if model is None:
            model = self._load_model(model_id)
        if model.config.vocab_size <= SCORE_TOKEN_ID:
            raise ValueError("Model vocabulary does not contain the MetricX score token")
        self.tokenizer = tokenizer
        self.pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
        self.model = model.to(device=self.device, dtype=self.dtype).eval()

    @staticmethod
    def _load_model(model_id: str):
        from transformers import MT5ForConditionalGeneration

        # Loaded in fp32 and cast afterwards; avoids the torch_dtype/dtype kwarg
        # rename across transformers versions.
        try:
            model, info = MT5ForConditionalGeneration.from_pretrained(
                model_id, output_loading_info=True
            )
            missing = [k for k in info.get("missing_keys", []) if "embed_tokens" not in k]
            if missing:
                raise RuntimeError(f"MetricX checkpoint is missing weights: {missing[:5]}")
        except TypeError:
            model = MT5ForConditionalGeneration.from_pretrained(model_id)
        return model

    def _encode(self, texts: Sequence[str]) -> list[list[int]]:
        enc = self.tokenizer(
            list(texts), max_length=self.max_input_length, truncation=True, padding=False
        )
        # Drop the trailing EOS token, as the official predict.py does.
        return [ids[:-1] for ids in enc["input_ids"]]

    def error_scores(
        self,
        sources: Sequence[str],
        hypotheses: Sequence[str],
        references: Sequence[str] | None = None,
    ) -> list[float]:
        """MetricX's native scale: 0 (no errors) to 25 (worst)."""
        import torch

        self._check_inputs(sources, hypotheses, references)
        if not hypotheses:
            return []
        refs = references if references is not None else [None] * len(hypotheses)
        texts = [make_input(s, h, r, self.qe) for s, h, r in zip(sources, hypotheses, refs)]
        ids = self._encode(texts)
        out = [0.0] * len(ids)
        batches = make_batches([len(x) for x in ids], self.max_tokens_per_batch, self.max_batch_size)
        with torch.inference_mode():
            for batch in batches:
                width = max(len(ids[i]) for i in batch)
                input_ids = torch.full((len(batch), width), self.pad_id, dtype=torch.long)
                mask = torch.zeros((len(batch), width), dtype=torch.long)
                for row, i in enumerate(batch):
                    input_ids[row, : len(ids[i])] = torch.tensor(ids[i], dtype=torch.long)
                    mask[row, : len(ids[i])] = 1
                input_ids = input_ids.to(self.device)
                mask = mask.to(self.device)
                decoder_input_ids = torch.zeros((len(batch), 1), dtype=torch.long, device=self.device)
                logits = self.model(
                    input_ids=input_ids,
                    attention_mask=mask,
                    decoder_input_ids=decoder_input_ids,
                ).logits
                preds = logits[:, 0, SCORE_TOKEN_ID].float().clamp(0.0, MAX_ERROR).cpu().tolist()
                for row, i in enumerate(batch):
                    out[i] = preds[row]
        return out

    def score(
        self,
        sources: Sequence[str],
        hypotheses: Sequence[str],
        references: Sequence[str] | None = None,
    ) -> list[float]:
        return [-x for x in self.error_scores(sources, hypotheses, references)]

    def close(self) -> None:
        self.model = None
        gc.collect()
        try:
            import torch

            torch.cuda.empty_cache()
        except Exception:
            pass
