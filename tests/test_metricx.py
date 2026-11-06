"""MetricX wrapper: input format, EOS handling and batching, with a tiny random mT5."""

import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

from mtrlhf.rewards import metricx  # noqa: E402


class FakeTokenizer:
    """Character-level stand-in for the mT5 tokenizer (appends EOS id 1)."""

    pad_token_id = 0

    def __init__(self):
        self.seen = []

    def __call__(self, texts, max_length, truncation, padding):
        self.seen.extend(texts)
        ids = [[2 + (ord(c) % 200) for c in t][: max_length - 1] + [1] for t in texts]
        return {"input_ids": ids, "attention_mask": [[1] * len(x) for x in ids]}


def tiny_model():
    torch.manual_seed(0)
    cfg = transformers.MT5Config(vocab_size=250100, d_model=16, d_kv=4, d_ff=32, num_layers=1,
                                 num_decoder_layers=1, num_heads=2, decoder_start_token_id=0)
    return transformers.MT5ForConditionalGeneration(cfg)


def make(qe=True, **kw):
    return metricx.MetricX24Reward("metricx24_qe" if qe else "metricx24_ref", qe=qe,
                                   model=tiny_model(), tokenizer=FakeTokenizer(), device="cpu", **kw)


def test_input_format_matches_official_predict_script():
    assert metricx.make_input("S", "H", None, qe=True) == "source: S candidate: H"
    assert metricx.make_input("S", "H", "R", qe=False) == "source: S candidate: H reference: R"


def test_eos_is_removed():
    r = make()
    ids = r._encode(["abc"])
    assert ids == [[2 + ord("a") % 200, 2 + ord("b") % 200, 2 + ord("c") % 200]]


def test_scores_are_negated_errors_and_batch_independent():
    srcs = ["The cat sleeps.", "A much longer source sentence with many more words in it.", "Hi"]
    hyps = ["Die Katze schläft.", "Ein viel längerer Satz mit vielen weiteren Wörtern darin.", "Hallo"]
    big = make(max_tokens_per_batch=10_000)
    small = make(max_tokens_per_batch=1)  # forces one input per batch
    a = big.score(srcs, hyps)
    b = small.score(srcs, hyps)
    assert all(-25.0 <= x <= 0.0 for x in a)
    assert a == pytest.approx(b, abs=1e-5)  # padding must not change scores
    assert big.error_scores(srcs, hyps) == pytest.approx([-x for x in a])


def test_reference_mode_requires_references():
    r = make(qe=False)
    with pytest.raises(ValueError):
        r.score(["s"], ["h"])
    r.score(["s"], ["h"], ["ref"])
    assert r.tokenizer.seen[-1] == "source: s candidate: h reference: ref"


def test_make_batches_respects_budget():
    lengths = [5, 50, 20, 20, 3]
    batches = metricx.make_batches(lengths, max_tokens=60, max_batch_size=8)
    assert sorted(i for b in batches for i in b) == list(range(5))
    for b in batches:
        assert len(b) == 1 or max(lengths[i] for i in b) * len(b) <= 60


def test_dtype_never_fp16_on_cpu():
    assert metricx.resolve_dtype("auto", "cpu") == torch.float32
