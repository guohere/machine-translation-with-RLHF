import gzip
import os
import io
import tarfile

import pytest

from mtrlhf.data import parallel, prompts, testsets
from mtrlhf.rewards import guards


# --- prompt and cleanup ------------------------------------------------------------

def test_messages_name_both_languages():
    m = prompts.build_messages("Hello world", "en", "de")
    assert m[0]["role"] == "user" and "English" in m[0]["content"] and "German" in m[0]["content"]
    assert m[0]["content"].endswith("Hello world")


@pytest.mark.parametrize("reply,src,expected", [
    ("Hallo Welt.", "Hello world.", "Hallo Welt."),
    ("German: Hallo Welt.", "Hello world.", "Hallo Welt."),
    ("**Übersetzung:** Hallo Welt.", "Hello world.", "Hallo Welt."),
    ("\"Hallo Welt.\"", "Hello world.", "Hallo Welt."),
    ("\"Hallo Welt.\"", "\"Hello world.\"", "\"Hallo Welt.\""),  # keep quotes the source has
    ("Hallo Welt.\n\nNote: 'world' can also mean 'Erde'.", "Hello world.", "Hallo Welt."),
])
def test_extract_translation(reply, src, expected):
    out, changed = prompts.extract_translation(reply, src)
    assert out == expected and changed == (out != reply.strip())


# --- guards ------------------------------------------------------------------------

def test_numbers_ignore_separators():
    assert guards.numbers("1,000.5 and 12%") == guards.numbers("1.000,5 und 12 %")
    assert not guards.number_mismatch("profits rose 12% in 2023", "Gewinne stiegen 2023 um 12 %")
    assert guards.number_mismatch("profits rose 12% in 2023", "Gewinne stiegen 2023 um 21 %")
    assert guards.number_mismatch("3 cats and 3 dogs", "3 Katzen und Hunde")  # counts matter


def test_wrong_language():
    pytest.importorskip("langid")
    assert not guards.wrong_language("Das Unternehmen meldete einen deutlichen Anstieg der Gewinne.")
    assert guards.wrong_language("The company reported a strong increase in profits this year.")
    assert not guards.wrong_language("Ja.")  # too short to judge


def test_length_and_copy():
    src = "The company reported a strong increase in profits this year."
    assert guards.length_ratio(src, "Ja.") and not guards.length_ratio(src, "Das Unternehmen meldete Gewinne.")
    assert guards.copy(src, src) and not guards.copy(src, "Das Unternehmen meldete einen Gewinnanstieg.")


# --- training data -----------------------------------------------------------------

def test_tsv_orientation_and_filters(tmp_path):
    rows = [("Die Regierung hat die Steuern nicht erhöht, und das ist gut.",
             "The government did not raise taxes, and that is good.")] * 3 + \
           [("Kurz.", "Short."), ("", "The empty side should be dropped here please.")]
    p = tmp_path / "nc.tsv.gz"
    with gzip.open(p, "wt", encoding="utf-8") as fh:
        for de, en in rows:
            fh.write(f"{de}\t{en}\n")
    pairs = parallel.read_tsv_pairs(str(p))
    assert pairs[0][0].startswith("The government")  # English first, whatever the file order
    assert parallel.keep_pair(*pairs[0]) and not parallel.keep_pair(*pairs[3]) and not parallel.keep_pair(*pairs[4])


def test_make_splits_disjoint_deterministic_and_excludes_tests():
    pairs = [(f"This is English sentence number {i} for the test.", f"Das ist deutscher Satz Nummer {i} für den Test.")
             for i in range(50)]
    excluded = parallel.normalize(pairs[7][0])
    a, stats = parallel.make_splits(pairs, {"sft_train": 20, "prompts": 10}, exclude={excluded}, seed=1,
                                    check_language=False)
    b, _ = parallel.make_splits(pairs, {"sft_train": 20, "prompts": 10}, exclude={excluded}, seed=1,
                                check_language=False)
    assert a == b
    srcs = [r["src"] for r in a["sft_train"]] + [r["src"] for r in a["prompts"]]
    assert len(set(srcs)) == 30 and pairs[7][0] not in srcs
    assert stats["n_sft_train"] == 20 and stats["n_prompts"] == 10


def test_flores_reader_from_local_archive(tmp_path):
    buf = tmp_path / "flores200_dataset.tar.gz"
    with tarfile.open(buf, "w:gz") as tar:
        for split in ("dev", "devtest"):
            for code in testsets.FLORES_CODES.values():
                data = "\n".join(f"{code} {split} {i}" for i in range(3)).encode()
                info = tarfile.TarInfo(f"flores200_dataset/{split}/{code}.{split}")
                info.size = len(data)
                tar.addfile(info, io.BytesIO(data))
    sets = testsets.flores_sets(str(tmp_path))
    assert set(sets) == {"flores_dev.en-de", "flores_devtest.en-de", "flores_devtest.de-en", "flores_devtest.en-zh"}
    row = sets["flores_devtest.de-en"][1]
    assert row["src"] == "deu_Latn devtest 1" and row["ref"] == "eng_Latn devtest 1"
    assert row["src_lang"] == "de" and row["tgt_lang"] == "en"


def test_debug_store_never_uses_the_hub(monkeypatch, tmp_path):
    from mtrlhf.models import CheckpointStore

    monkeypatch.setenv("MTRLHF_HUB_REPO", "someone/real-adapters")
    assert CheckpointStore.from_env(str(tmp_path / "ckpt")).hub_repo == "someone/real-adapters"
    assert CheckpointStore.from_env(str(tmp_path / "ckpt_debug")).hub_repo is None
    assert CheckpointStore.from_env(str(tmp_path / "ckpt"), "none").hub_repo is None


# --- two model families ------------------------------------------------------------

def test_model_configs_per_family():
    from mtrlhf.train.methods import build_config

    cfg, ckpt, prefix = build_config("configs/p2/grpo.yaml", "gemma3-1b")
    assert cfg["base_model"] == "google/gemma-3-1b-it" and ckpt == "ckpt/gemma3-1b" and prefix == "gemma3-1b"
    assert (cfg["batch_size"], cfg["grad_accum"], cfg["epochs"]) == (4, 16, 1)  # still 8 prompts per step
    assert cfg["batch_size"] * cfg["grad_accum"] // cfg["num_generations"] == 8
    assert cfg["data"] == "data/p2/prompts.jsonl" and cfg["out_root"] == "runs/p2/gemma3-1b"
    cfg, _, _ = build_config("configs/p2/dpo.yaml", "qwen3-1.7b")
    assert cfg["data"] == "data/p2/qwen3-1.7b/dpo_pairs.jsonl" and cfg["base_model"] == "Qwen/Qwen3-1.7B"
    cfg, ckpt, _ = build_config("configs/p2/grpo.yaml", "qwen3-1.7b", debug=True, sets=["beta=0.0"])
    assert ckpt == "ckpt/qwen3-1.7b_debug" and cfg["max_steps"] == 4 and cfg["beta"] == 0.0
    assert cfg["batch_size"] * cfg["grad_accum"] % cfg["num_generations"] == 0


def test_precision_follows_the_model():
    from mtrlhf.models import fp16_safe, pick_dtype

    assert fp16_safe("Qwen/Qwen3-1.7B") and not fp16_safe("google/gemma-3-1b-it")
    import torch

    assert pick_dtype("cpu", "google/gemma-3-1b-it") == torch.float32


QWEN3_LIKE_TEMPLATE = (
    "{%- for message in messages %}"
    "{%- if message.role == 'user' %}{{- '<|im_start|>user\\n' + message.content + '<|im_end|>\\n' }}"
    "{%- elif message.role == 'assistant' %}"
    "{{- '<|im_start|>assistant\\n<think>\\n\\n</think>\\n\\n' + message.content + '<|im_end|>\\n' }}"
    "{%- endif %}{%- endfor %}"
    "{%- if add_generation_prompt %}{{- '<|im_start|>assistant\\n' }}"
    "{%- if enable_thinking is defined and enable_thinking is false %}{{- '<think>\\n\\n</think>\\n\\n' }}{%- endif %}"
    "{%- endif %}"
)


def test_qwen3_thinking_is_off_everywhere(tmp_path):
    pytest.importorskip("transformers")
    from tests.tiny_lm import build_tiny_lm
    from mtrlhf.models import disable_thinking
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(build_tiny_lm(str(tmp_path / "m"), ["hallo welt", "hello world"]))
    tok.chat_template = QWEN3_LIKE_TEMPLATE
    msgs = [{"role": "user", "content": "Translate: hello"}]
    before = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    disable_thinking(tok)
    disable_thinking(tok)  # idempotent
    prompt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    full = tok.apply_chat_template(msgs + [{"role": "assistant", "content": "hallo"}], tokenize=False)
    assert not before.endswith("</think>\n\n") and prompt.endswith("<think>\n\n</think>\n\n")
    assert full.startswith(prompt)  # SFT/DPO train on exactly the prompt used at inference
    # even an explicit request for thinking is ignored
    assert tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True,
                                   enable_thinking=True) == prompt


def test_think_blocks_are_stripped():
    assert prompts.extract_translation("<think>\nhmm\n</think>\n\nHallo Welt.", "Hello world.")[0] == "Hallo Welt."
    assert prompts.extract_translation("<think></think>Hallo.", "Hi.")[0] == "Hallo."


def test_store_uses_a_per_model_folder_in_the_hub_repo(tmp_path, monkeypatch):
    import huggingface_hub

    from mtrlhf.models import CheckpointStore

    uploads = []
    monkeypatch.setattr(huggingface_hub, "upload_folder",
                        lambda **kw: uploads.append((kw["repo_id"], kw["path_in_repo"])))

    def fake_snapshot(repo_id, allow_patterns, local_dir):
        remote = allow_patterns[0].rstrip("/*")
        os.makedirs(os.path.join(local_dir, remote), exist_ok=True)
        open(os.path.join(local_dir, remote, "mtrlhf_meta.json"), "w").write('{"base_model": "x", "parent": null}')

    monkeypatch.setattr(huggingface_hub, "snapshot_download", fake_snapshot)
    store = CheckpointStore(str(tmp_path / "ckpt" / "gemma3-1b"), "me/adapters", "gemma3-1b")
    os.makedirs(store.path("sft"))
    store.save("sft", store.path("sft"), "google/gemma-3-1b-it", None)
    assert uploads == [("me/adapters", "gemma3-1b/sft")]
    assert store.ensure("dpo") == store.path("dpo") and store.meta("dpo")["base_model"] == "x"


def test_incompatible_torchao_is_ignored(monkeypatch):
    peft_import_utils = pytest.importorskip("peft.import_utils")
    import peft.tuners.lora.torchao as lora_torchao

    from mtrlhf.compat import ignore_incompatible_torchao

    def kaggle_check():
        raise ImportError("Found an incompatible version of torchao. Found version 0.10.0, "
                          "but only versions above 0.16.0 are supported")

    monkeypatch.setattr(peft_import_utils, "is_torchao_available", kaggle_check)
    monkeypatch.setattr(lora_torchao, "is_torchao_available", kaggle_check)
    assert ignore_incompatible_torchao() is True
    assert peft_import_utils.is_torchao_available() is False and lora_torchao.is_torchao_available() is False
    assert ignore_incompatible_torchao() is False  # nothing left to patch


def test_other_import_errors_are_not_hidden(monkeypatch):
    peft_import_utils = pytest.importorskip("peft.import_utils")
    from mtrlhf.compat import ignore_incompatible_torchao

    def broken():
        raise ImportError("something else is broken")

    monkeypatch.setattr(peft_import_utils, "is_torchao_available", broken)
    with pytest.raises(ImportError, match="something else"):
        ignore_incompatible_torchao()


# --- sentence-level NMT on paragraphs (OPUS-MT crashed on WMT24) ------------------------

def test_split_sentences():
    from mtrlhf.models import split_sentences

    para = "The first sentence is here. A second one follows! Is this the third? Yes.\nNew line."
    assert split_sentences(para) == ["The first sentence is here.", "A second one follows!",
                                     "Is this the third?", "Yes.", "New line."]
    assert split_sentences("Dr. Smith arrived at 5 p.m. today.") == ["Dr. Smith arrived at 5 p.m. today."]


def test_seq2seq_paragraphs_are_translated_sentence_by_sentence():
    torch = pytest.importorskip("torch")
    from mtrlhf.models import Translator

    class Tok:
        pad_token_id = 0
        padding_side = "right"

        def __init__(self):
            self.calls = []

        def __call__(self, texts, **kw):
            self.calls.append((list(texts), kw))
            ids = torch.ones((len(texts), 3), dtype=torch.long)

            class Enc(dict):
                def to(self, device):
                    return self
            return Enc(input_ids=ids, attention_mask=ids)

        def batch_decode(self, ids, skip_special_tokens=True):
            return [f"T{int(row[0])}" for row in ids]

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.w = torch.nn.Parameter(torch.zeros(1))
            self.config = type("C", (), {"max_position_embeddings": 512})()
            self.seen = []

        def generate(self, input_ids, attention_mask, **gen):
            self.seen.append(gen)
            return torch.arange(input_ids.shape[0]).unsqueeze(1)  # one "token" = row index

    tok, model = Tok(), Model()
    tr = Translator(model, tok, kind="seq2seq", batch_size=8, max_new_tokens=768)
    out = tr.generate(["One. Two. Three.", "Single sentence"])
    assert len(out) == 2 and len(out[0][0].split()) == 3 and len(out[1][0].split()) == 1
    texts, kw = tok.calls[0]
    assert sorted(texts) == sorted(["One.", "Two.", "Three.", "Single sentence"])
    assert kw["truncation"] is True and kw["max_length"] == 512
    assert model.seen[0]["max_new_tokens"] == 511  # capped below the 512 positions
