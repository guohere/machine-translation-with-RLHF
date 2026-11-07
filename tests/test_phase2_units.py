import gzip
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
