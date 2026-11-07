import random

from mtrlhf.meta_eval import probes

SRC = "Angela Merkel visited Paris on Tuesday to meet President Macron, according to 12 officials."
TGT = "Angela Merkel besuchte am Dienstag Paris, um Präsident Macron zu treffen, so 12 Beamte."


def rng():
    return random.Random(0)


def test_omission_drops_a_clause():
    out = probes.omission(SRC, TGT, rng(), {})
    assert out and len(out.split()) <= len(TGT.split()) - 3
    assert out.startswith("Angela Merkel")


def test_omission_drops_a_sentence():
    tgt = "Der Zug war spät. Viele Fahrgäste verpassten ihre Anschlüsse am Bahnhof. Es regnete."
    out = probes.omission("x", tgt, rng(), {})
    assert out and out.count(".") == 2


def test_number_changes_shared_number():
    out = probes.number(SRC, TGT, rng(), {})
    assert out == TGT.replace("12", "21")
    assert probes.change_number("2019") == "2091"
    assert probes.change_number("7") == "8" and probes.change_number("9") == "8"
    assert probes.number("no digits here", TGT, rng(), {}) is None


def test_entities_and_swap():
    ents = probes.entities(SRC, TGT)
    assert "Merkel" in ents and "Paris" in ents and "Macron" in ents
    assert "Angela" not in ents  # sentence-initial words are skipped
    out = probes.named_entity(SRC, TGT, rng(), {"entity_pool": ["Berlin", "Scholz"]})
    assert out != TGT and ("Berlin" in out or "Scholz" in out)


def test_negation():
    assert probes.negation("", "Das Museum wird nicht vor 2025 öffnen.", rng(), {}) == \
        "Das Museum wird vor 2025 öffnen."
    assert probes.negation("", "Es gibt keine Belege dafür.", rng(), {}) == "Es gibt eine Belege dafür."
    assert probes.negation("", "Kein Zug fuhr.", rng(), {}) == "Ein Zug fuhr."
    assert probes.negation("", "Er sagte nichts.", rng(), {}) is None


def test_other_categories():
    assert probes.untranslated(SRC, TGT, rng(), {}) == SRC
    assert probes.wrong_language(SRC, TGT, rng(), {"other_language": {SRC: ("en-cs", "Ahoj")}}) == "Ahoj"
    assert probes.wrong_language(SRC, TGT, rng(), {}) is None
    wo = probes.word_order(SRC, TGT, rng(), {})
    assert wo != TGT and sorted(wo.split()) == sorted(TGT.split())
    rep = probes.repetition(SRC, TGT, rng(), {})
    assert len(rep.split()) > len(TGT.split())
    add = probes.addition(SRC, TGT, rng(), {"sentence_pool": ["Das Wetter war schön und warm."]})
    assert add.endswith("Das Wetter war schön und warm.")


def test_build_probe_rows_is_deterministic():
    items = [{"src": SRC, "ref": TGT, "alt_ref": "Merkel reiste am Dienstag nach Paris."},
             {"src": "The museum will not open before 2025 because of the renovation work.",
              "ref": "Das Museum wird wegen der Renovierung nicht vor 2025 öffnen.", "alt_ref": "Anders."}]
    a = probes.build_probe_rows(items, n_items=10, seed=1)
    b = probes.build_probe_rows(items, n_items=10, seed=1)
    assert a == b
    cats = {r["category"] for r in a}
    assert {"original", "untranslated", "number", "negation"} <= cats
    assert all(r["ref"] for r in a)  # reference-based metrics get the alternative reference
