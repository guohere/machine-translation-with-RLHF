"""Shared fixtures: a tiny WMT22 en-de dataset in mt-metrics-eval's on-disk format."""

from __future__ import annotations

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

SRC = [
    "The company reported a 12% rise in profits in the third quarter, analysts said on Monday.",
    "Angela Merkel visited Paris on Tuesday to meet President Macron, according to officials.",
    "The museum will not open before 2025 because of the renovation work in the old town.",
    "There is no evidence that the new policy has reduced traffic in Berlin this year.",
    "Researchers at Oxford found that 40 percent of the samples were contaminated with bacteria.",
    "The train was delayed by two hours, and many passengers missed their connections.",
    "Google announced a new version of its translation app for Android users last week.",
    "The weather will be sunny tomorrow, but it may rain again on the weekend.",
]
REF_A = [
    "Das Unternehmen meldete im dritten Quartal einen Gewinnanstieg von 12 %, sagten Analysten am Montag.",
    "Angela Merkel besuchte am Dienstag Paris, um Präsident Macron zu treffen, so Beamte.",
    "Das Museum wird wegen der Renovierungsarbeiten in der Altstadt nicht vor 2025 öffnen.",
    "Es gibt keine Belege dafür, dass die neue Politik den Verkehr in Berlin dieses Jahr verringert hat.",
    "Forscher in Oxford stellten fest, dass 40 Prozent der Proben mit Bakterien verunreinigt waren.",
    "Der Zug hatte zwei Stunden Verspätung, und viele Fahrgäste verpassten ihre Anschlüsse.",
    "Google hat letzte Woche eine neue Version seiner Übersetzungs-App für Android-Nutzer angekündigt.",
    "Morgen wird es sonnig, aber am Wochenende könnte es wieder regnen.",
]
REF_B = [
    "Das Unternehmen verzeichnete im dritten Quartal ein Gewinnplus von 12 %, wie Analysten am Montag mitteilten.",
    "Angela Merkel reiste am Dienstag nach Paris, um sich mit Präsident Macron zu treffen, wie Beamte mitteilten.",
    "Wegen der Renovierung in der Altstadt wird das Museum nicht vor 2025 eröffnet.",
    "Es gibt keinen Beweis, dass die neue Politik den Verkehr in Berlin in diesem Jahr reduziert hat.",
    "Forschende in Oxford fanden heraus, dass 40 Prozent der Proben mit Bakterien kontaminiert waren.",
    "Der Zug verspätete sich um zwei Stunden, und viele Reisende verpassten ihre Anschlüsse.",
    "Google kündigte letzte Woche eine neue Version seiner Übersetzungs-App für Android-Nutzer an.",
    "Morgen scheint die Sonne, aber am Wochenende kann es wieder regnen.",
]


def _degrade(text: str, k: int) -> str:
    words = text.split()
    return " ".join(words[: max(3, len(words) - k)])


SYSTEMS = {
    "sysA": REF_A,
    "sysB": [_degrade(t, 3) for t in REF_A],
    "sysC": [_degrade(t, 7) for t in REF_A],
    "M2M100_1.2B-B4": SRC,  # outlier system in WMT22 en-de meta info
    "refB": REF_B,  # human translation, also scored as a "system"
}
# MQM error points (stored negated, higher is better); None = not annotated.
MQM = {
    "sysA": [-0.0, -0.0, -1.0, -0.0, -0.1, -0.0, -0.0, -1.0],
    "sysB": [-5.0, -1.0, -5.0, -5.0, None, -2.0, -5.0, -1.0],
    "sysC": [-10.0, -6.0, -15.0, -10.0, -11.0, -7.0, -10.0, -5.0],
    "M2M100_1.2B-B4": [-25.0] * 8,
    "refB": [-0.0, -1.0, -0.0, -0.0, -0.0, -0.1, -0.0, -0.0],
}


def _write(path: str, lines) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")


def build_fake_mtme(root: str) -> str:
    d = os.path.join(root, "wmt22")
    lp = "en-de"
    docs = ["news doc1"] * 4 + ["social doc2"] * 4
    _write(f"{d}/documents/{lp}.docs", docs)
    _write(f"{d}/sources/{lp}.txt", SRC)
    _write(f"{d}/references/{lp}.refA.txt", REF_A)
    _write(f"{d}/references/{lp}.refB.txt", REF_B)
    for name, lines in SYSTEMS.items():
        _write(f"{d}/system-outputs/{lp}/{name}.txt", lines)
    seg = [f"{s} {'None' if v is None else v}" for s, vals in MQM.items() for v in vals]
    _write(f"{d}/human-scores/{lp}.mqm.seg.score", seg)
    sysl, doml = [], []
    for s, vals in MQM.items():
        ok = [v for v in vals if v is not None]
        sysl.append(f"{s} {sum(ok) / len(ok)}")
        for dom, part in (("news", vals[:4]), ("social", vals[4:])):
            ok = [v for v in part if v is not None]
            doml.append(f"{dom} {s} {sum(ok) / len(ok)}")
    _write(f"{d}/human-scores/{lp}.mqm.sys.score", sysl)
    _write(f"{d}/human-scores/{lp}.mqm.domain.score", doml)
    # Stored metric scores: a fake "COMETKiwi" that tracks MQM, and chrF vs refA.
    kiwi = [f"{s} {0.8 + (0.0 if v is None else v) / 50}" for s, vals in MQM.items() for v in vals]
    _write(f"{d}/metric-scores/{lp}/COMETKiwi-src.seg.score", kiwi)
    from sacrebleu.metrics import CHRF

    chrf = CHRF()
    chrf_lines = [f"{s} {chrf.sentence_score(h, [r]).score}" for s, hyps in SYSTEMS.items()
                  for h, r in zip(hyps, REF_A)]
    _write(f"{d}/metric-scores/{lp}/chrF-refA.seg.score", chrf_lines)
    return root


@pytest.fixture()
def fake_mtme(tmp_path):
    return build_fake_mtme(str(tmp_path / "mtme"))
