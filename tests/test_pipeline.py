"""
Tests unitaires — Pipeline Sensi (NLP + TTS)
Lance avec : pytest tests/test_pipeline.py -v
"""

import pytest
from app.services.pipeline import pipeline_sensi


# ============================================================
# CAS NOMINAUX
# ============================================================

def test_pipeline_retourne_tuple():
    """Le pipeline retourne bien un tuple (str, bytes)."""
    result = pipeline_sensi(["BONJOUR", "JE_SUIS", "CONTENT"])
    assert isinstance(result, tuple)
    assert len(result) == 2


def test_pipeline_phrase_est_string():
    """La phrase retournée est une string."""
    phrase, _ = pipeline_sensi(["BONJOUR", "MERCI"])
    assert isinstance(phrase, str)


def test_pipeline_audio_est_bytes():
    """L'audio retourné est des bytes."""
    _, audio = pipeline_sensi(["BONJOUR", "MERCI"])
    assert isinstance(audio, bytes)


def test_pipeline_phrase_non_vide():
    """La phrase retournée n'est pas vide."""
    phrase, _ = pipeline_sensi(["BONJOUR", "JE_SUIS", "CONTENT"])
    assert len(phrase) > 0


def test_pipeline_audio_non_vide():
    """L'audio retourné n'est pas vide."""
    _, audio = pipeline_sensi(["BONJOUR", "JE_SUIS", "CONTENT"])
    assert len(audio) > 0


def test_pipeline_phrase_termine_par_ponctuation():
    """La phrase se termine par un signe de ponctuation."""
    phrase, _ = pipeline_sensi(["BONJOUR", "JE_SUIS", "CONTENT", "PRESENTER", "PROJET"])
    assert phrase[-1] in [".", "!", "?"]


# ============================================================
# CAS D'ERREUR
# ============================================================

def test_pipeline_glosses_vides():
    """Lève une erreur si la liste est vide."""
    with pytest.raises(Exception):
        pipeline_sensi([])


# ============================================================
# CAS LIMITES
# ============================================================

@pytest.mark.parametrize("glosses", [
    ["BONJOUR"],
    ["BONJOUR", "MERCI"],
    ["AUJOURD_HUI", "JE_VEUX", "PRESENTER", "PROJET"],
    ["JE_SUIS", "SOURD", "JE_VEUX", "COMMUNIQUER", "LANGUE_DES_SIGNES"],
])
def test_pipeline_cas_variés(glosses):
    """Pipeline fonctionne pour différentes séquences de glosses."""
    phrase, audio = pipeline_sensi(glosses)
    assert isinstance(phrase, str) and len(phrase) > 0
    assert isinstance(audio, bytes) and len(audio) > 0
