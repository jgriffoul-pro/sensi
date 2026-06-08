"""
Tests unitaires — Service NLP (BARThez)
Lance avec : pytest tests/test_nlp.py -v
"""

import pytest
from app.services.nlp import generate_phrase


# ============================================================
# CAS NOMINAUX
# ============================================================

def test_generate_phrase_retourne_string():
    """La fonction retourne bien une string."""
    glosses = ["BONJOUR", "JE_SUIS", "CONTENT"]
    result = generate_phrase(glosses)
    assert isinstance(result, str)


def test_generate_phrase_non_vide():
    """La phrase générée n'est pas vide."""
    glosses = ["BONJOUR", "MERCI"]
    result = generate_phrase(glosses)
    assert len(result) > 0


def test_generate_phrase_termine_par_ponctuation():
    """La phrase générée se termine par un signe de ponctuation."""
    glosses = ["BONJOUR", "JE_SUIS", "CONTENT", "PRESENTER", "PROJET"]
    result = generate_phrase(glosses)
    assert result[-1] in [".", "!", "?"]


def test_generate_phrase_une_seule_gloss():
    """Fonctionne avec une seule gloss."""
    result = generate_phrase(["BONJOUR"])
    assert isinstance(result, str)
    assert len(result) > 0


def test_generate_phrase_sequence_longue():
    """Fonctionne avec une séquence longue."""
    glosses = [
        "BONJOUR", "JE_SUIS", "CONTENT",
        "PRESENTER", "PROJET", "LANGUE_DES_SIGNES",
        "AIDER", "ENTENDANTS"
    ]
    result = generate_phrase(glosses)
    assert isinstance(result, str)
    assert len(result) > 0


# ============================================================
# CAS D'ERREUR
# ============================================================

def test_generate_phrase_glosses_vides():
    """Lève ValueError si la liste est vide."""
    with pytest.raises(ValueError):
        generate_phrase([])


# ============================================================
# CAS LIMITES
# ============================================================

@pytest.mark.parametrize("glosses", [
    ["BONJOUR"],
    ["MERCI"],
    ["BONJOUR", "MERCI"],
    ["AUJOURD_HUI", "JE_VEUX", "PRESENTER", "PROJET"],
    ["JE_SUIS", "SOURD", "JE_VEUX", "COMMUNIQUER"],
])
def test_generate_phrase_vocabulaire_connu(glosses):
    """Fonctionne pour toutes les glosses du vocabulaire."""
    result = generate_phrase(glosses)
    assert isinstance(result, str)
    assert len(result) > 0
