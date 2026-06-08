"""
Tests unitaires — Service Preprocessing
Lance avec : pytest tests/test_preprocessing.py -v
"""

import pytest
from pathlib import Path
from unittest.mock import patch
from app.services.preprocessing import read_sequence_from_file


# ============================================================
# CAS NOMINAUX
# ============================================================

def test_read_sequence_retourne_liste():
    """La fonction retourne bien une liste."""
    with patch("app.services.preprocessing.LAST_PHRASE_TXT") as mock_path:
        mock_path.exists.return_value = True
        mock_path.read_text.return_value = "bonjour je_suis content"
        result = read_sequence_from_file()
    assert isinstance(result, list)


def test_read_sequence_majuscules():
    """Les glosses sont bien converties en majuscules."""
    with patch("app.services.preprocessing.LAST_PHRASE_TXT") as mock_path:
        mock_path.exists.return_value = True
        mock_path.read_text.return_value = "bonjour je_suis content"
        result = read_sequence_from_file()
    assert result == ["BONJOUR", "JE_SUIS", "CONTENT"]


def test_read_sequence_fichier_inexistant():
    """Retourne une liste vide si le fichier n'existe pas."""
    with patch("app.services.preprocessing.LAST_PHRASE_TXT") as mock_path:
        mock_path.exists.return_value = False
        result = read_sequence_from_file()
    assert result == []


def test_read_sequence_fichier_vide():
    """Retourne une liste vide si le fichier est vide."""
    with patch("app.services.preprocessing.LAST_PHRASE_TXT") as mock_path:
        mock_path.exists.return_value = True
        mock_path.read_text.return_value = ""
        result = read_sequence_from_file()
    assert result == []


def test_read_sequence_un_mot():
    """Fonctionne avec un seul mot."""
    with patch("app.services.preprocessing.LAST_PHRASE_TXT") as mock_path:
        mock_path.exists.return_value = True
        mock_path.read_text.return_value = "bonjour"
        result = read_sequence_from_file()
    assert result == ["BONJOUR"]


# ============================================================
# CAS LIMITES
# ============================================================

@pytest.mark.parametrize("content,expected", [
    ("bonjour", ["BONJOUR"]),
    ("bonjour merci", ["BONJOUR", "MERCI"]),
    ("je_suis content presenter projet", ["JE_SUIS", "CONTENT", "PRESENTER", "PROJET"]),
    ("BONJOUR JE_SUIS", ["BONJOUR", "JE_SUIS"]),  # déjà en majuscules
])
def test_read_sequence_cas_variés(content, expected):
    """Fonctionne pour différents contenus de fichier."""
    with patch("app.services.preprocessing.LAST_PHRASE_TXT") as mock_path:
        mock_path.exists.return_value = True
        mock_path.read_text.return_value = content
        result = read_sequence_from_file()
    assert result == expected
