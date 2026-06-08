"""
Tests unitaires — Service TTS (gTTS)
Lance avec : pytest tests/test_tts.py -v
"""

import pytest
from app.services.tts import text_to_speech


# ============================================================
# CAS NOMINAUX
# ============================================================

def test_tts_retourne_bytes():
    """La fonction retourne bien des bytes."""
    result = text_to_speech("Bonjour.")
    assert isinstance(result, bytes)


def test_tts_bytes_non_vides():
    """Les bytes retournés ne sont pas vides."""
    result = text_to_speech("Bonjour.")
    assert len(result) > 0


def test_tts_signature_mp3():
    """Les bytes commencent par la signature MP3 (ID3 ou 0xFF 0xFB)."""
    result = text_to_speech("Bonjour.")
    # MP3 commence soit par ID3 soit par 0xFF
    assert result[:2] in [b"ID", b"\xff\xfb", b"\xff\xf3", b"\xff\xf2"]


def test_tts_phrase_courte():
    """Fonctionne avec une phrase courte."""
    result = text_to_speech("Merci.")
    assert isinstance(result, bytes)
    assert len(result) > 0


def test_tts_phrase_longue():
    """Fonctionne avec une phrase longue."""
    phrase = "Bonjour, je suis content de vous présenter ce projet de traduction de la langue des signes française."
    result = text_to_speech(phrase)
    assert isinstance(result, bytes)
    assert len(result) > 0


def test_tts_slow_mode():
    """Fonctionne en mode lent."""
    result = text_to_speech("Bonjour.", slow=True)
    assert isinstance(result, bytes)
    assert len(result) > 0


# ============================================================
# CAS LIMITES
# ============================================================

@pytest.mark.parametrize("phrase", [
    "Bonjour.",
    "Merci.",
    "Je suis content.",
    "Aujourd'hui, je veux vous présenter le projet.",
    "Bonjour, je suis malentendant et je veux communiquer.",
])
def test_tts_phrases_variees(phrase):
    """Fonctionne pour différentes phrases françaises."""
    result = text_to_speech(phrase)
    assert isinstance(result, bytes)
    assert len(result) > 0
