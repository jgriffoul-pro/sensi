from app.services.nlp import generate_phrase
from app.services.tts import text_to_speech


def pipeline_sensi(glosses: list[str]) -> tuple[str, bytes]:
    """
    Pipeline complet Sensi.

    Args:
        glosses : liste de glosses prédites par le LSTM
                  ex: ["BONJOUR", "JE_SUIS", "CONTENT"]

    Returns:
        tuple : (phrase française, audio MP3 en bytes)
    """
    # Étape 1 — NLP : glosses → phrase française
    phrase = generate_phrase(glosses)

    # Étape 2 — TTS : phrase → audio MP3
    audio = text_to_speech(phrase)

    return phrase, audio
