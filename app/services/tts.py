import io
from gtts import gTTS


def text_to_speech(phrase: str, lang: str = "fr", slow: bool = False) -> bytes:
    """
    Convertit une phrase française en audio MP3.

    Args:
        phrase : la phrase à lire
        lang   : langue (fr par défaut)
        slow   : débit lent si True (utile pour la pédagogie)

    Returns:
        bytes : fichier MP3 en mémoire, prêt à être retourné par l'API
    """
    tts = gTTS(text=phrase, lang=lang, slow=slow)

    # Écriture en mémoire — pas de fichier temporaire sur le disque
    audio_buffer = io.BytesIO()
    tts.write_to_fp(audio_buffer)

    # Retour au début du buffer avant lecture
    audio_buffer.seek(0)

    return audio_buffer.read()
