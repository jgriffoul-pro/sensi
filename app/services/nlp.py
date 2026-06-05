import re
import torch
from transformers import pipeline, GenerationConfig
from transformers import CamembertTokenizer, AutoModelForSeq2SeqLM
from pathlib import Path

MODEL_DIR = Path("models/barthez_sensi_final")

# Chargement du device
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# Chargement du modèle fine-tuné
tokenizer = CamembertTokenizer.from_pretrained(MODEL_DIR)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_DIR)
model = model.to(device)

# Pipeline de génération
generator = pipeline(
    "text2text-generation",
    model=model,
    tokenizer=tokenizer,
    device=device,
)


def generate_phrase(glosses: list[str]) -> str:
    """
    Convertit une liste de glosses LSF en phrase française naturelle.

    Args:
        glosses : liste de glosses prédites par le LSTM
                  ex: ["BONJOUR", "JE_SUIS", "CONTENT", "PRESENTER", "PROJET"]

    Returns:
        str : phrase française naturelle
    """
    # Conversion liste → string
    glosses_str = " ".join(glosses)

    # Génération de la phrase
    result = generator(
        glosses_str,
        max_new_tokens=40,
        num_beams=4,
        early_stopping=True,
        no_repeat_ngram_size=3,
        repetition_penalty=2.0,
    )
    phrase = result[0]["generated_text"]

    # Coupe à la première fin de phrase
    match = re.search(r'[.!?]', phrase)
    if match:
        phrase = phrase[:match.end()]

    return phrase.strip()
