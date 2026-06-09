"""
Service NLP — BARThez
Convertit une liste de glosses LSF en phrase française naturelle.
"""

import re
import logging
from pathlib import Path

import yaml
import torch
from transformers import pipeline, CamembertTokenizer, AutoModelForSeq2SeqLM

# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent.parent
CONFIG_PATH = BASE_DIR / "config" / "config.yaml"

with open(CONFIG_PATH, encoding="utf-8") as f:
    config = yaml.safe_load(f)

MODEL_DIR = BASE_DIR / config["paths"]["barthez_dir"]
MAX_NEW_TOKENS = config["nlp"]["max_new_tokens"]
NUM_BEAMS = config["nlp"]["num_beams"]
EARLY_STOPPING = config["nlp"]["early_stopping"]
NO_REPEAT_NGRAM_SIZE = config["nlp"]["no_repeat_ngram_size"]
REPETITION_PENALTY = config["nlp"]["repetition_penalty"]

# ============================================================
# LOGGING
# ============================================================

logger = logging.getLogger(__name__)

# ============================================================
# CHARGEMENT DU MODÈLE — une seule fois au démarrage
# ============================================================

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
logger.info(f"NLP — Device : {device}")
logger.info(f"NLP — Chargement du modèle : {MODEL_DIR}")

tokenizer = CamembertTokenizer.from_pretrained(MODEL_DIR)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_DIR).to(device)

generator = pipeline(
    "text2text-generation",
    model=model,
    tokenizer=tokenizer,
    device=device,
)

logger.info("NLP — Modèle BARThez chargé ✅")


# ============================================================
# FONCTION PRINCIPALE
# ============================================================

def generate_phrase(glosses: list[str]) -> str:
    """
    Convertit une liste de glosses LSF en phrase française naturelle.

    Args:
        glosses : liste de glosses prédites par le LSTM
                  ex: ["BONJOUR", "JE_SUIS", "CONTENT", "PRESENTER", "PROJET"]

    Returns:
        str : phrase française naturelle
              ex: "Bonjour, je suis content de vous présenter le projet."

    Raises:
        ValueError : si la liste de glosses est vide
        RuntimeError : si la génération échoue
    """
    if not glosses:
        raise ValueError("La liste de glosses est vide.")

    # Conversion liste → string
    glosses_str = " ".join(glosses)
    logger.info(f"NLP — Génération pour : {glosses_str}")

    try:
        result = generator(
            glosses_str,
            max_new_tokens=MAX_NEW_TOKENS,
            num_beams=NUM_BEAMS,
            early_stopping=EARLY_STOPPING,
            no_repeat_ngram_size=NO_REPEAT_NGRAM_SIZE,
            repetition_penalty=REPETITION_PENALTY,
        )
        phrase = result[0]["generated_text"]

# Couper au premier point/!/?
        match = re.search(r'[.!?]', phrase)
        if match:
            phrase = phrase[:match.end()].strip()
        # Sécurité : virer les tokens parasites résiduels
        phrase = re.sub(r'[\s]*[A-Z]{3,}.*$', '', phrase).strip()
        if not phrase.endswith(('.', '!', '?')):
            phrase = phrase + '.'

        logger.info(f"NLP — Phrase générée : {phrase}")
        return phrase

    except Exception as e:
        logger.error(f"NLP — Erreur lors de la génération : {e}")
        raise RuntimeError(f"Erreur lors de la génération de la phrase : {e}")
