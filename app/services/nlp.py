import re
import os
import logging
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from pathlib import Path

logger = logging.getLogger(__name__)

# MODIF 006 — Resize embeddings + generation directe (sans pipeline wrapper)
# Réf : modifs/006_resize_embeddings.txt

BASE_DIR = Path(os.getenv("MODEL_BASE_DIR", "models"))
MODEL_DIR = BASE_DIR / "barthez_sensi_final"

device = torch.device(
    "mps" if torch.backends.mps.is_available()
    else "cpu"
)

tokenizer = AutoTokenizer.from_pretrained(str(MODEL_DIR), local_files_only=True)
model = AutoModelForSeq2SeqLM.from_pretrained(str(MODEL_DIR), local_files_only=True)

# Le tokenizer a des tokens jusqu'à l'ID 50003 mais le modèle a vocab_size=50002.
# On resize pour couvrir tous les IDs connus du tokenizer.
vocab_size = max(tokenizer.get_vocab().values()) + 1
model.resize_token_embeddings(vocab_size)
logger.info(f"Embeddings resized to {vocab_size} (tokenizer has {len(tokenizer)} tokens)")

model = model.to(device)
model.eval()


def generate_phrase(glosses: list[str]) -> str:
    glosses_str = " ".join(glosses)

    inputs = tokenizer(
        glosses_str,
        max_length=64,
        truncation=True,
        return_tensors="pt",
    ).to(device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=40,
            num_beams=4,
            early_stopping=True,
            no_repeat_ngram_size=3,
            repetition_penalty=2.0,
        )

    phrase = tokenizer.decode(outputs[0], skip_special_tokens=True)

    match = re.search(r'[.!?]', phrase)
    if match:
        phrase = phrase[:match.end()]

    return phrase.strip()
