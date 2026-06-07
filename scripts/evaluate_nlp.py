"""
Script d'évaluation du modèle BARThez fine-tuné.
Teste la qualité des traductions glosses → phrase française.

Usage :
    python scripts/evaluate_nlp.py
    python scripts/evaluate_nlp.py --examples 20
"""

import argparse
import json
import logging
from pathlib import Path

import yaml
import torch
from transformers import CamembertTokenizer, AutoModelForSeq2SeqLM, pipeline

# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent
CONFIG_PATH = BASE_DIR / "config" / "config.yaml"

with open(CONFIG_PATH, encoding="utf-8") as f:
    config = yaml.safe_load(f)

# Chemins
MODEL_DIR = BASE_DIR / config["paths"]["barthez_dir"]
DATASET_JSON = BASE_DIR / config["paths"]["nlp_dataset_json"]

# Paramètres de génération
MAX_NEW_TOKENS = config["nlp"]["max_new_tokens"]
NUM_BEAMS = config["nlp"]["num_beams"]
NO_REPEAT_NGRAM_SIZE = config["nlp"]["no_repeat_ngram_size"]
REPETITION_PENALTY = config["nlp"]["repetition_penalty"]

# ============================================================
# LOGGING
# ============================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s — %(levelname)s — %(message)s",
)
logger = logging.getLogger(__name__)


# ============================================================
# CHARGEMENT DU MODÈLE
# ============================================================

def load_model():
    """Charge le modèle fine-tuné et le tokenizer."""
    logger.info(f"Chargement du modèle : {MODEL_DIR}")

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    logger.info(f"Device : {device}")

    tokenizer = CamembertTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_DIR).to(device)

    generator = pipeline(
        "text2text-generation",
        model=model,
        tokenizer=tokenizer,
        device=device,
    )
    return generator


# ============================================================
# ÉVALUATION
# ============================================================

def evaluate(n_examples: int = 10):
    """
    Évalue le modèle sur le dataset d'entraînement.
    Affiche les prédictions vs les phrases attendues.

    Args:
        n_examples : nombre d'exemples à tester
    """
    # Chargement du dataset
    logger.info(f"Chargement du dataset : {DATASET_JSON}")
    with open(DATASET_JSON, encoding="utf-8") as f:
        dataset = json.load(f)

    # Chargement du modèle
    generator = load_model()

    # Sélection des exemples
    examples = dataset[:n_examples]

    # Évaluation
    logger.info(f"\n{'='*60}")
    logger.info(f"ÉVALUATION — {n_examples} exemples")
    logger.info(f"{'='*60}\n")

    correct = 0
    results = []

    for i, example in enumerate(examples):
        source = example["source"]
        expected = example["target"]

        # Prédiction
        result = generator(
            source,
            max_new_tokens=MAX_NEW_TOKENS,
            num_beams=NUM_BEAMS,
            no_repeat_ngram_size=NO_REPEAT_NGRAM_SIZE,
            repetition_penalty=REPETITION_PENALTY,
        )
        predicted = result[0]["generated_text"].strip()

        # Comparaison exacte
        is_correct = predicted.lower() == expected.lower()
        if is_correct:
            correct += 1

        results.append({
            "source": source,
            "expected": expected,
            "predicted": predicted,
            "correct": is_correct,
        })

        # Affichage
        status = "✅" if is_correct else "❌"
        print(f"\n{status} Exemple {i+1}/{n_examples}")
        print(f"   Glosses   : {source}")
        print(f"   Attendu   : {expected}")
        print(f"   Prédit    : {predicted}")

    # Résumé
    accuracy = correct / n_examples
    print(f"\n{'='*60}")
    print(f"RÉSUMÉ")
    print(f"{'='*60}")
    print(f"Exemples testés  : {n_examples}")
    print(f"Corrects exacts  : {correct}/{n_examples}")
    print(f"Accuracy exacte  : {accuracy:.1%}")
    print(f"{'='*60}\n")

    logger.info("Évaluation terminée ✅")
    return results


# ============================================================
# TESTS MANUELS
# ============================================================

def test_manual(glosses_list: list[list[str]]):
    """
    Teste le modèle sur des exemples manuels.

    Args:
        glosses_list : liste de listes de glosses à tester
    """
    generator = load_model()

    print(f"\n{'='*60}")
    print("TESTS MANUELS")
    print(f"{'='*60}\n")

    for glosses in glosses_list:
        source = " ".join(glosses)
        result = generator(
            source,
            max_new_tokens=MAX_NEW_TOKENS,
            num_beams=NUM_BEAMS,
            no_repeat_ngram_size=NO_REPEAT_NGRAM_SIZE,
            repetition_penalty=REPETITION_PENALTY,
        )
        predicted = result[0]["generated_text"].strip()
        print(f"Glosses : {source}")
        print(f"Phrase  : {predicted}\n")


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Évaluation BARThez — Sensi NLP")
    parser.add_argument(
        "--examples",
        type=int,
        default=10,
        help="Nombre d'exemples à évaluer"
    )
    parser.add_argument(
        "--manual",
        action="store_true",
        help="Lance les tests manuels prédéfinis"
    )
    args = parser.parse_args()

    if args.manual:
        # Tests manuels prédéfinis — à adapter selon les besoins
        test_cases = [
            ["BONJOUR", "JE_SUIS", "CONTENT"],
            ["MERCI", "AMI"],
            ["AUJOURD_HUI", "JE_VEUX", "PRESENTER", "PROJET"],
            ["BONJOUR", "JE_SUIS", "SOURD", "JE_VEUX", "COMMUNIQUER"],
            ["LANGUE_DES_SIGNES", "AIDER", "ENTENDANTS"],
        ]
        test_manual(test_cases)
    else:
        evaluate(n_examples=args.examples)
        