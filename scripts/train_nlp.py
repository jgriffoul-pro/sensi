"""
Script d'entraînement BARThez pour la traduction glosses LSF → phrase française.

Usage :
    python scripts/train_nlp.py
    python scripts/train_nlp.py --epochs 30 --batch-size 4
"""

import argparse
import json
import logging
from pathlib import Path

import yaml
import torch
from datasets import Dataset
from transformers import (
    CamembertTokenizer,
    AutoModelForSeq2SeqLM,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    DataCollatorForSeq2Seq,
)

# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent
CONFIG_PATH = BASE_DIR / "config" / "config.yaml"

with open(CONFIG_PATH, encoding="utf-8") as f:
    config = yaml.safe_load(f)

# Chemins
DATASET_JSON = BASE_DIR / config["paths"]["nlp_dataset_json"]
MODEL_OUTPUT_DIR = BASE_DIR / config["paths"]["barthez_dir"]

# Paramètres d'entraînement
BASE_MODEL = config["nlp"]["base_model"]
EPOCHS = config["training_nlp"]["epochs"]
BATCH_SIZE = config["training_nlp"]["batch_size"]
LEARNING_RATE = config["training_nlp"]["learning_rate"]
WARMUP_STEPS = config["training_nlp"]["warmup_steps"]
WEIGHT_DECAY = config["training_nlp"]["weight_decay"]
EVAL_STEPS = config["training_nlp"]["eval_steps"]
SAVE_STEPS = config["training_nlp"]["save_steps"]
LOGGING_STEPS = config["training_nlp"]["logging_steps"]
MAX_INPUT_LENGTH = config["training_nlp"]["max_input_length"]
MAX_TARGET_LENGTH = config["training_nlp"]["max_target_length"]

# ============================================================
# LOGGING
# ============================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s — %(levelname)s — %(message)s",
)
logger = logging.getLogger(__name__)

# ============================================================
# DEVICE
# ============================================================

def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


# ============================================================
# CHARGEMENT DES DONNÉES
# ============================================================

def load_dataset_from_json(path: Path) -> Dataset:
    """Charge le dataset depuis le fichier JSON."""
    logger.info(f"Chargement du dataset : {path}")
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    dataset = Dataset.from_list(data)
    logger.info(f"Dataset chargé : {len(dataset)} exemples")
    return dataset


def tokenize_dataset(dataset: Dataset, tokenizer: CamembertTokenizer) -> Dataset:
    """Tokenize le dataset pour l'entraînement."""

    def tokenize(batch):
        inputs = tokenizer(
            batch["source"],
            max_length=MAX_INPUT_LENGTH,
            truncation=True,
            padding="max_length",
        )
        targets = tokenizer(
            batch["target"],
            max_length=MAX_TARGET_LENGTH,
            truncation=True,
            padding="max_length",
        )
        inputs["labels"] = targets["input_ids"]
        return inputs

    logger.info("Tokenisation du dataset...")
    tokenized = dataset.map(tokenize, batched=True)
    return tokenized


# ============================================================
# ENTRAÎNEMENT
# ============================================================

def train(epochs: int = EPOCHS, batch_size: int = BATCH_SIZE):
    """Pipeline complet d'entraînement BARThez."""

    device = get_device()
    logger.info(f"Device : {device}")

    # Chargement tokenizer et modèle
    logger.info(f"Chargement du modèle de base : {BASE_MODEL}")
    tokenizer = CamembertTokenizer.from_pretrained(BASE_MODEL)
    model = AutoModelForSeq2SeqLM.from_pretrained(BASE_MODEL)
    model = model.to(device)

    # Chargement et tokenisation du dataset
    dataset = load_dataset_from_json(DATASET_JSON)
    dataset = dataset.train_test_split(test_size=0.1, seed=42)
    train_dataset = tokenize_dataset(dataset["train"], tokenizer)
    eval_dataset = tokenize_dataset(dataset["test"], tokenizer)

    logger.info(f"Train : {len(train_dataset)} exemples")
    logger.info(f"Eval  : {len(eval_dataset)} exemples")

    # Data collator
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        padding=True,
    )

    # Arguments d'entraînement
    training_args = Seq2SeqTrainingArguments(
        output_dir=str(MODEL_OUTPUT_DIR),
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        learning_rate=LEARNING_RATE,
        warmup_steps=WARMUP_STEPS,
        weight_decay=WEIGHT_DECAY,
        eval_strategy="steps",
        eval_steps=EVAL_STEPS,
        save_steps=SAVE_STEPS,
        logging_steps=LOGGING_STEPS,
        predict_with_generate=True,
        save_total_limit=1,
        load_best_model_at_end=True,
        use_mps_device=(device.type == "mps"),
    )

    # Trainer
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    # Entraînement
    logger.info("Démarrage de l'entraînement...")
    trainer.train()

    # Sauvegarde du modèle final
    logger.info(f"Sauvegarde du modèle : {MODEL_OUTPUT_DIR}")
    trainer.save_model(str(MODEL_OUTPUT_DIR))
    tokenizer.save_pretrained(str(MODEL_OUTPUT_DIR))

    logger.info("Entraînement terminé ✅")


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Entraînement BARThez — Sensi NLP")
    parser.add_argument("--epochs", type=int, default=EPOCHS, help="Nombre d'epochs")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE, help="Taille des batchs")
    args = parser.parse_args()

    train(epochs=args.epochs, batch_size=args.batch_size)
