"""
Script d'entraînement du modèle LSTM pour la reconnaissance des signes LSF.
Basé sur le notebook notebooks/lstm_best_version.ipynb

Usage :
    python scripts/train_lstm.py
    python scripts/train_lstm.py --epochs 100 --batch-size 32
"""

import argparse
import json
import logging
from pathlib import Path

import yaml
import numpy as np
import tensorflow as tf
from tensorflow import keras

# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent
CONFIG_PATH = BASE_DIR / "config" / "config.yaml"

with open(CONFIG_PATH, encoding="utf-8") as f:
    config = yaml.safe_load(f)

# Chemins
MODEL_OUTPUT = BASE_DIR / config["paths"]["lstm_model"]
METADATA_OUTPUT = BASE_DIR / config["paths"]["lstm_metadata"]

# Paramètres LSTM
TARGET_FRAMES = config["lstm"]["target_frames"]
N_FEATURES = config["lstm"]["n_features"]
CONFIDENCE_THRESHOLD = config["lstm"]["confidence_threshold"]

# Paramètres d'entraînement
EPOCHS = config["training_lstm"]["epochs"]
BATCH_SIZE = config["training_lstm"]["batch_size"]
LEARNING_RATE = config["training_lstm"]["learning_rate"]
VALIDATION_SPLIT = config["training_lstm"]["validation_split"]
PATIENCE = config["training_lstm"]["patience"]

# Vocabulaire
GLOSSES = config["glosses"]

# ============================================================
# LOGGING
# ============================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s — %(levelname)s — %(message)s",
)
logger = logging.getLogger(__name__)


# ============================================================
# CHARGEMENT DES DONNÉES
# ============================================================

def load_data(data_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    """
    Charge les fichiers .npy générés par capture/video_vectorizer.py
    et retourne X (séquences) et y (labels encodés).

    Structure attendue :
        data/
            BONJOUR/
                sequence_001.npy   (shape: n_frames x n_features)
                sequence_002.npy
            MERCI/
                sequence_001.npy
            ...

    Args:
        data_dir : dossier racine contenant les sous-dossiers par gloss

    Returns:
        X : np.ndarray de shape (n_samples, TARGET_FRAMES, N_FEATURES)
        y : np.ndarray de shape (n_samples,) — indices des classes
    """
    X, y = [], []
    sign_to_idx = {sign: idx for idx, sign in enumerate(GLOSSES)}

    logger.info(f"Chargement des données depuis : {data_dir}")

    for gloss in GLOSSES:
        gloss_dir = data_dir / gloss.lower()
        if not gloss_dir.exists():
            logger.warning(f"Dossier manquant : {gloss_dir}")
            continue

        npy_files = list(gloss_dir.glob("*.npy"))
        logger.info(f"  {gloss} : {len(npy_files)} séquences")

        for npy_path in npy_files:
            sequence = np.load(npy_path)

            # Normalisation à TARGET_FRAMES frames
            if len(sequence) >= TARGET_FRAMES:
                sequence = sequence[:TARGET_FRAMES]
            else:
                pad = np.zeros((TARGET_FRAMES - len(sequence), N_FEATURES), dtype=np.float32)
                sequence = np.concatenate([sequence, pad], axis=0)

            X.append(sequence)
            y.append(sign_to_idx[gloss])

    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.int32)

    logger.info(f"Dataset chargé : {X.shape} — {len(GLOSSES)} classes")
    return X, y


# ============================================================
# MODÈLE
# ============================================================

def build_model(n_classes: int) -> keras.Model:
    """
    Construit le modèle LSTM pour la reconnaissance des signes.

    Architecture :
        LSTM(128) → Dropout → LSTM(64) → Dropout → Dense(64) → Dense(n_classes)
    """
    model = keras.Sequential([
        keras.layers.Input(shape=(TARGET_FRAMES, N_FEATURES)),
        keras.layers.LSTM(128, return_sequences=True),
        keras.layers.Dropout(0.3),
        keras.layers.LSTM(64),
        keras.layers.Dropout(0.3),
        keras.layers.Dense(64, activation="relu"),
        keras.layers.Dense(n_classes, activation="softmax"),
    ])

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    logger.info(model.summary())
    return model


# ============================================================
# ENTRAÎNEMENT
# ============================================================

def train(data_dir: Path, epochs: int = EPOCHS, batch_size: int = BATCH_SIZE):
    """Pipeline complet d'entraînement LSTM."""

    logger.info(f"TensorFlow version : {tf.__version__}")
    logger.info(f"GPU disponible : {tf.config.list_physical_devices('GPU')}")

    # Chargement des données
    X, y = load_data(data_dir)
    n_classes = len(GLOSSES)

    # Construction du modèle
    model = build_model(n_classes)

    # Callbacks
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_accuracy",
            patience=PATIENCE,
            restore_best_weights=True,
            verbose=1,
        ),
        keras.callbacks.ModelCheckpoint(
            filepath=str(MODEL_OUTPUT),
            monitor="val_accuracy",
            save_best_only=True,
            verbose=1,
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=5,
            verbose=1,
        ),
    ]

    # Entraînement
    logger.info(f"Démarrage de l'entraînement — {epochs} epochs, batch {batch_size}")
    history = model.fit(
        X, y,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=VALIDATION_SPLIT,
        callbacks=callbacks,
        verbose=1,
    )

    # Métriques finales
    best_val_acc = max(history.history["val_accuracy"])
    logger.info(f"Meilleure val_accuracy : {best_val_acc:.2%}")

    # Sauvegarde des métadonnées
    idx_to_sign = {idx: sign.lower() for idx, sign in enumerate(GLOSSES)}
    metadata = {
        "idx_to_sign": idx_to_sign,
        "n_classes": n_classes,
        "n_features": N_FEATURES,
        "target_frames": TARGET_FRAMES,
        "best_val_accuracy": best_val_acc,
        "glosses": GLOSSES,
    }
    with open(METADATA_OUTPUT, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    logger.info(f"Modèle sauvegardé : {MODEL_OUTPUT}")
    logger.info(f"Métadonnées sauvegardées : {METADATA_OUTPUT}")
    logger.info("Entraînement terminé ✅")


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Entraînement LSTM — Sensi LSF")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=BASE_DIR / "data" / "lstm",
        help="Dossier contenant les séquences .npy par gloss"
    )
    parser.add_argument("--epochs", type=int, default=EPOCHS, help="Nombre d'epochs")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE, help="Taille des batchs")
    args = parser.parse_args()

    train(data_dir=args.data_dir, epochs=args.epochs, batch_size=args.batch_size)
