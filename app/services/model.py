from pathlib import Path
from app.services.preprocessing import read_sequence_from_file

# ============================================================
# CHARGEMENT MODELE LSTM — sensi_team_v6.keras
# ============================================================

# import json
# import numpy as np
# import tensorflow as tf
# from collections import deque

# MODEL_DIR = Path("models")
# MODEL_NAME = "sensi_team_v6.keras"
# METADATA_NAME = "sensi_team_v6_metadata.json"

# CONFIDENCE_THRESHOLD = 0.50
# TARGET_FRAMES = 60

# model_lstm = tf.keras.models.load_model(MODEL_DIR / MODEL_NAME)

# with open(MODEL_DIR / METADATA_NAME, "r") as f:
#     metadata = json.load(f)

# idx_to_sign = {int(k): v for k, v in metadata["idx_to_sign"].items()}


def get_glosses_from_sequence() -> list[str]:
    """
    Récupère les glosses depuis output/sequence.txt (écrit par test_team_live).
    Filtre le signe 'INCONNU' qui ne doit pas être transmis au NLP.

    Returns:
        list[str] : liste de glosses en majuscules, sans INCONNU
                    ex: ["BONJOUR", "JE_SUIS", "CONTENT"]
    """
    glosses = read_sequence_from_file()

    # Filtre le signe inconnu — ne doit pas être transmis au modèle NLP
    glosses = [g for g in glosses if g != "INCONNU"]

    return glosses


# ============================================================
# FONCTION EN ATTENTE — Inférence LSTM directe (option future)
# À décommenter si on intègre la webcam directement dans l'API
# ============================================================

# def predict_sign(sequence: np.ndarray) -> tuple[str, float]:
#     """
#     Prédit le signe depuis une séquence de 60 frames x 150 features.
#
#     Args:
#         sequence : np.ndarray de shape (60, 150)
#
#     Returns:
#         tuple : (signe prédit en minuscules, confiance)
#                 ex: ("bonjour", 0.92)
#     """
#     batch = sequence[np.newaxis, ...]   # (1, 60, 150)
#     probs = model_lstm.predict(batch, verbose=0)[0]
#     pred_idx = int(np.argmax(probs))
#     confidence = float(probs[pred_idx])
#     sign = idx_to_sign[pred_idx]
#     return sign, confidence
