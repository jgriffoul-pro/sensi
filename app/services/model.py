"""
Service Model — Sensi
Chargement et inférence du modèle BiLSTM (option future).
"""

# ============================================================
# CHARGEMENT MODELE LSTM — sensi_team_v6.keras (option future)
# À décommenter si on intègre la webcam directement dans l'API
# ============================================================

# import json
# import numpy as np
# import tensorflow as tf
# from pathlib import Path

# MODEL_DIR = Path("models")
# MODEL_NAME = "sensi_team_v6.keras"
# METADATA_NAME = "sensi_team_v6_metadata.json"

# CONFIDENCE_THRESHOLD = 0.50
# TARGET_FRAMES = 60

# model_lstm = tf.keras.models.load_model(MODEL_DIR / MODEL_NAME)

# with open(MODEL_DIR / METADATA_NAME, "r") as f:
#     metadata = json.load(f)

# idx_to_sign = {int(k): v for k, v in metadata["idx_to_sign"].items()}


# ============================================================
# FONCTION EN ATTENTE — Inférence LSTM directe (option future)
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
#     batch = sequence[np.newaxis, ...]
#     probs = model_lstm.predict(batch, verbose=0)[0]
#     pred_idx = int(np.argmax(probs))
#     confidence = float(probs[pred_idx])
#     sign = idx_to_sign[pred_idx]
#     return sign, confidence
