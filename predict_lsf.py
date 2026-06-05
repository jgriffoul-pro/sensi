"""
predict_lsf.py — Inférence du modèle LSF (LSF Sign Language Recognition)

Usage :
    python predict_lsf.py                          # Webcam (appuyer sur ESPACE pour prédire, Q pour quitter)
    python predict_lsf.py --video chemin/video.mp4 # Prédire sur une vidéo enregistrée
    python predict_lsf.py --video chemin/video.mp4 --window 45  # Fenêtre personnalisée
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import cv2
import numpy as np
import mediapipe as mp

# TensorFlow lazy import (heavy)
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences


# ─────────────────────────────
# CONFIG
# ─────────────────────────────

MODEL_PATH = Path(__file__).parent / "model_lsf_bilstm.keras"
METADATA_PATH = Path(__file__).parent / "model_lsf_metadata.json"

# MediaPipe
mp_holistic = mp.solutions.holistic
mp_draw = mp.solutions.drawing_utils

FACE_LANDMARKS_SELECTED = [
    13, 14, 61, 291,    # bouche
    159, 386,            # yeux
    70, 300,             # sourcils
]


# ─────────────────────────────
# FEATURE EXTRACTION (identique à video_vectorizer.py)
# ─────────────────────────────

def landmarks_to_vector(results) -> np.ndarray:
    """282 features : pose(132) + face(24) + left_hand(63) + right_hand(63)."""
    if results.pose_landmarks:
        pose = np.array(
            [[lm.x, lm.y, lm.z, lm.visibility] for lm in results.pose_landmarks.landmark],
            dtype=np.float32,
        ).flatten()
    else:
        pose = np.zeros(33 * 4, dtype=np.float32)

    if results.face_landmarks:
        face = np.array(
            [
                [
                    results.face_landmarks.landmark[i].x,
                    results.face_landmarks.landmark[i].y,
                    results.face_landmarks.landmark[i].z,
                ]
                for i in FACE_LANDMARKS_SELECTED
            ],
            dtype=np.float32,
        ).flatten()
    else:
        face = np.zeros(len(FACE_LANDMARKS_SELECTED) * 3, dtype=np.float32)

    if results.left_hand_landmarks:
        left_hand = np.array(
            [[lm.x, lm.y, lm.z] for lm in results.left_hand_landmarks.landmark],
            dtype=np.float32,
        ).flatten()
    else:
        left_hand = np.zeros(21 * 3, dtype=np.float32)

    if results.right_hand_landmarks:
        right_hand = np.array(
            [[lm.x, lm.y, lm.z] for lm in results.right_hand_landmarks.landmark],
            dtype=np.float32,
        ).flatten()
    else:
        right_hand = np.zeros(21 * 3, dtype=np.float32)

    return np.concatenate([pose, face, left_hand, right_hand])


# ─────────────────────────────
# PREPROCESSING PIPELINE
# ─────────────────────────────

def add_velocity(X: np.ndarray) -> np.ndarray:
    """Concatène les vitesses frame-to-frame (identique au notebook)."""
    velocity = np.zeros_like(X)
    velocity[:, 1:, :] = X[:, 1:, :] - X[:, :-1, :]
    return np.concatenate([X, velocity], axis=-1)


def preprocess(sequence: list[np.ndarray], max_len: int, mean: float, std: float) -> np.ndarray:
    """
    Transforme une liste de vecteurs 282-d en input modèle (1, 60, 564).
    Pipeline : pad → normalize → add_velocity.
    """
    X = np.array(sequence, dtype=np.float32)[np.newaxis, ...]  # (1, T, 282)

    # Pad
    X = pad_sequences(X, maxlen=max_len, dtype="float32", padding="post", truncating="post", value=0.0)

    # Normalize (même formule que le notebook)
    X = np.where(X != 0.0, (X - mean) / std, 0.0)

    # Velocity
    X = add_velocity(X)

    return X  # (1, 60, 564)


# ─────────────────────────────
# PREDICTION
# ─────────────────────────────

def predict(model, sequence: list[np.ndarray], max_len: int, mean: float, std: float, labels: list[str]):
    """Prédit le signe à partir d'une séquence de vecteurs 282-d."""
    if len(sequence) < 5:
        return None, 0.0

    X = preprocess(sequence, max_len, mean, std)
    probs = model.predict(X, verbose=0)[0]
    idx = int(np.argmax(probs))
    return labels[idx], float(probs[idx])


# ─────────────────────────────
# WEBCAM MODE
# ─────────────────────────────

def run_phrase(model, max_len: int, mean: float, std: float, labels: list[str]) -> list[list[str]]:
    cap = cv2.VideoCapture(1)
    if not cap.isOpened():
        print("Erreur : impossible d'ouvrir la webcam.")
        sys.exit(1)

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    window = int(fps * 2)
    prediction_interval = max(1, window // 3)
    stop_label = "STOP"

    buffer: list[np.ndarray] = []
    frame_idx = 0
    prediction_text = ""
    prediction_conf = 0.0
    last_predicted = ""

    phrases: list[list[str]] = [[]]
    current_phrase: list[str] = []

    print(f"\nLSF Phrase Builder — Mode continu ({window/fps:.1f}s)")
    print(f"STOP = nouvelle phrase | Q = quitter et afficher\n")

    with mp_holistic.Holistic(
        static_image_mode=False,
        model_complexity=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ) as holistic:

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = holistic.process(rgb)

            if results.pose_landmarks:
                mp_draw.draw_landmarks(frame, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)
            if results.left_hand_landmarks:
                mp_draw.draw_landmarks(frame, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
            if results.right_hand_landmarks:
                mp_draw.draw_landmarks(frame, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)

            vector = landmarks_to_vector(results)
            buffer.append(vector)
            if len(buffer) > window:
                buffer = buffer[-window:]

            if len(buffer) >= window and frame_idx % prediction_interval == 0:
                label, conf = predict(model, buffer, max_len, mean, std, labels)
                if label and conf > 0.5 and label != last_predicted and label.strip():
                    prediction_text = label
                    prediction_conf = conf
                    last_predicted = label
                    print(f"  → {label} ({conf:.1%})")

                    if label == stop_label:
                        if current_phrase:
                            phrases[-1] = current_phrase
                            print(f"  ── Phrase {len(phrases)} : {' '.join(current_phrase)}")
                        phrases.append([])
                        current_phrase = []
                    else:
                        current_phrase.append(label)

            h, w = frame.shape[:2]

            cv2.putText(frame, f"{len(buffer)}/{window}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

            if current_phrase:
                cv2.putText(frame, " ".join(current_phrase), (10, 65),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 200), 2)

            if prediction_text:
                color = (0, 255, 0) if prediction_conf > 0.8 else (0, 200, 200)
                cv2.putText(frame, f"{prediction_text} ({prediction_conf:.1%})", (10, h - 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)

            phrase_num_text = f"Phrase {len(phrases)}"
            cv2.putText(frame, phrase_num_text, (w - 200, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

            cv2.imshow("LSF Phrase Builder", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

            frame_idx += 1

    cap.release()
    cv2.destroyAllWindows()

    if current_phrase:
        phrases[-1] = current_phrase

    phrases = [p for p in phrases if p]

    print(f"\n{'='*50}")
    print("Résultat :")
    for i, phrase in enumerate(phrases, 1):
        print(f"  Phrase {i} : {' '.join(phrase)}")
    print(f"{'='*50}")

    return phrases
    cap = cv2.VideoCapture(1)
    if not cap.isOpened():
        print("Erreur : impossible d'ouvrir la webcam.")
        sys.exit(1)

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    window = int(fps * 2)
    prediction_interval = max(1, window // 3)

    buffer: list[np.ndarray] = []
    frame_idx = 0
    prediction_text = ""
    prediction_conf = 0.0

    print(f"\nLSF Predictor — Mode continu ({window} frames / {window/fps:.1f}s)")
    print("Q = quitter\n")

    with mp_holistic.Holistic(
        static_image_mode=False,
        model_complexity=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ) as holistic:

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = holistic.process(rgb)

            if results.pose_landmarks:
                mp_draw.draw_landmarks(frame, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)
            if results.left_hand_landmarks:
                mp_draw.draw_landmarks(frame, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
            if results.right_hand_landmarks:
                mp_draw.draw_landmarks(frame, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)

            vector = landmarks_to_vector(results)
            buffer.append(vector)
            if len(buffer) > window:
                buffer = buffer[-window:]

            if len(buffer) >= window and frame_idx % prediction_interval == 0:
                label, conf = predict(model, buffer, max_len, mean, std, labels)
                if label and conf > 0.5:
                    prediction_text = label
                    prediction_conf = conf
                    print(f"  → {label} ({conf:.1%})")

            h, w = frame.shape[:2]

            cv2.putText(frame, f"{len(buffer)}/{window}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

            if prediction_text:
                color = (0, 255, 0) if prediction_conf > 0.8 else (0, 200, 200)
                cv2.putText(frame, f"{prediction_text} ({prediction_conf:.1%})", (10, h - 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)

            cv2.imshow("LSF Predictor", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

            frame_idx += 1

    cap.release()
    cv2.destroyAllWindows()


# ─────────────────────────────
# WEBCAM MODE (continu)
# ─────────────────────────────

def run_webcam(model, max_len: int, mean: float, std: float, labels: list[str]):
    cap = cv2.VideoCapture(1)
    if not cap.isOpened():
        print("Erreur : impossible d'ouvrir la webcam.")
        sys.exit(1)

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    window = int(fps * 2)
    prediction_interval = max(1, window // 3)

    buffer: list[np.ndarray] = []
    frame_idx = 0
    prediction_text = ""
    prediction_conf = 0.0
    last_predicted = ""

    print(f"\nLSF Predictor — Mode continu ({window/fps:.1f}s)")
    print("Q = quitter\n")

    with mp_holistic.Holistic(
        static_image_mode=False,
        model_complexity=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ) as holistic:

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = holistic.process(rgb)

            if results.pose_landmarks:
                mp_draw.draw_landmarks(frame, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)
            if results.left_hand_landmarks:
                mp_draw.draw_landmarks(frame, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
            if results.right_hand_landmarks:
                mp_draw.draw_landmarks(frame, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)

            vector = landmarks_to_vector(results)
            buffer.append(vector)
            if len(buffer) > window:
                buffer = buffer[-window:]

            if len(buffer) >= window and frame_idx % prediction_interval == 0:
                label, conf = predict(model, buffer, max_len, mean, std, labels)
                if label and conf > 0.5 and label != last_predicted and label.strip():
                    prediction_text = label
                    prediction_conf = conf
                    last_predicted = label
                    print(f"  → {label} ({conf:.1%})")

            h, w = frame.shape[:2]

            cv2.putText(frame, f"{len(buffer)}/{window}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

            if prediction_text:
                color = (0, 255, 0) if prediction_conf > 0.8 else (0, 200, 200)
                cv2.putText(frame, f"{prediction_text} ({prediction_conf:.1%})", (10, h - 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)

            cv2.imshow("LSF Predictor", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

            frame_idx += 1

    cap.release()
    cv2.destroyAllWindows()


# ─────────────────────────────
# VIDEO MODE
# ─────────────────────────────

def run_video(model, video_path: str, window: int, max_len: int, mean: float, std: float, labels: list[str]):
    """Prédit des signes sur une vidéo avec une fenêtre glissante."""

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Erreur : impossible d'ouvrir {video_path}")
        sys.exit(1)

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    print(f"\nVidéo : {video_path}")
    print(f"Frames : {total_frames} | FPS : {fps:.0f}")
    print(f"Fenêtre de prédiction : {window} frames\n")

    sequence: list[np.ndarray] = []
    predictions: list[tuple[str, float]] = []

    with mp_holistic.Holistic(
        static_image_mode=False,
        model_complexity=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ) as holistic:

        frame_idx = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = holistic.process(rgb)

            vector = landmarks_to_vector(results)
            sequence.append(vector)

            # Dessiner
            if results.pose_landmarks:
                mp_draw.draw_landmarks(frame, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)
            if results.left_hand_landmarks:
                mp_draw.draw_landmarks(frame, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
            if results.right_hand_landmarks:
                mp_draw.draw_landmarks(frame, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)

            # Prédire toutes les `window` frames
            if len(sequence) >= window:
                label, conf = predict(model, sequence, max_len, mean, std, labels)
                if label:
                    predictions.append((label, conf))
                    color = (0, 255, 0) if conf > 0.8 else (0, 200, 200)
                    cv2.putText(frame, f"{label} ({conf:.1%})", (10, frame.shape[0] - 20),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)
                    print(f"  Frame {frame_idx:4d} → {label} ({conf:.1%})")
                sequence = []  # reset pour la prochaine fenêtre

            # Frame counter
            cv2.putText(frame, f"{frame_idx}/{total_frames}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

            cv2.imshow("LSF Predictor — Video", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

            frame_idx += 1

    cap.release()
    cv2.destroyAllWindows()

    # Résumé
    if predictions:
        print(f"\n{'='*40}")
        print("Résultats :")
        for label, conf in predictions:
            print(f"  {label:25s} {conf:.1%}")
        print(f"{'='*40}")
    else:
        print("\nAucune prédiction (pas assez de frames ou mains non détectées).")


# ─────────────────────────────
# MAIN
# ─────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Prédiction LSF — vidéo ou webcam")
    parser.add_argument("--video", type=str, help="Chemin vers une vidéo .mp4")
    parser.add_argument("--window", type=int, default=60, help="Fenêtre de frames pour la prédiction vidéo (défaut: 60)")
    parser.add_argument("--phrase", action="store_true", help="Mode phrase : accumule les mots, STOP = nouvelle phrase")
    args = parser.parse_args()

    # Charger modèle + métadonnées
    if not MODEL_PATH.exists():
        print(f"Modèle introuvable : {MODEL_PATH}")
        sys.exit(1)
    if not METADATA_PATH.exists():
        print(f"Métadonnées introuvables : {METADATA_PATH}")
        sys.exit(1)

    print("Chargement du modèle...")
    model = load_model(MODEL_PATH)

    with open(METADATA_PATH, "r", encoding="utf-8") as f:
        meta = json.load(f)

    labels = meta["labels"]
    max_len = meta["max_len"]
    mean = meta["normalization"]["mean"]
    std = meta["normalization"]["std"]

    print(f"Classes : {len(labels)} | Input : {meta['input_shape']}")
    print(f"Labels  : {', '.join(labels)}")

    if args.video:
        run_video(model, args.video, args.window, max_len, mean, std, labels)
    elif args.phrase:
        result = run_phrase(model, max_len, mean, std, labels)
        print(f"\nListe retournée : {result}")
    else:
        run_webcam(model, max_len, mean, std, labels)


if __name__ == "__main__":
    main()
