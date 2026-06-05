from __future__ import annotations

import os
from pathlib import Path

import cv2
import numpy as np
import mediapipe as mp


# =========================
# CONFIGURATION
# =========================

VIDEOS_DIR = Path("prompteur")
OUTPUT_DIR = Path("train_data/prompteur")

OUTPUT_FPS = 30

# Optionnel : évite certains soucis protobuf sans modifier ton venv
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"


# =========================
# MEDIAPIPE
# =========================

mp_holistic = mp.solutions.holistic


FACE_LANDMARKS_SELECTED = [
    # bouche
    13,   # lèvre supérieure
    14,   # lèvre inférieure
    61,   # coin gauche bouche
    291,  # coin droit bouche

    # yeux
    159,  # oeil gauche
    386,  # oeil droit

    # sourcils
    70,   # sourcil gauche
    300,  # sourcil droit
]



def landmarks_to_vector(results) -> np.ndarray:
    """
    Convertit les landmarks MediaPipe en vecteur fixe.

    Contenu :
    - pose complète
    - visage sélectionné
    - main gauche complète
    - main droite complète
    """

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


# =========================
# TRAITEMENT VIDEO
# =========================

def vectorize_video(video_path: Path, output_dir: Path) -> Path:
    """
    Lit une vidéo complète et sauvegarde tous les landmarks dans :
    {nom_de_la_video}.npy
    """

    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        raise RuntimeError(f"Impossible d'ouvrir la vidéo : {video_path}")

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{video_path.stem}.npy"

    sequence: list[np.ndarray] = []

    print(f"\nTraitement : {video_path.name}")

    with mp_holistic.Holistic(
        static_image_mode=False,
        model_complexity=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ) as holistic:

        frame_count = 0

        while True:
            ret, frame = cap.read()

            if not ret:
                break

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = holistic.process(rgb)

            vector = landmarks_to_vector(results)
            sequence.append(vector)

            frame_count += 1

    cap.release()

    if not sequence:
        raise RuntimeError(f"Aucune frame traitée pour : {video_path}")

    arr = np.array(sequence, dtype=np.float32)
    np.save(output_path, arr)

    print(f"Frames : {len(sequence)}")
    print(f"Shape  : {arr.shape}")
    print(f"Sauvé  : {output_path}")

    return output_path


def vectorize_folder(
    videos_dir: Path = VIDEOS_DIR,
    output_dir: Path = OUTPUT_DIR,
) -> None:
    """
    Vectorise toutes les vidéos .mp4 du dossier.
    Produit un .npy par vidéo.
    """

    video_files = sorted(videos_dir.glob("*.mp4"))

    if not video_files:
        raise FileNotFoundError(f"Aucune vidéo .mp4 trouvée dans : {videos_dir.resolve()}")

    print("Vectorisation dataset lancée.")
    print(f"Dossier vidéos : {videos_dir.resolve()}")
    print(f"Sortie         : {output_dir.resolve()}")
    print(f"Vidéos trouvées : {len(video_files)}")

    for video_path in video_files:
        vectorize_video(video_path, output_dir)

    print("\nTerminé.")
    print(f"{len(video_files)} fichier(s) .npy généré(s).")


# =========================
# MAIN
# =========================

if __name__ == "__main__":
    vectorize_folder()
