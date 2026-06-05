"""
acquisition_video_ou_direct.py

Acquisition depuis :
1) un flux direct caméra ;
2) une vidéo .mp4.

Sauvegarde automatiquement, pour chaque geste détecté :
- un fichier .npy avec les landmarks ;
- un fichier .mp4 avec le squelette dessiné.
"""

from .utils.utils_sauvegarde import build_output_paths, save_landmarks_sequence, save_video_clip
from .utils.utils_mediapipe import landmarks_to_vector, has_hand_detected, compute_movement, draw_landmarks
from __future__ import annotations

import time
from pathlib import Path
from collections import deque

import cv2
import numpy as np
import mediapipe as mp


# =========================
# CONFIGURATION
# =========================

CAMERA_INDEX = 0
OUTPUT_DIR = Path("data/acquisitions_directes")

MOVEMENT_THRESHOLD = 0.002
MIN_ACTIVE_FRAMES = 2
SEQUENCE_LENGTH = 50
SAVE_COOLDOWN_SECONDS = 4


PRE_ROLL_FRAMES = 90
POST_ROLL_FRAMES = 0

MAX_GESTURE_FRAMES = 300
OUTPUT_FPS = 30

MOVEMENT_TIMEOUT_FRAMES = 10
SAVE_COOLDOWN_SECONDS = 0

WINDOW_NAME = "Acquisition - q pour quitter"


# =========================
# MEDIAPIPE
# =========================

# =========================
# SAUVEGARDE
# =========================


# =========================
# CHOIX DE LA SOURCE
# =========================

# =========================
# ACQUISITION GENERIQUE
# =========================

def acquisition_depuis_source(
    source: int | str,
    output_dir: Path = OUTPUT_DIR,
    sequence_length: int = SEQUENCE_LENGTH,
    movement_threshold: float = MOVEMENT_THRESHOLD,
    min_active_frames: int = MIN_ACTIVE_FRAMES,
    save_cooldown_seconds: float = SAVE_COOLDOWN_SECONDS,
) -> None:
    """
    Lit une source vidéo, caméra ou fichier, et sauvegarde automatiquement les gestes détectés.
    """

    cap = cv2.VideoCapture(source)

    if not cap.isOpened():
        raise RuntimeError(f"Impossible d'ouvrir la source vidéo : {source}")

    source_fps = cap.get(cv2.CAP_PROP_FPS)
    fps_sortie = int(source_fps) if source_fps and source_fps > 0 else OUTPUT_FPS

    landmark_buffer: deque[np.ndarray] = deque(maxlen=sequence_length)
    video_buffer: deque[np.ndarray] = deque(maxlen=sequence_length)

    previous_vector: np.ndarray | None = None
    active_frames = 0
    last_save_time = 0.0
    saved_count = 0

    print("Acquisition lancée.")
    print(f"Source : {source}")
    print("Appuie sur q dans la fenêtre vidéo pour quitter.")
    print(f"Sortie : {output_dir.resolve()}")

    with mp_holistic.Holistic(
        static_image_mode=False,
        model_complexity=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ) as holistic:

        while True:
            ret, frame = cap.read()

            if not ret:
                print("Fin de la vidéo ou frame non lue. Arrêt.")
                break

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = holistic.process(rgb)

            vector = landmarks_to_vector(results)

            movement_score = compute_movement(previous_vector, vector)
            previous_vector = vector

            hands_visible = has_hand_detected(results)
            is_moving = hands_visible and movement_score > movement_threshold

            if is_moving:
                active_frames += 1
            else:
                active_frames = max(0, active_frames - 1)

            skeleton_color = (0, 255, 0) if is_moving or active_frames > 0 else (255, 255, 255)

            annotated_frame = frame.copy()
            annotated_frame = draw_landmarks(
                annotated_frame,
                results,
                color=skeleton_color,
            )

            status = "MOUVEMENT" if is_moving else "repos"
            text_color = (0, 255, 0) if is_moving else (255, 255, 255)

            cv2.putText(
                annotated_frame,
                f"{status} | movement={movement_score:.4f} | saved={saved_count}",
                (30, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                text_color,
                2,
            )

            landmark_buffer.append(vector)
            video_buffer.append(annotated_frame.copy())

            now = time.time()
            can_save = (
                len(landmark_buffer) == sequence_length
                and active_frames >= min_active_frames
                and now - last_save_time >= save_cooldown_seconds
            )

            if can_save:
                video_to_save = []
                for saved_frame in list(video_buffer):
                    f = saved_frame.copy()
                    cv2.putText(
                        f,
                        "GESTE DETECTE",
                        (30, 80),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.9,
                        (0, 255, 0),
                        2,
                    )
                    video_to_save.append(f)

                saved_count += 1
                npy_path, mp4_path = save_hybrid_sequence(
                    landmark_sequence=list(landmark_buffer),
                    video_sequence=video_to_save,
                    output_dir=output_dir,
                    saved_count=saved_count,
                )

                last_save_time = now
                active_frames = 0

                print(f"Séquence sauvegardée #{saved_count}:")
                print(f"  landmarks : {npy_path}")
                print(f"  vidéo     : {mp4_path}")

            cv2.imshow(WINDOW_NAME, annotated_frame)

            # q pour quitter.
            # Pour une vidéo .mp4, le waitKey peut être ajusté selon le FPS.
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()

    print("Acquisition terminée.")
    print(f"{saved_count} séquence(s) sauvegardée(s).")


def acquisition_directe(
    camera_index: int = CAMERA_INDEX,
    output_dir: Path = OUTPUT_DIR,
    sequence_length: int = SEQUENCE_LENGTH,
    movement_threshold: float = MOVEMENT_THRESHOLD,
    min_active_frames: int = MIN_ACTIVE_FRAMES,
    save_cooldown_seconds: float = SAVE_COOLDOWN_SECONDS,
) -> None:
    """Compatibilité avec ton ancien appel : lance directement la caméra."""
    acquisition_depuis_source(
        source=camera_index,
        output_dir=output_dir,
        sequence_length=sequence_length,
        movement_threshold=movement_threshold,
        min_active_frames=min_active_frames,
        save_cooldown_seconds=save_cooldown_seconds,
    )


if __name__ == "__main__":
    source_video = choisir_source_video()
    acquisition_depuis_source(source_video)
