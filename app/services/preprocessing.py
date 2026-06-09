"""
Service Preprocessing — Sensi
Extraction des keypoints MediaPipe (option future).
"""

# ============================================================
# FONCTIONS EN ATTENTE — Webcam + MediaPipe (option future)
# À décommenter si on intègre la webcam directement dans l'API
# ============================================================

# import cv2
# import mediapipe as mp
# import numpy as np
# from pathlib import Path

# FACE_LANDMARKS_SELECTED = [13, 14, 61, 291, 159, 386, 70, 300]

# mp_holistic = mp.solutions.holistic
# holistic = mp_holistic.Holistic(
#     static_image_mode=False,
#     model_complexity=1,
#     refine_face_landmarks=True,
#     min_detection_confidence=0.4,
#     min_tracking_confidence=0.4,
# )


# def landmarks_to_vector_no_pose(results) -> np.ndarray:
#     """Extrait 150 features (visage + mains) depuis les résultats MediaPipe."""
#     ...

# def extract_keypoints_from_frame(frame_bytes: bytes) -> np.ndarray | None:
#     """Extrait les keypoints MediaPipe depuis une frame image en bytes."""
#     ...

# def extract_keypoints_from_video(video_bytes: bytes) -> np.ndarray | None:
#     """Extrait les keypoints MediaPipe depuis une vidéo en bytes."""
#     ...
