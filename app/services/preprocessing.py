import os
from pathlib import Path

# ============================================================
# MODIF 002 — Chemin output/ adaptatif pour Docker
# Réf : modifs/002_chemin_output_docker.txt
# ============================================================
# En local : OUTPUT_DIR non défini → chemin relatif au projet
# En Docker : OUTPUT_DIR=/app/output → défini dans le Dockerfile
# ============================================================

_output_dir_default = str(Path(__file__).resolve().parent.parent.parent / "output")
OUTPUT_DIR = Path(os.getenv("OUTPUT_DIR", _output_dir_default))

SEQUENCE_TXT = OUTPUT_DIR / "sequence.txt"
LAST_PHRASE_TXT = OUTPUT_DIR / "last_phrase.txt"


def read_sequence_from_file() -> list[str]:
    """
    Lit la dernière phrase validée (touche V) écrite par test_team_live(llm).py
    depuis output/last_phrase.txt et retourne une liste de glosses en majuscules.

    Returns:
        list[str] : liste de glosses en majuscules
                    ex: ["BONJOUR", "JE_SUIS", "CONTENT", "PRESENTER", "PROJET"]
                    [] si le fichier est vide ou introuvable
    """
    if not LAST_PHRASE_TXT.exists():
        return []

    content = LAST_PHRASE_TXT.read_text(encoding="utf-8").strip()

    if not content:
        return []

    # Conversion en majuscules pour correspondre au vocabulaire BARThez
    # ex: "bonjour je_suis content presenter" → ["BONJOUR", "JE_SUIS", "CONTENT", "PRESENTER"]
    glosses = [gloss.upper() for gloss in content.split()]

    return glosses


# ============================================================
# FONCTIONS EN ATTENTE — Webcam + MediaPipe (option future)
# À décommenter si on intègre la webcam directement dans l'API
# ============================================================

# import cv2
# import mediapipe as mp
# import numpy as np

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
#     if results.face_landmarks:
#         face = np.array(
#             [[results.face_landmarks.landmark[i].x,
#               results.face_landmarks.landmark[i].y,
#               results.face_landmarks.landmark[i].z]
#              for i in FACE_LANDMARKS_SELECTED],
#             dtype=np.float32,
#         ).flatten()
#     else:
#         face = np.zeros(len(FACE_LANDMARKS_SELECTED) * 3, dtype=np.float32)

#     if results.left_hand_landmarks:
#         left_hand = np.array(
#             [[lm.x, lm.y, lm.z] for lm in results.left_hand_landmarks.landmark],
#             dtype=np.float32,
#         ).flatten()
#     else:
#         left_hand = np.zeros(21 * 3, dtype=np.float32)

#     if results.right_hand_landmarks:
#         right_hand = np.array(
#             [[lm.x, lm.y, lm.z] for lm in results.right_hand_landmarks.landmark],
#             dtype=np.float32,
#         ).flatten()
#     else:
#         right_hand = np.zeros(21 * 3, dtype=np.float32)

#     return np.concatenate([face, left_hand, right_hand])


# def extract_keypoints_from_frame(frame_bytes: bytes) -> np.ndarray | None:
#     """Extrait les keypoints MediaPipe depuis une frame image en bytes."""
#     nparr = np.frombuffer(frame_bytes, np.uint8)
#     frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
#     rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#     results = holistic.process(rgb)
#     vector = landmarks_to_vector_no_pose(results)
#     return vector


# def extract_keypoints_from_video(video_bytes: bytes) -> np.ndarray | None:
#     """Extrait les keypoints MediaPipe depuis une vidéo en bytes."""
#     tmp_path = Path("/tmp/sensi_upload.mp4")
#     tmp_path.write_bytes(video_bytes)
#     cap = cv2.VideoCapture(str(tmp_path))
#     frames = []
#     while len(frames) < 60:
#         ret, frame = cap.read()
#         if not ret:
#             break
#         rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#         results = holistic.process(rgb)
#         frames.append(landmarks_to_vector_no_pose(results))
#     cap.release()
#     if not frames:
#         return None
#     sequence = np.array(frames, dtype=np.float32)
#     if sequence.shape[0] < 60:
#         pad = np.zeros((60 - sequence.shape[0], 150), dtype=np.float32)
#         sequence = np.concatenate([sequence, pad], axis=0)
#     return sequence[:60]
