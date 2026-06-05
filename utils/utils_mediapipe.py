
from __future__ import annotations



import numpy as np
import mediapipe as mp


mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils


def landmarks_to_vector(results) -> np.ndarray:
    """Convertit les landmarks MediaPipe en vecteur fixe de 258 valeurs."""

    if results.pose_landmarks:
        pose = np.array(
            [[lm.x, lm.y, lm.z, lm.visibility] for lm in results.pose_landmarks.landmark],
            dtype=np.float32,
        ).flatten()
    else:
        pose = np.zeros(33 * 4, dtype=np.float32)

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

    return np.concatenate([pose, left_hand, right_hand])


def has_hand_detected(results) -> bool:
    """Retourne True si au moins une main est détectée."""
    return results.left_hand_landmarks is not None or results.right_hand_landmarks is not None


def compute_movement(previous_vector: np.ndarray | None, current_vector: np.ndarray) -> float:
    """Calcule le mouvement moyen des mains entre deux frames."""
    if previous_vector is None:
        return 0.0

    previous_hands = previous_vector[132:]
    current_hands = current_vector[132:]

    return float(np.mean(np.abs(current_hands - previous_hands)))


def draw_landmarks(frame, results, color=(255, 255, 255)):
    """Dessine le squelette MediaPipe avec une couleur donnée."""

    drawing_spec = mp_drawing.DrawingSpec(
        color=color,
        thickness=2,
        circle_radius=2,
    )

    if results.pose_landmarks:
        mp_drawing.draw_landmarks(
            frame,
            results.pose_landmarks,
            mp_holistic.POSE_CONNECTIONS,
            drawing_spec,
            drawing_spec,
        )

    if results.left_hand_landmarks:
        mp_drawing.draw_landmarks(
            frame,
            results.left_hand_landmarks,
            mp_holistic.HAND_CONNECTIONS,
            drawing_spec,
            drawing_spec,
        )

    if results.right_hand_landmarks:
        mp_drawing.draw_landmarks(
            frame,
            results.right_hand_landmarks,
            mp_holistic.HAND_CONNECTIONS,
            drawing_spec,
            drawing_spec,
        )

    return frame
