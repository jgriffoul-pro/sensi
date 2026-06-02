"""
Test live du modèle SignLive sur la webcam.
Capture 1.5s de signe, prédit, affiche le résultat, recommence.

Usage:
    python test_signs_live.py

Pré-requis:
    - Fichiers ./models/signlive_v2.keras et ./models/signlive_v2_metadata.json
    - Webcam accessible
    - Packages : tensorflow, mediapipe, opencv-python, numpy

Quitter: appuyer sur Q dans la fenêtre webcam.
"""

import time
import json
from pathlib import Path

import numpy as np
import cv2
import mediapipe as mp
import tensorflow as tf
from tensorflow.keras.utils import pad_sequences


# ============================================================
# CONFIGURATION
# ============================================================
MODEL_DIR = Path('./models')
MODEL_NAME = 'signlive_v2.keras'
METADATA_NAME = 'signlive_v2_metadata.json'

WINDOW_DURATION = 1.5         # secondes d'enregistrement par signe
TARGET_FRAMES = 40            # nombre de frames attendu par le modèle
CONFIDENCE_THRESHOLD = 0.4    # seuil minimum pour afficher une prédiction

# Indices des landmarks de pose à garder (haut du corps)
# 0=nez, 11=épaule gauche, 12=épaule droite, 13=coude gauche,
# 14=coude droit, 15=poignet gauche, 16=poignet droit
POSE_INDICES = [0, 11, 12, 13, 14, 15, 16]


# ============================================================
# CHARGEMENT DU MODÈLE
# ============================================================
print('🔧 Chargement du modèle SignLive...')

model_path = MODEL_DIR / MODEL_NAME
metadata_path = MODEL_DIR / METADATA_NAME

if not model_path.exists():
    print(f'❌ Modèle introuvable : {model_path.absolute()}')
    print('   Vérifie que tu as bien lancé la cellule de sauvegarde dans ton notebook.')
    exit(1)

if not metadata_path.exists():
    print(f'❌ Métadonnées introuvables : {metadata_path.absolute()}')
    exit(1)

from tensorflow.keras.layers import LSTM as KerasLSTM

class LSTM_NoCudnn(KerasLSTM):
    def __init__(self, *args, **kwargs):
        kwargs['use_cudnn'] = False
        super().__init__(*args, **kwargs)

model = tf.keras.models.load_model(
    model_path,
    custom_objects={'LSTM': LSTM_NoCudnn},
)

with open(metadata_path, 'r') as f:
    metadata = json.load(f)

idx_to_sign = {int(k): v for k, v in metadata['idx_to_sign'].items()}

print(f'✅ Modèle chargé. {len(idx_to_sign)} signes reconnus :')
for i, sign in sorted(idx_to_sign.items()):
    print(f'   {i:2d} → {sign}')


# ============================================================
# INIT MEDIAPIPE
# ============================================================
print('\n🔧 Initialisation MediaPipe Holistic...')
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

holistic = mp_holistic.Holistic(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
    model_complexity=1,
)
print('✅ MediaPipe prêt.')


# ============================================================
# FONCTIONS D'EXTRACTION
# ============================================================
def extract_landmarks_from_frame(frame_rgb):
    """Extrait left_hand, right_hand, pose d'une frame RGB."""
    results = holistic.process(frame_rgb)

    if results.left_hand_landmarks:
        left = np.array([[lm.x, lm.y, lm.z]
                        for lm in results.left_hand_landmarks.landmark])
    else:
        left = np.zeros((21, 3))

    if results.right_hand_landmarks:
        right = np.array([[lm.x, lm.y, lm.z]
                         for lm in results.right_hand_landmarks.landmark])
    else:
        right = np.zeros((21, 3))

    if results.pose_landmarks:
        pose = np.array([[lm.x, lm.y, lm.z]
                        for lm in results.pose_landmarks.landmark])
    else:
        pose = np.zeros((33, 3))

    return left, right, pose, results


def build_features_from_sequence(lefts, rights, poses):
    """
    Construit le tableau (frames, 147) à partir des listes de landmarks.
    Applique la même normalisation que pendant l'entraînement :
    soustraction du centre des épaules, division par la largeur des épaules.
    """
    lefts = np.array(lefts, dtype=np.float32)
    rights = np.array(rights, dtype=np.float32)
    poses = np.array(poses, dtype=np.float32)

    # Filtrer la pose pour ne garder que les 7 landmarks utiles
    pose_top = poses[:, POSE_INDICES, :]

    # Normalisation par les épaules
    left_shoulder = pose_top[:, 1, :]
    right_shoulder = pose_top[:, 2, :]
    center = (left_shoulder + right_shoulder) / 2

    shoulder_width = np.linalg.norm(left_shoulder - right_shoulder, axis=1, keepdims=True)
    shoulder_width = np.maximum(shoulder_width, 1e-6)

    center_r = center[:, np.newaxis, :]
    scale_r = shoulder_width[:, np.newaxis, :]

    lefts_norm = (lefts - center_r) / scale_r
    rights_norm = (rights - center_r) / scale_r
    pose_norm = (pose_top - center_r) / scale_r

    # Aplatir chaque type
    n_frames = lefts.shape[0]
    left_flat = lefts_norm.reshape(n_frames, -1)
    right_flat = rights_norm.reshape(n_frames, -1)
    pose_flat = pose_norm.reshape(n_frames, -1)

    # Concaténer
    features = np.concatenate([left_flat, right_flat, pose_flat], axis=1)
    return features.astype(np.float32)


def predict_sign(lefts, rights, poses):
    """Construit les features, padde, prédit. Retourne (signe, confiance, top_3)."""
    if len(lefts) < 5:
        return None, 0.0, []

    lefts_arr = np.array(lefts, dtype=np.float32)
    rights_arr = np.array(rights, dtype=np.float32)
    poses_arr = np.array(poses, dtype=np.float32)

    # Garder uniquement les frames où la pose a été détectée
    # (sinon division par zéro à la normalisation + masking non contigu)
    valid_mask = np.any(poses_arr.reshape(len(poses_arr), -1) != 0, axis=1)
    
    if valid_mask.sum() < 3:
        return None, 0.0, []

    lefts_arr = lefts_arr[valid_mask]
    rights_arr = rights_arr[valid_mask]
    poses_arr = poses_arr[valid_mask]

    features = build_features_from_sequence(lefts_arr, rights_arr, poses_arr)

    # Tronquer si trop long, padder à droite si trop court
    if features.shape[0] >= TARGET_FRAMES:
        features_padded = features[:TARGET_FRAMES][np.newaxis, ...]
    else:
        n_pad = TARGET_FRAMES - features.shape[0]
        padding = np.zeros((n_pad, features.shape[1]), dtype=np.float32)
        features_padded = np.concatenate([features, padding], axis=0)[np.newaxis, ...]

    probs = model.predict(features_padded, verbose=0)[0]
    idx = int(probs.argmax())
    confidence = float(probs[idx])

    top_3_idx = np.argsort(probs)[::-1][:3]
    top_3 = [(idx_to_sign[int(i)], float(probs[i])) for i in top_3_idx]

    return idx_to_sign[idx], confidence, top_3


# ============================================================
# BOUCLE PRINCIPALE
# ============================================================
def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print('❌ Impossible d\'ouvrir la webcam.')
        print('   Sur Mac : Réglages → Confidentialité → Caméra → autoriser Terminal/VSCode.')
        return

    print(f'\n🎥 Webcam active — fenêtre de {WINDOW_DURATION}s par signe')
    print(f'   Seuil de confiance pour afficher : {CONFIDENCE_THRESHOLD:.0%}')
    print('   Appuyez sur Q pour quitter\n')

    last_prediction = '...'
    last_confidence = 0.0
    last_top_3 = []

    recording_start = time.time()
    lefts_buf, rights_buf, poses_buf = [], [], []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Effet miroir
        frame = cv2.flip(frame, 1)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Extraction des landmarks
        left, right, pose, results = extract_landmarks_from_frame(frame_rgb)

        # Dessin des landmarks
        if results.pose_landmarks:
            mp_drawing.draw_landmarks(
                frame, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
                landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style(),
            )
        if results.left_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
            )
        if results.right_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
            )

        # Buffer la frame
        lefts_buf.append(left)
        rights_buf.append(right)
        poses_buf.append(pose)

        # Si la fenêtre est complète → prédiction
        now = time.time()
        if now - recording_start >= WINDOW_DURATION:
            sign, conf, top_3 = predict_sign(lefts_buf, rights_buf, poses_buf)
            if sign is not None:
                last_prediction = sign
                last_confidence = conf
                last_top_3 = top_3
                affichage = sign if conf >= CONFIDENCE_THRESHOLD else '?'
                print(f'\n→ {affichage}  ({conf:.0%})')
                for i, (s, p) in enumerate(top_3):
                    bar = '█' * int(p * 20)
                    print(f'   {i+1}. {s:15s} {bar} {p:.0%}')

            # Reset pour le prochain signe
            recording_start = now
            lefts_buf, rights_buf, poses_buf = [], [], []

        # ===== AFFICHAGE =====
        # Prédiction principale en haut
        if last_confidence >= CONFIDENCE_THRESHOLD:
            color = (0, 255, 0)
            label = last_prediction
        else:
            color = (0, 200, 200)
            label = '?'

        text = f'{label}  ({last_confidence:.0%})'
        cv2.putText(frame, text, (20, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.4, color, 3)

        # Top 3 en dessous, petit
        for i, (s, p) in enumerate(last_top_3):
            y = 90 + i * 25
            txt = f'{i+1}. {s} ({p:.0%})'
            cv2.putText(frame, txt, (20, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

        # Barre de progression de l'enregistrement
        elapsed = now - recording_start
        progress = min(elapsed / WINDOW_DURATION, 1.0)
        bar_width = int(progress * 400)
        bar_y = frame.shape[0] - 30
        cv2.rectangle(frame, (20, bar_y), (20 + bar_width, bar_y + 15),
                      (0, 255, 255), -1)
        cv2.rectangle(frame, (20, bar_y), (420, bar_y + 15),
                      (255, 255, 255), 2)
        cv2.putText(frame, 'Enregistrement', (20, bar_y - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # Aide en bas à droite
        cv2.putText(frame, 'Q pour quitter',
                    (frame.shape[1] - 200, frame.shape[0] - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        cv2.imshow('SignLive - Test webcam', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    holistic.close()
    print('\n✅ Test terminé.')


if __name__ == '__main__':
    main()
