"""
Test live SENSI — Duel de modeles.

Deux modeles en parallele, le meilleur gagne.
    model_1 : sensi_team_v6  (150 features, sans pose, 21 classes)
    model_2 : Model_VTH_20   (564 features, avec pose + velocity, 21 classes)

A chaque prediction, les deux modeles votent.
Celui avec la meilleure confiance est affiche et sauvegarde.

Touches :
    R = reset sequence | V = valider phrase | Q = quitter
"""

import json
from collections import deque
from datetime import datetime
from pathlib import Path

import numpy as np
import cv2
import mediapipe as mp
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences


# ============================================================
# CONFIG
# ============================================================

MODEL_DIR = Path('./models')

OUTPUT_DIR = Path('./output')
OUTPUT_DIR.mkdir(exist_ok=True)

SEQUENCE_TXT = OUTPUT_DIR / 'sequence.txt'
SEQUENCE_JSON = OUTPUT_DIR / 'sequence.json'
PHRASES_LOG = OUTPUT_DIR / 'phrases.log'
SESSION_LOG = OUTPUT_DIR / 'session_log.json'

TARGET_FRAMES = 60
PREDICT_EVERY_N_FRAMES = 5
CONFIDENCE_THRESHOLD = 0.50
STABILITY_FRAMES = 3
COOLDOWN_FRAMES = 30
NO_HANDS_THRESHOLD = 10
FALLBACK_CONFIDENCE = 0.40

FACE_LANDMARKS_SELECTED = [13, 14, 61, 291, 159, 386, 70, 300]


# ============================================================
# FEATURE EXTRACTION
# ============================================================

def landmarks_to_vector_no_pose(results) -> np.ndarray:
    if results.face_landmarks:
        face = np.array(
            [[results.face_landmarks.landmark[i].x,
              results.face_landmarks.landmark[i].y,
              results.face_landmarks.landmark[i].z]
             for i in FACE_LANDMARKS_SELECTED],
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

    return np.concatenate([face, left_hand, right_hand])


def landmarks_to_vector_with_pose(results) -> np.ndarray:
    if results.pose_landmarks:
        pose = np.array(
            [[lm.x, lm.y, lm.z, lm.visibility] for lm in results.pose_landmarks.landmark],
            dtype=np.float32,
        ).flatten()
    else:
        pose = np.zeros(33 * 4, dtype=np.float32)

    if results.face_landmarks:
        face = np.array(
            [[results.face_landmarks.landmark[i].x,
              results.face_landmarks.landmark[i].y,
              results.face_landmarks.landmark[i].z]
             for i in FACE_LANDMARKS_SELECTED],
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


# ============================================================
# PREPROCESSING MODEL 2 (VTH)
# ============================================================

def add_velocity(X: np.ndarray) -> np.ndarray:
    velocity = np.zeros_like(X)
    velocity[:, 1:, :] = X[:, 1:, :] - X[:, :-1, :]
    return np.concatenate([X, velocity], axis=-1)


def preprocess_vth(sequence: np.ndarray, max_len: int, mean: float, std: float) -> np.ndarray:
    X = sequence[np.newaxis, ...].astype(np.float32)
    X = pad_sequences(X, maxlen=max_len, dtype="float32", padding="post", truncating="post", value=0.0)
    X = np.where(X != 0.0, (X - mean) / std, 0.0)
    X = add_velocity(X)
    return X


# ============================================================
# CHARGEMENT MODELES
# ============================================================

print('Chargement des modeles...')

# --- Model 1 : sensi_team_v6 ---
model_1_path = MODEL_DIR / 'sensi_team_v6.keras'
model_1_meta_path = MODEL_DIR / 'sensi_team_v6_metadata.json'

model_1 = tf.keras.models.load_model(model_1_path)
with open(model_1_meta_path, 'r') as f:
    meta_1 = json.load(f)
idx_to_sign_1 = {int(k): v for k, v in meta_1['idx_to_sign'].items()}
N_CLASSES_1 = meta_1['n_classes']

print(f'  [1] sensi_team_v6 : {N_CLASSES_1} classes, 150 features, val={meta_1.get("best_val_accuracy", "?"):.1%}')

# --- Model 2 : Model_VTH_20 ---
model_2_path = MODEL_DIR / 'Model_VTH_20.keras'
model_2_meta_path = MODEL_DIR / 'Model_VTH_20.json'

model_2 = tf.keras.models.load_model(model_2_path)
with open(model_2_meta_path, 'r') as f:
    meta_2 = json.load(f)
labels_2 = meta_2['labels']
VTH_MEAN = meta_2['normalization']['mean']
VTH_STD = meta_2['normalization']['std']
VTH_MAX_LEN = meta_2['max_len']
N_CLASSES_2 = meta_2['num_classes']

val_2 = meta_2.get("val_accuracy")
val_2_str = f'{val_2:.1%}' if isinstance(val_2, (int, float)) else '?'
print(f'  [2] Model_VTH_20  : {N_CLASSES_2} classes, 564 features, val={val_2_str}')

SKIP_INCONNU = True

print(f'\nExport auto vers :')
print(f'   {SEQUENCE_TXT.absolute()}')
print(f'   {SEQUENCE_JSON.absolute()}')
print(f'   {PHRASES_LOG.absolute()}')


# ============================================================
# MEDIAPIPE
# ============================================================

mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

holistic = mp_holistic.Holistic(
    static_image_mode=False,
    model_complexity=1,
    refine_face_landmarks=True,
    min_detection_confidence=0.4,
    min_tracking_confidence=0.4,
)


def draw_landmarks(frame, results):
    if results.pose_landmarks:
        mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)
    if results.face_landmarks:
        mp_drawing.draw_landmarks(frame, results.face_landmarks, mp_holistic.FACEMESH_CONTOURS)
    if results.left_hand_landmarks:
        mp_drawing.draw_landmarks(frame, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
    if results.right_hand_landmarks:
        mp_drawing.draw_landmarks(frame, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
    return frame


# ============================================================
# EXPORT
# ============================================================

def save_sequence(detected_sequence, detected_meta):
    SEQUENCE_TXT.write_text(' '.join(detected_sequence), encoding='utf-8')
    SEQUENCE_JSON.write_text(
        json.dumps({
            'signs': detected_sequence,
            'phrase_brute': ' '.join(detected_sequence),
            'details': detected_meta,
            'last_update': datetime.now().isoformat(timespec='seconds'),
        }, ensure_ascii=False, indent=2),
        encoding='utf-8'
    )


def log_phrase(phrase):
    timestamp = datetime.now().isoformat(timespec='seconds')
    with open(PHRASES_LOG, 'a', encoding='utf-8') as f:
        f.write(f'[{timestamp}] {phrase}\n')
    (OUTPUT_DIR / 'last_phrase.txt').write_text(phrase, encoding='utf-8')


# ============================================================
# BOUCLE PRINCIPALE
# ============================================================

def hands_visible(results) -> bool:
    return results.left_hand_landmarks is not None or results.right_hand_landmarks is not None


def main():
    cap = cv2.VideoCapture(1)
    if not cap.isOpened():
        print('Webcam inaccessible.')
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, 30)

    print(f'\nLive duel — [1] team_v6 vs [2] VTH-20')
    print(f'   R = reset | V = valider phrase | Q = quitter\n')

    buf_1 = deque(maxlen=TARGET_FRAMES)
    buf_2 = deque(maxlen=TARGET_FRAMES)
    stability_history = deque(maxlen=STABILITY_FRAMES)
    detected_sequence = []
    detected_meta = []
    last_added_sign = None
    cooldown_remaining = 0
    last_validated_phrase = ''
    no_hands_count = 0

    best_label = '?'
    best_conf = 0.0
    best_model = 0
    top3_1 = []
    top3_2 = []
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = holistic.process(rgb)

        has_hands = hands_visible(results)

        if has_hands:
            no_hands_count = 0
            buf_1.append(landmarks_to_vector_no_pose(results))
            buf_2.append(landmarks_to_vector_with_pose(results))
        else:
            no_hands_count += 1
            if no_hands_count >= NO_HANDS_THRESHOLD:
                buf_1.clear()
                buf_2.clear()
                stability_history.clear()

        if cooldown_remaining > 0:
            cooldown_remaining -= 1

        can_predict = (
            len(buf_1) == TARGET_FRAMES
            and frame_count % PREDICT_EVERY_N_FRAMES == 0
            and has_hands
            and no_hands_count == 0
        )

        if can_predict:

            # --- Model 1 : team_v6 (150 features, raw) ---
            seq_1 = np.array(buf_1, dtype=np.float32)
            probs_1 = model_1.predict(seq_1[np.newaxis, ...], verbose=0)[0]
            idx1 = int(np.argmax(probs_1))
            conf1 = float(probs_1[idx1])
            sign1 = idx_to_sign_1.get(idx1, '?')
            top3_1 = [(idx_to_sign_1.get(int(i), '?'), float(probs_1[i])) for i in np.argsort(probs_1)[::-1][:3]]

            # --- Model 2 : VTH (282 features → normalize → velocity → 564) ---
            seq_2 = preprocess_vth(np.array(buf_2, dtype=np.float32), VTH_MAX_LEN, VTH_MEAN, VTH_STD)
            probs_2 = model_2.predict(seq_2, verbose=0)[0]
            idx2 = int(np.argmax(probs_2))
            conf2 = float(probs_2[idx2])
            sign2 = labels_2[idx2] if idx2 < len(labels_2) else '?'
            top3_2 = [(labels_2[int(i)] if int(i) < len(labels_2) else '?', float(probs_2[i])) for i in np.argsort(probs_2)[::-1][:3]]

            # --- VTH-20 principal, v6 fallback ---
            if conf2 >= FALLBACK_CONFIDENCE:
                best_label = sign2
                best_conf = conf2
                best_model = 2
            else:
                best_label = sign1
                best_conf = conf1
                best_model = 1

            if best_conf >= CONFIDENCE_THRESHOLD:
                stability_history.append(best_label)
            else:
                stability_history.clear()

            if (len(stability_history) == STABILITY_FRAMES
                and len(set(stability_history)) == 1
                and cooldown_remaining == 0):

                validated_sign = stability_history[0]

                if SKIP_INCONNU and validated_sign == 'inconnu':
                    print(f'   (inconnu detecte mais non ajoute)')
                    stability_history.clear()
                    cooldown_remaining = 10
                else:
                    detected_sequence.append(validated_sign)
                    detected_meta.append({
                        'sign': validated_sign,
                        'confidence': round(best_conf, 3),
                        'winner': f'model_{best_model}',
                        'model_1': {'sign': sign1, 'confidence': round(conf1, 3)},
                        'model_2': {'sign': sign2, 'confidence': round(conf2, 3)},
                        'timestamp': datetime.now().isoformat(timespec='seconds'),
                    })
                    last_added_sign = validated_sign
                    cooldown_remaining = COOLDOWN_FRAMES
                    stability_history.clear()
                    print(f'  {validated_sign} ({best_conf:.0%}) via [{best_model}]  -> sauvegarde')
                    save_sequence(detected_sequence, detected_meta)

        frame = draw_landmarks(frame, results)

        # ===== AFFICHAGE =====
        h, w = frame.shape[:2]

        if best_conf >= CONFIDENCE_THRESHOLD:
            color = (0, 255, 0)
        else:
            color = (0, 200, 200)

        model_tag = f'[M{best_model}]' if best_model else ''
        cv2.putText(frame, f'{best_label}  ({best_conf:.0%}) {model_tag}', (20, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.4, color, 3)

        if top3_1:
            cv2.putText(frame, 'M1 (v6):', (20, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 180, 255), 1)
            for i, (s, p) in enumerate(top3_1[:3]):
                cv2.putText(frame, f'  {s} ({p:.0%})', (20, 112 + i * 22),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                            (255, 255, 255) if i == 0 else (160, 160, 160), 1)

        if top3_2:
            x_off = w // 2
            cv2.putText(frame, 'M2 (VTH-20):', (x_off, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 255, 180), 1)
            for i, (s, p) in enumerate(top3_2[:3]):
                cv2.putText(frame, f'  {s} ({p:.0%})', (x_off, 112 + i * 22),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                            (255, 255, 255) if i == 0 else (160, 160, 160), 1)

        stab_text = f'Stabilite : {len(stability_history)}/{STABILITY_FRAMES}'
        stab_color = (0, 255, 0) if len(stability_history) >= STABILITY_FRAMES else (200, 200, 0)
        cv2.putText(frame, stab_text, (w - 280, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, stab_color, 2)

        if cooldown_remaining > 0:
            cv2.putText(frame, f'Cooldown : {cooldown_remaining}', (w - 280, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 100, 255), 1)

        cv2.putText(frame, f'Buffer : {len(buf_1)}/{TARGET_FRAMES}', (w - 280, 110),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        if no_hands_count > 0:
            cv2.putText(frame, f'Pas de mains : {no_hands_count}', (w - 280, 140),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        else:
            cv2.putText(frame, 'Auto-export -> output/sequence.txt', (w - 380, 140),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 200), 1)

        overlay = frame.copy()
        cv2.rectangle(overlay, (0, h - 130), (w, h), (0, 0, 0), -1)
        frame = cv2.addWeighted(overlay, 0.7, frame, 0.3, 0)

        cv2.putText(frame, 'Sequence detectee :', (20, h - 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        if detected_sequence:
            seq_text = ' -> '.join(detected_sequence)
            if len(seq_text) > 80:
                seq_text = '... ' + seq_text[-77:]
            cv2.putText(frame, seq_text, (20, h - 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        else:
            cv2.putText(frame, '(vide)', (20, h - 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (150, 150, 150), 2)

        if last_validated_phrase:
            cv2.putText(frame, f'Phrase : {last_validated_phrase}', (20, h - 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        cv2.putText(frame, 'R = reset  |  V = valider phrase  |  Q = quitter',
                    (20, h - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)

        cv2.imshow('Sensi — Duel', frame)

        frame_count += 1

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        if key == ord('r'):
            detected_sequence.clear()
            detected_meta.clear()
            last_added_sign = None
            last_validated_phrase = ''
            cooldown_remaining = 0
            save_sequence(detected_sequence, detected_meta)
            print('Reset.')
        if key == ord('v'):
            if detected_sequence:
                last_validated_phrase = ' '.join(detected_sequence)
                print(f'\nPhrase validee : {last_validated_phrase}\n')
                log_phrase(last_validated_phrase)
                detected_sequence.clear()
                detected_meta.clear()
                last_added_sign = None
                save_sequence(detected_sequence, detected_meta)
            else:
                print('Sequence vide.')

    cap.release()
    cv2.destroyAllWindows()
    holistic.close()

    if detected_meta:
        export = {
            'sequence': detected_sequence,
            'phrase': ' '.join(detected_sequence),
            'details': detected_meta,
            'closed_at': datetime.now().isoformat(timespec='seconds'),
        }
        SESSION_LOG.write_text(json.dumps(export, ensure_ascii=False, indent=2), encoding='utf-8')
        print(f'\nExport session -> {SESSION_LOG.absolute()}')
    else:
        print('\nAucun signe detecte, pas d\'export.')

    print('Fini.')


if __name__ == '__main__':
    main()
