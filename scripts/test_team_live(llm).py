"""
Test live SENSI v6 — avec ENREGISTREMENT AUTOMATIQUE de la séquence.

Modèle : sensi_team_v6.keras (sans pose, 150 features)

À chaque signe validé, la séquence est sauvegardée automatiquement dans :
    output/sequence.txt        (séquence courante en cours de construction)
    output/sequence.json       (avec timestamps + confidences)

Quand tu appuies sur V (valider la phrase), elle est aussi loguée dans :
    output/phrases.log         (historique de toutes les phrases validées)

Touches :
    R = reset sequence | V = valider phrase | Q = quitter
"""

import json
import os
from collections import deque
from datetime import datetime
from pathlib import Path

import numpy as np
import cv2
import mediapipe as mp
import tensorflow as tf
import requests as http_requests


MODEL_DIR = Path('./models')
MODEL_NAME = 'sensi_team_v6.keras'
METADATA_NAME = 'sensi_team_v6_metadata.json'

# Dossier d'export — créé automatiquement si absent
OUTPUT_DIR = Path('./output')
OUTPUT_DIR.mkdir(exist_ok=True)

SEQUENCE_TXT = OUTPUT_DIR / 'sequence.txt'
SEQUENCE_JSON = OUTPUT_DIR / 'sequence.json'
PHRASES_LOG = OUTPUT_DIR / 'phrases.log'

TARGET_FRAMES = 60
PREDICT_EVERY_N_FRAMES = 5
CONFIDENCE_THRESHOLD = 0.50
STABILITY_FRAMES = 5
COOLDOWN_FRAMES = 30
SKIP_INCONNU = True

FACE_LANDMARKS_SELECTED = [13, 14, 61, 291, 159, 386, 70, 300]

# MODIF 008 — API directe (Cloud Run ou local Docker)
# Réf : modifs/008_live_api_integration.txt
API_URL = os.getenv('SENSI_API_URL', 'https://sensi-api-294525653433.europe-west1.run.app/api/v1')

# MODIF 001 — Mapping vocabulaire BiLSTM → NLP (côté client)
LABEL_MAP = {
    "SOURDE": "SOURD",
    "ENTENDANT": "ENTENDANTS",
}


# ============================================================
# CHARGEMENT MODELE
# ============================================================
print('🔧 Chargement du modèle Sensi v6...')
model_path = MODEL_DIR / MODEL_NAME
metadata_path = MODEL_DIR / METADATA_NAME

if not model_path.exists():
    print(f'❌ Modèle introuvable : {model_path.absolute()}')
    exit(1)

model = tf.keras.models.load_model(model_path)

with open(metadata_path, 'r') as f:
    metadata = json.load(f)

idx_to_sign = {int(k): v for k, v in metadata['idx_to_sign'].items()}

print(f'✅ Modèle chargé. {len(idx_to_sign)} signes.')
print(f'   Features : {metadata.get("n_features", "?")} (sans pose)')
if 'best_val_accuracy' in metadata:
    print(f'📊 Val accuracy : {metadata["best_val_accuracy"]:.2%}')

print(f'\n📁 Export auto vers :')
print(f'   {SEQUENCE_TXT.absolute()}')
print(f'   {SEQUENCE_JSON.absolute()}')
print(f'   {PHRASES_LOG.absolute()}')
print(f'\n🌐 API : {API_URL}')


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


def landmarks_to_vector_no_pose(results) -> np.ndarray:
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

    return np.concatenate([face, left_hand, right_hand])


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
# MODIF 008 — Envoi direct à l'API
# Réf : modifs/008_live_api_integration.txt
# ============================================================
LAST_AUDIO = OUTPUT_DIR / 'last_audio.mp3'


def send_to_api(glosses: list[str]) -> str | None:
    """
    Envoie les glosses à l'API, récupère phrase + audio.
    Retourne la phrase traduite ou None si erreur.
    """
    api_glosses = [LABEL_MAP.get(g.upper(), g.upper()) for g in glosses]
    payload = {"glosses": api_glosses}

    try:
        phrase_resp = http_requests.post(
            f'{API_URL}/predict/sentence',
            json=payload,
            timeout=60,
        )
        phrase_resp.raise_for_status()
        phrase = phrase_resp.json().get('phrase', '')

        audio_resp = http_requests.post(
            f'{API_URL}/predict/sentence/audio',
            json=payload,
            timeout=60,
        )
        audio_resp.raise_for_status()

        LAST_AUDIO.write_bytes(audio_resp.content)
        print(f'🗣️  {phrase}')
        os.system(f'afplay "{LAST_AUDIO}" &')
        return phrase

    except http_requests.exceptions.ConnectionError:
        print(f'❌ API inaccessible : {API_URL}')
    except http_requests.exceptions.Timeout:
        print(f'❌ API timeout (>60s)')
    except Exception as e:
        print(f'❌ Erreur API : {e}')

    return None


# ============================================================
# EXPORT AUTO
# ============================================================
def save_sequence(detected_sequence, detected_meta):
    """Sauve la séquence courante dans txt + json. Appelée à chaque ajout."""
    # Fichier texte simple : un mot par ligne, séparés par espace pour le LLM
    SEQUENCE_TXT.write_text(' '.join(detected_sequence), encoding='utf-8')

    # Fichier JSON avec timestamps et confidences
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
    """Ajoute la phrase validée au log historique."""
    timestamp = datetime.now().isoformat(timespec='seconds')
    with open(PHRASES_LOG, 'a', encoding='utf-8') as f:
        f.write(f'[{timestamp}] {phrase}\n')

    # Sauvegarde de la dernière phrase validée — lue par l'API
    LAST_PHRASE = OUTPUT_DIR / 'last_phrase.txt'
    LAST_PHRASE.write_text(phrase, encoding='utf-8')


# ============================================================
# BOUCLE PRINCIPALE
# ============================================================
def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print('❌ Webcam inaccessible.')
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, 30)

    print(f'\n🎥 Live v6 — export auto')
    print(f'   R = reset | V = valider phrase | Q = quitter\n')

    sequence_buf = deque(maxlen=TARGET_FRAMES)
    stability_history = deque(maxlen=STABILITY_FRAMES)
    detected_sequence = []         # liste de strings : ["bonjour", "merci", ...]
    detected_meta = []             # liste de dicts avec confidence + timestamp
    last_added_sign = None
    cooldown_remaining = 0
    last_validated_phrase = ''
    api_response_line = ''

    current_pred_idx = -1
    current_pred_conf = 0.0
    current_top_3 = []
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = holistic.process(rgb)

        vector = landmarks_to_vector_no_pose(results)
        sequence_buf.append(vector)

        if cooldown_remaining > 0:
            cooldown_remaining -= 1

        if len(sequence_buf) == TARGET_FRAMES and frame_count % PREDICT_EVERY_N_FRAMES == 0:
            sequence = np.array(sequence_buf, dtype=np.float32)
            batch = sequence[np.newaxis, ...]
            probs = model.predict(batch, verbose=0)[0]

            top_3_idx = np.argsort(probs)[::-1][:3]
            current_pred_idx = int(top_3_idx[0])
            current_pred_conf = float(probs[current_pred_idx])
            current_top_3 = [(idx_to_sign[int(i)], float(probs[i])) for i in top_3_idx]

            if current_pred_conf >= CONFIDENCE_THRESHOLD:
                stability_history.append(current_pred_idx)
            else:
                stability_history.clear()

            if (len(stability_history) == STABILITY_FRAMES
                and len(set(stability_history)) == 1
                and cooldown_remaining == 0):

                validated_sign = idx_to_sign[stability_history[0]]

                if SKIP_INCONNU and validated_sign == 'inconnu':
                    print(f'   (inconnu detecte mais non ajoute)')
                    stability_history.clear()
                    cooldown_remaining = 10
                elif validated_sign != last_added_sign:
                    detected_sequence.append(validated_sign)
                    detected_meta.append({
                        'sign': validated_sign,
                        'confidence': round(current_pred_conf, 3),
                        'timestamp': datetime.now().isoformat(timespec='seconds'),
                    })
                    last_added_sign = validated_sign
                    cooldown_remaining = COOLDOWN_FRAMES
                    stability_history.clear()
                    print(f'✅ {validated_sign} ({current_pred_conf:.0%})  → sauvegardé')

                    # ENREGISTREMENT AUTOMATIQUE
                    save_sequence(detected_sequence, detected_meta)

        frame = draw_landmarks(frame, results)

        # ===== AFFICHAGE =====
        h, w = frame.shape[:2]

        if current_pred_idx >= 0 and current_pred_conf >= CONFIDENCE_THRESHOLD:
            label = idx_to_sign[current_pred_idx]
            if label == 'inconnu' and SKIP_INCONNU:
                color = (100, 100, 255)
            else:
                color = (0, 255, 0)
        else:
            color = (0, 200, 200)
            label = '?' if current_pred_idx < 0 else idx_to_sign[current_pred_idx]

        cv2.putText(frame, f'{label}  ({current_pred_conf:.0%})', (20, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.4, color, 3)

        for i, (s, p) in enumerate(current_top_3):
            y = 90 + i * 28
            col = (255, 255, 255) if i == 0 else (180, 180, 180)
            if s == 'inconnu' and SKIP_INCONNU and i == 0:
                col = (100, 100, 255)
            cv2.putText(frame, f'{i+1}. {s} ({p:.0%})', (20, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, col, 1)

        stab_text = f'Stabilite : {len(stability_history)}/{STABILITY_FRAMES}'
        stab_color = (0, 255, 0) if len(stability_history) >= STABILITY_FRAMES else (200, 200, 0)
        cv2.putText(frame, stab_text, (w - 280, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, stab_color, 2)

        if cooldown_remaining > 0:
            cv2.putText(frame, f'Cooldown : {cooldown_remaining}', (w - 280, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 100, 255), 1)

        cv2.putText(frame, f'Buffer : {len(sequence_buf)}/{TARGET_FRAMES}', (w - 280, 110),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

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

        # MODIF 008 — affichage réponse API
        if api_response_line:
            cv2.putText(frame, api_response_line, (20, h - 155),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2)

        cv2.putText(frame, 'R = reset  |  V = valider phrase  |  Q = quitter',
                    (20, h - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)

        cv2.imshow('Sensi v6 - Live avec export auto', frame)

        frame_count += 1

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        if key == ord('r'):
            detected_sequence.clear()
            detected_meta.clear()
            last_added_sign = None
            last_validated_phrase = ''
            api_response_line = ''
            cooldown_remaining = 0
            save_sequence(detected_sequence, detected_meta)
            print('🔄 Reset (fichiers vides).')
        if key == ord('v'):
            if detected_sequence:
                last_validated_phrase = ' '.join(detected_sequence)
                print(f'\n📝 Phrase validee : {last_validated_phrase}')

                # MODIF 008 — envoi direct à l'API
                api_phrase = send_to_api(detected_sequence)
                if api_phrase:
                    api_response_line = f'→ {api_phrase}'
                else:
                    api_response_line = '→ API indisponible'

                log_phrase(last_validated_phrase)
                detected_sequence.clear()
                detected_meta.clear()
                last_added_sign = None
                save_sequence(detected_sequence, detected_meta)
                print()
            else:
                print('⚠️ Sequence vide.')
                api_response_line = ''

    cap.release()
    cv2.destroyAllWindows()
    holistic.close()
    print('\n✅ Fini.')


if __name__ == '__main__':
    main()
