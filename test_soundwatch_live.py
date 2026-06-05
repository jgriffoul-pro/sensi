"""
Test live SoundWatch v5 (YAMNet + MLP) — avec FILTRE + SEUILS PAR CLASSE.

- Filtre YAMNet : ignore speech, silence, conversation, noise
- Seuils par classe : 85% pour clapping et door (sons souvent faux positifs)
                     50% pour siren, alarm, baby (plus distincts)

USAGE :
    python test_soundwatch_live.py

Ctrl+C pour quitter.
"""

import json
import time
from collections import deque
from pathlib import Path

import numpy as np
import sounddevice as sd
import tensorflow as tf
import tensorflow_hub as hub


# ============================================================
# CONFIGURATION
# ============================================================
MODEL_DIR = Path('./models')
MLP_NAME = 'soundwatch_v5_yamnet.keras'
METADATA_NAME = 'soundwatch_v5_yamnet_metadata.json'

SAMPLE_RATE = 16000
BUFFER_SECONDS = 2
BUFFER_SIZE = SAMPLE_RATE * BUFFER_SECONDS

PREDICT_INTERVAL = 1.0

# Seuils PAR CLASSE (au lieu d'un seuil unique)
# 85% pour les sons souvent en faux positif (transients brefs)
# 50% pour les sons plus distinctifs
CONFIDENCE_THRESHOLDS = {
    'siren':           0.50,
    'clock_alarm':     0.50,
    'crying_baby':     0.50,
    'door_wood_knock': 0.85,    # ← strict
    'clapping':        0.85,    # ← strict
    'dog':             0.50,
    'cat':             0.50,
    'glass_breaking':  0.50,
}
DEFAULT_THRESHOLD = 0.50

# Classes YAMNet à filtrer (parole, silence, bruit ambiant)
YAMNET_FILTER_CLASSES = {
    0:   'Speech',
    2:   'Conversation',
    3:   'Narration',
    494: 'Silence',
    507: 'Noise',
    508: 'Environmental noise',
    500: 'Inside, small room',
    501: 'Inside, large room',
    502: 'Inside, public space',
}
YAMNET_TOP_K = 5
YAMNET_FILTER_THRESHOLD = 0.15

EMOJIS = {
    'siren':           '🚨',
    'clock_alarm':     '⏰',
    'crying_baby':     '👶',
    'door_wood_knock': '🚪',
    'clapping':        '👏',
    'dog':             '🐶',
    'cat':             '🐱',
    'glass_breaking':  '🥃',
}


# ============================================================
# CHARGEMENT
# ============================================================
print('🔧 Chargement de YAMNet...')
yamnet_model = hub.load('https://tfhub.dev/google/yamnet/1')
print('✅ YAMNet chargé.')

print(f'\n🔧 Chargement du classifieur ({MLP_NAME})...')
mlp = tf.keras.models.load_model(MODEL_DIR / MLP_NAME)
print('✅ MLP chargé.')

with open(MODEL_DIR / METADATA_NAME, 'r') as f:
    metadata = json.load(f)

idx_to_category = {int(k): v for k, v in metadata['idx_to_category'].items()}
classes = metadata['classes']

print(f'\n📊 Classes Sensi : {classes}')
print(f'📊 Val accuracy : {metadata.get("best_val_accuracy", "?"):.2%}')
print(f'\n🎯 Seuils de confiance par classe :')
for cls in classes:
    threshold = CONFIDENCE_THRESHOLDS.get(cls, DEFAULT_THRESHOLD)
    marker = ' ⬅️ STRICT' if threshold >= 0.80 else ''
    print(f'     {cls:18s} : {threshold:.0%}{marker}')


# ============================================================
# BUFFER AUDIO
# ============================================================
audio_buffer = deque(maxlen=BUFFER_SIZE)


def audio_callback(indata, frames, time_info, status):
    if status:
        print(f'⚠️ {status}')
    audio_buffer.extend(indata[:, 0])


def predict_current_buffer():
    if len(audio_buffer) < BUFFER_SIZE:
        return None
    
    audio = np.array(audio_buffer, dtype=np.float32)
    rms = float(np.sqrt(np.mean(audio**2)))
    
    scores, embeddings, spectrogram = yamnet_model(audio)
    scores_mean = tf.reduce_mean(scores, axis=0).numpy()
    
    # Filtre YAMNet
    top_k_indices = np.argsort(scores_mean)[::-1][:YAMNET_TOP_K]
    filter_match = None
    for idx in top_k_indices:
        if idx in YAMNET_FILTER_CLASSES and scores_mean[idx] >= YAMNET_FILTER_THRESHOLD:
            filter_match = (int(idx), YAMNET_FILTER_CLASSES[int(idx)], float(scores_mean[idx]))
            break
    
    # MLP
    embedding_avg = tf.reduce_mean(embeddings, axis=0).numpy()
    probs = mlp.predict(embedding_avg[np.newaxis, :], verbose=0)[0]
    mlp_idx = int(probs.argmax())
    mlp_conf = float(probs[mlp_idx])
    mlp_class = idx_to_category[mlp_idx]
    
    return {
        'rms': rms,
        'filter_match': filter_match,
        'mlp_class': mlp_class,
        'mlp_conf': mlp_conf,
    }


# ============================================================
# MAIN
# ============================================================
def main():
    print('\n' + '=' * 60)
    print('🎤 SOUNDWATCH LIVE — écoute en cours')
    print('=' * 60)
    print(f'Buffer : {BUFFER_SECONDS}s à {SAMPLE_RATE} Hz')
    print('Ctrl+C pour quitter\n')
    
    last_pred_time = time.time()
    
    try:
        with sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=1,
            callback=audio_callback,
            blocksize=int(SAMPLE_RATE * 0.1),
        ):
            while True:
                time.sleep(0.05)
                
                now = time.time()
                if now - last_pred_time < PREDICT_INTERVAL:
                    continue
                last_pred_time = now
                
                result = predict_current_buffer()
                
                if result is None:
                    print(f'  ⏳ Remplissage du buffer... ({len(audio_buffer)}/{BUFFER_SIZE})')
                    continue
                
                level_bar = '▁▂▃▄▅▆▇█'
                level_idx = min(int(result['rms'] * 50), 7)
                level = level_bar[level_idx]
                
                # Filtre YAMNet
                if result['filter_match'] is not None:
                    idx, name, score = result['filter_match']
                    print(f'  💬 [filtré: {name} {score:.0%}]  [niveau {level}]')
                    continue
                
                # Seuil par classe
                threshold = CONFIDENCE_THRESHOLDS.get(result['mlp_class'], DEFAULT_THRESHOLD)
                emoji = EMOJIS.get(result['mlp_class'], '🔊')
                
                if result['mlp_conf'] >= threshold:
                    color = '\033[92m'
                    print(f'{color}  {emoji}  {result["mlp_class"]:18s}  {result["mlp_conf"]:5.0%}  [niveau {level}]\033[0m')
                else:
                    # En dessous du seuil (le seuil peut être 85% pour clapping/door, 50% pour les autres)
                    print(f'  👂 ({result["mlp_class"]} {result["mlp_conf"]:.0%} < {threshold:.0%})  [niveau {level}]')
    
    except KeyboardInterrupt:
        print('\n\n✅ Fini.')


if __name__ == '__main__':
    main()
