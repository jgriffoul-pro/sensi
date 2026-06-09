"""
SoundWatch App — version simple et qui marche.
Son détecté = émoji géant qui s'affiche. Sans animation bricolée.
"""

import json
import time
from pathlib import Path

import numpy as np
import sounddevice as sd
import streamlit as st
import tensorflow as tf
import tensorflow_hub as hub


# ============================================================
# CONFIG
# ============================================================
SAMPLE_RATE = 16000
BUFFER_SECONDS = 2

MODEL_DIR = Path('./models')
MLP_NAME = 'soundwatch_v5_yamnet.keras'
METADATA_NAME = 'soundwatch_v5_yamnet_metadata.json'

CONFIDENCE_THRESHOLDS = {
    'siren':           0.50,
    'clock_alarm':     0.50,
    'crying_baby':     0.50,
    'door_wood_knock': 0.40,
    'clapping':        0.85,
}
DEFAULT_THRESHOLD = 0.50

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

SOUND_DISPLAY = {
    'siren':           {'emoji': '🚨', 'name': 'Sirène',            'color': '#EF4444'},
    'clock_alarm':     {'emoji': '⏰', 'name': 'Alarme',            'color': '#F59E0B'},
    'crying_baby':     {'emoji': '👶', 'name': 'Bébé qui pleure',   'color': '#EC4899'},
    'door_wood_knock': {'emoji': '🚪', 'name': 'Toc à la porte',    'color': '#3B82F6'},
    'clapping':        {'emoji': '👏', 'name': 'Applaudissements',  'color': '#10B981'},
}


st.set_page_config(
    page_title='SoundWatch',
    page_icon='🔊',
    layout='wide',
    initial_sidebar_state='collapsed',
)


@st.cache_resource
def load_models():
    yamnet = hub.load('https://tfhub.dev/google/yamnet/1')
    mlp = tf.keras.models.load_model(MODEL_DIR / MLP_NAME)
    with open(MODEL_DIR / METADATA_NAME, 'r') as f:
        metadata = json.load(f)
    idx_to_category = {int(k): v for k, v in metadata['idx_to_category'].items()}
    return yamnet, mlp, idx_to_category, metadata


if 'listening' not in st.session_state:
    st.session_state.listening = False


def record_and_predict(yamnet, mlp, idx_to_category):
    audio = sd.rec(
        int(BUFFER_SECONDS * SAMPLE_RATE),
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype='float32',
    )
    sd.wait()
    audio = audio.flatten()

    rms = float(np.sqrt(np.mean(audio**2)))

    scores, embeddings, _ = yamnet(audio)
    scores_mean = tf.reduce_mean(scores, axis=0).numpy()

    top_k_indices = np.argsort(scores_mean)[::-1][:YAMNET_TOP_K]
    filter_match = None
    for idx in top_k_indices:
        if idx in YAMNET_FILTER_CLASSES and scores_mean[idx] >= YAMNET_FILTER_THRESHOLD:
            filter_match = (int(idx), YAMNET_FILTER_CLASSES[int(idx)], float(scores_mean[idx]))
            break

    embedding_avg = tf.reduce_mean(embeddings, axis=0).numpy()
    probs = mlp.predict(embedding_avg[np.newaxis, :], verbose=0)[0]
    mlp_idx = int(probs.argmax())
    mlp_conf = float(probs[mlp_idx])
    mlp_class = idx_to_category[mlp_idx]

    threshold = CONFIDENCE_THRESHOLDS.get(mlp_class, DEFAULT_THRESHOLD)
    validated = (filter_match is None) and (mlp_conf >= threshold)

    return {
        'rms': rms,
        'filter_match': filter_match,
        'mlp_class': mlp_class,
        'mlp_conf': mlp_conf,
        'threshold': threshold,
        'validated': validated,
    }


# ============================================================
# CSS (sans animation, juste le look)
# ============================================================
st.markdown("""
<style>
    .stApp { background-color: #0E1117; }

    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

    .main-title {
        text-align: center; color: white;
        font-size: 3rem; font-weight: 300; margin-top: 2rem;
    }
    .subtitle {
        text-align: center; color: #888;
        font-size: 1.2rem; margin-top: -1rem; margin-bottom: 3rem;
    }
    .stButton > button[kind="primary"] {
        background-color: #10B981; color: white;
        font-size: 1.3rem; font-weight: 500;
        padding: 0.8rem 3rem; border-radius: 12px; border: none;
        transition: all 0.2s ease;
    }
    .stButton > button[kind="primary"]:hover { background-color: #059669; transform: translateY(-2px); }
    .stButton > button[kind="secondary"] {
        background-color: #EF4444; color: white;
        font-size: 1.3rem; font-weight: 500;
        padding: 0.8rem 3rem; border-radius: 12px; border: none;
        transition: all 0.2s ease;
    }
    .stButton > button[kind="secondary"]:hover { background-color: #DC2626; transform: translateY(-2px); }
    .status-text { text-align: center; color: #4ADE80; font-size: 1.1rem; margin-top: 2rem; }
    .status-text-off { text-align: center; color: #888; font-size: 1.1rem; margin-top: 2rem; }

    .detection-zone {
        min-height: 500px;
        display: flex;
        align-items: center;
        justify-content: center;
        flex-direction: column;
        margin: 3rem auto;
        text-align: center;
    }

    .sound-emoji {
        font-size: 18rem;
        line-height: 1;
        margin-bottom: 1rem;
    }
    .sound-name {
        font-size: 3rem;
        font-weight: 400;
        margin-top: 1rem;
    }
    .sound-confidence {
        font-size: 1.2rem;
        color: #94A3B8;
        margin-top: 0.5rem;
        font-weight: 300;
    }
    .listening-state {
        color: #6B7280;
        font-size: 1.5rem;
        font-weight: 300;
    }
    .filtered-emoji {
        font-size: 5rem;
        opacity: 0.4;
        margin-bottom: 1rem;
    }
    .filtered-name {
        color: #6B7280;
        font-size: 1.5rem;
        font-weight: 300;
    }

    .audio-level-container {
        position: fixed;
        bottom: 2rem;
        left: 50%;
        transform: translateX(-50%);
        width: 80%;
        max-width: 600px;
        padding: 0.8rem 1.5rem;
        background: #1E293B;
        border-radius: 12px;
        z-index: 100;
    }
    .audio-level-label {
        color: #94A3B8;
        font-size: 0.75rem;
        margin-bottom: 0.4rem;
        text-align: center;
    }
    .audio-level-bar {
        width: 100%;
        height: 8px;
        background: #334155;
        border-radius: 4px;
        overflow: hidden;
    }
    .audio-level-fill {
        height: 100%;
        background: linear-gradient(90deg, #10B981 0%, #FCD34D 60%, #EF4444 100%);
        border-radius: 4px;
    }
</style>
""", unsafe_allow_html=True)


st.markdown('<div class="main-title">🔊 SoundWatch</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Détection sonore pour personnes sourdes</div>', unsafe_allow_html=True)


with st.spinner('🔧 Chargement des modèles (YAMNet + MLP)...'):
    yamnet, mlp, idx_to_category, metadata = load_models()


col1, col2, col3 = st.columns([1, 1, 1])
with col2:
    if not st.session_state.listening:
        if st.button('▶️  Démarrer l\'écoute', key='start', type='primary', use_container_width=True):
            st.session_state.listening = True
            st.rerun()
    else:
        if st.button('⏸  Arrêter', key='stop', type='secondary', use_container_width=True):
            st.session_state.listening = False
            st.rerun()


if st.session_state.listening:
    st.markdown('<div class="status-text">🎤 Écoute en cours...</div>', unsafe_allow_html=True)
else:
    st.markdown('<div class="status-text-off">⏸ En attente</div>', unsafe_allow_html=True)


if st.session_state.listening:
    detection_placeholder = st.empty()
    level_placeholder = st.empty()

    # Affichage initial
    detection_placeholder.markdown("""
        <div class="detection-zone">
            <div class="listening-state">👂 En écoute...</div>
        </div>
    """, unsafe_allow_html=True)

    while st.session_state.listening:
        # Capture + prédiction (2 sec bloquantes)
        result = record_and_predict(yamnet, mlp, idx_to_category)

        # Niveau audio
        level_pct = min(result['rms'] * 2000, 100)
        level_placeholder.markdown(f"""
            <div class="audio-level-container">
                <div class="audio-level-label">Niveau audio</div>
                <div class="audio-level-bar">
                    <div class="audio-level-fill" style="width: {level_pct}%;"></div>
                </div>
            </div>
        """, unsafe_allow_html=True)

        if result['filter_match'] is not None:
            # Speech, silence, noise... → affichage discret
            idx, name, score = result['filter_match']
            detection_placeholder.markdown(f"""
                <div class="detection-zone">
                    <div class="filtered-emoji">💬</div>
                    <div class="filtered-name">{name}</div>
                </div>
            """, unsafe_allow_html=True)
        elif result['validated']:
            # Son détecté → émoji géant
            display = SOUND_DISPLAY.get(
                result['mlp_class'],
                {'emoji': '🔊', 'name': result['mlp_class'], 'color': '#FFFFFF'}
            )
            detection_placeholder.markdown(f"""
                <div class="detection-zone">
                    <div class="sound-emoji">{display['emoji']}</div>
                    <div class="sound-name" style="color: {display['color']};">{display['name']}</div>
                    <div class="sound-confidence">Confiance : {result['mlp_conf']:.0%}</div>
                </div>
            """, unsafe_allow_html=True)
        else:
            # Sous seuil → en écoute
            detection_placeholder.markdown("""
                <div class="detection-zone">
                    <div class="listening-state">👂 En écoute...</div>
                </div>
            """, unsafe_allow_html=True)
