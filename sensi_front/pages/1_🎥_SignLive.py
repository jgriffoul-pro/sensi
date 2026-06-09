"""
SignLive — Reconnaissance LSF en temps réel dans Streamlit.

Webcam → MediaPipe Holistic (150 features, no pose) → LSTM v6 → vote majoritaire
→ séquence détectée → bouton Traduire → API NLP+TTS → audio.

Layout demo day : sidebar masquée, vue split webcam/infos, zéro scroll.
"""

import json
import os
from collections import deque
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
import requests
import streamlit as st
import tensorflow as tf
from dotenv import load_dotenv


# ============================================================
# CONFIG
# ============================================================

load_dotenv()
API_URL = os.getenv("API_URL", "http://localhost:8000/api/v1")

st.set_page_config(
    page_title="SignLive — Sensi",
    page_icon="🎥",
    layout="wide",
    initial_sidebar_state="collapsed",  # sidebar fermée par défaut
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
MODEL_DIR = PROJECT_ROOT / "models"
MODEL_NAME = "sensi_team_v6.keras"
METADATA_NAME = "sensi_team_v6_metadata.json"

TARGET_FRAMES = 60
FACE_LANDMARKS_SELECTED = [13, 14, 61, 291, 159, 386, 70, 300]


# ============================================================
# CSS — style demo day
# ============================================================

st.markdown("""
<style>
.main .block-container { padding-top: 3.2rem; padding-bottom: 1rem; max-width: 100%; }

.sensi-card {
    background: #ffffff;
    border: 1px solid #e5e7eb;
    border-radius: 12px;
    padding: 0.9rem 1.05rem;
    margin-bottom: 0.7rem;
}
.sensi-eyebrow {
    font-family: ui-monospace, "SF Mono", Menlo, monospace;
    font-size: 0.7rem; letter-spacing: 0.14em; text-transform: uppercase;
    color: #6b7280; margin-bottom: 0.55rem;
}
.sensi-big {
    font-size: 1.6rem; font-weight: 700; line-height: 1.15; letter-spacing: -0.01em;
    color: #16a34a;
}
.sensi-meta {
    display: flex; flex-wrap: wrap; gap: 1.1rem; margin-top: 0.55rem;
    font-family: ui-monospace, "SF Mono", Menlo, monospace; font-size: 0.82rem;
    color: #4b5563;
}
.sensi-meta b { color: #111827; font-weight: 600; }

.t3-row { display: flex; align-items: center; gap: 0.7rem; margin: 0.5rem 0; }
.t3-name { flex: 0 0 110px; font-weight: 600; font-size: 0.92rem; color: #16a34a; }
.t3-name.dim { color: #4b5563; }
.t3-bar-bg { flex: 1; height: 7px; background: #f3f4f6; border-radius: 4px; overflow: hidden; }
.t3-bar { height: 100%; background: #16a34a; border-radius: 4px; }
.t3-bar.dim { background: #d1d5db; }
.t3-pct { flex: 0 0 42px; text-align: right; font-family: ui-monospace, monospace; font-size: 0.82rem; color: #4b5563; }

.seq-wrap { display: flex; flex-wrap: wrap; align-items: center; gap: 0.4rem; }
.seq-pill {
    display: inline-block; padding: 0.38rem 0.7rem; border-radius: 6px;
    background: #ecfdf5; border: 1px solid #a7f3d0;
    color: #15803d; font-family: ui-monospace, monospace; font-size: 0.84rem; font-weight: 600;
}
.seq-pill.last { background: #16a34a; border-color: #16a34a; color: #ffffff; }
.seq-arr { color: #9ca3af; font-family: ui-monospace, monospace; font-size: 0.9rem; }
.seq-empty { color: #9ca3af; font-style: italic; font-size: 0.92rem; }

.sensi-title { font-size: 1.7rem; font-weight: 800; letter-spacing: -0.02em; margin: 0; color: #111827; }
.sensi-title .accent { color: #16a34a; }
.sensi-subtitle { color: #6b7280; font-size: 0.9rem; margin-top: 0.1rem; }

.stButton > button { font-weight: 600; }
</style>
""", unsafe_allow_html=True)


# ============================================================
# CHARGEMENT (cache)
# ============================================================

@st.cache_resource
def load_model_and_holistic():
    model = tf.keras.models.load_model(MODEL_DIR / MODEL_NAME)
    with open(MODEL_DIR / METADATA_NAME, "r") as f:
        metadata = json.load(f)
    idx_to_sign = {int(k): v for k, v in metadata["idx_to_sign"].items()}

    mp_holistic = mp.solutions.holistic
    holistic = mp_holistic.Holistic(
        static_image_mode=False,
        model_complexity=1,
        refine_face_landmarks=True,
        min_detection_confidence=0.4,
        min_tracking_confidence=0.4,
    )
    mp_drawing = mp.solutions.drawing_utils
    return model, idx_to_sign, holistic, mp_holistic, mp_drawing, metadata


# ============================================================
# FEATURES
# ============================================================

def landmarks_to_vector_no_pose(results) -> np.ndarray:
    if results.face_landmarks:
        face = np.array(
            [
                [results.face_landmarks.landmark[i].x,
                 results.face_landmarks.landmark[i].y,
                 results.face_landmarks.landmark[i].z]
                for i in FACE_LANDMARKS_SELECTED
            ], dtype=np.float32,
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


def draw_landmarks(frame, results, mp_holistic, mp_drawing):
    if results.face_landmarks:
        mp_drawing.draw_landmarks(frame, results.face_landmarks, mp_holistic.FACEMESH_CONTOURS)
    if results.left_hand_landmarks:
        mp_drawing.draw_landmarks(frame, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
    if results.right_hand_landmarks:
        mp_drawing.draw_landmarks(frame, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
    return frame


# ============================================================
# HELPERS d'affichage
# ============================================================

def render_detection_card(instant_label, instant_conf, vote_label, vote_ratio, cooldown):
    """Carte 'Détection' : instant + vote + cooldown."""
    return (
        '<div class="sensi-card">'
        '<div class="sensi-eyebrow">Détection en cours</div>'
        f'<div class="sensi-big">{instant_label}</div>'
        '<div class="sensi-meta">'
        f'<span><b>{instant_conf:.0%}</b> instant</span>'
        f'<span><b>{vote_label}</b> · {vote_ratio:.0%} vote</span>'
        f'<span>cooldown <b>{cooldown}</b></span>'
        '</div>'
        '</div>'
    )


def render_top3_card(top_3):
    """Carte 'Top 3' avec barres."""
    if not top_3:
        rows_html = '<div class="seq-empty">en attente…</div>'
    else:
        max_p = max((p for _, p in top_3), default=1.0) or 1.0
        rows = []
        for i, (name, p) in enumerate(top_3):
            width = int((p / max_p) * 100)
            is_top = (i == 0)
            bar_class = "t3-bar" if is_top else "t3-bar dim"
            name_class = "t3-name" if is_top else "t3-name dim"
            rows.append(
                '<div class="t3-row">'
                f'<div class="{name_class}">{name}</div>'
                f'<div class="t3-bar-bg"><div class="{bar_class}" style="width:{width}%"></div></div>'
                f'<div class="t3-pct">{p:.0%}</div>'
                '</div>'
            )
        rows_html = "".join(rows)
    return (
        '<div class="sensi-card">'
        '<div class="sensi-eyebrow">Top 3 prédictions</div>'
        f'{rows_html}'
        '</div>'
    )


def render_sequence(seq):
    """Séquence en pills (le dernier signe est mis en valeur)."""
    if not seq:
        inner = '<div class="seq-empty">aucun signe — fais un signe face à la caméra</div>'
    else:
        parts = []
        for i, s in enumerate(seq):
            last = " last" if i == len(seq) - 1 else ""
            parts.append(f'<span class="seq-pill{last}">{s}</span>')
            if i < len(seq) - 1:
                parts.append('<span class="seq-arr">→</span>')
        inner = '<div class="seq-wrap">' + "".join(parts) + '</div>'
    plural = 's' if len(seq) > 1 else ''
    return (
        '<div class="sensi-card">'
        f'<div class="sensi-eyebrow">Séquence en cours · {len(seq)} signe{plural}</div>'
        f'{inner}'
        '</div>'
    )


# ============================================================
# SESSION STATE
# ============================================================

if "camera_active" not in st.session_state:
    st.session_state.camera_active = False
if "detected_sequence" not in st.session_state:
    st.session_state.detected_sequence = []


# ============================================================
# HEADER + BOUTONS (ligne du haut, zéro scroll)
# ============================================================

head_col, btn_col = st.columns([2, 3], gap="medium")

with head_col:
    st.markdown(
        '<div class="sensi-title">🎥 <span class="accent">SignLive</span></div>'
        '<div class="sensi-subtitle">Reconnaissance LSF temps réel · LSTM v6</div>',
        unsafe_allow_html=True,
    )

with btn_col:
    b1, b2, b3, b4 = st.columns(4, gap="small")
    with b1:
        if st.button("▶ Démarrer", type="primary",
                     disabled=st.session_state.camera_active,
                     use_container_width=True, key="start_cam"):
            st.session_state.camera_active = True
            st.rerun()
    with b2:
        if st.button("⏸ Arrêter",
                     disabled=not st.session_state.camera_active,
                     use_container_width=True, key="stop_cam"):
            st.session_state.camera_active = False
            st.rerun()
    with b3:
        if st.button("🔄 Reset", use_container_width=True, key="reset_seq"):
            st.session_state.detected_sequence = []
            st.rerun()
    with b4:
        translate_clicked = st.button(
            "🔊 Traduire",
            disabled=not st.session_state.detected_sequence,
            use_container_width=True, key="translate",
        )


# ============================================================
# CHARGEMENT MODÈLE (spinner discret en haut)
# ============================================================

with st.spinner("Chargement du modèle Sensi v6…"):
    model, idx_to_sign, holistic, mp_holistic, mp_drawing, metadata = load_model_and_holistic()


# ============================================================
# SIDEBAR — paramètres avancés (masqués par défaut)
# ============================================================

with st.sidebar:
    st.header("⚙️ Paramètres")
    st.caption("Réglages avancés pour le tuning live. La sidebar est masquée par défaut en démo.")

    confidence_threshold = st.slider("Seuil de confiance modèle", 0.10, 0.95, 0.50, 0.05)
    stability_frames = st.slider("Fenêtre de vote (frames)", 3, 25, 10, 1)
    vote_ratio_threshold = st.slider("Ratio vote majoritaire", 0.50, 1.0, 0.70, 0.05)
    cooldown_frames = st.slider("Cooldown après validation (frames)", 5, 60, 25, 5)
    predict_every_n_frames = st.slider("Prédire toutes les N frames", 1, 15, 4, 1)
    skip_inconnu = st.checkbox("Ignorer la classe 'inconnu'", value=True)

    st.divider()
    st.caption(
        f"📊 Modèle : `{MODEL_NAME}`  \n"
        f"{len(idx_to_sign)} signes · "
        f"val acc {metadata.get('best_val_accuracy', 0):.0%}"
    )


# ============================================================
# RENDU AUDIO (quand on clique Traduire)
# ============================================================

if translate_clicked:
    with st.spinner("Traduction & synthèse vocale…"):
        try:
            response = requests.post(
                f"{API_URL}/predict/sentence/audio",
                json={"glosses": st.session_state.detected_sequence},
                timeout=60,
            )
            if response.status_code == 200:
                phrase = response.headers.get("x-phrase", "")
                st.success(f"**{phrase}**")
                st.audio(response.content, format="audio/mpeg", autoplay=True)
            else:
                st.error(f"Erreur API : {response.status_code}")
        except requests.exceptions.ConnectionError:
            st.error("API injoignable. Vérifie qu'uvicorn tourne sur :8000.")
        except requests.exceptions.Timeout:
            st.error("L'API met trop de temps à répondre.")


# ============================================================
# LAYOUT PRINCIPAL — webcam à gauche, infos à droite
# ============================================================

col_video, col_infos = st.columns([2, 1], gap="medium")

with col_video:
    frame_placeholder = st.empty()

with col_infos:
    detection_placeholder = st.empty()
    top3_placeholder = st.empty()

# Séquence en bas, pleine largeur
sequence_placeholder = st.empty()

# Affichage initial (avant démarrage)
detection_placeholder.markdown(
    render_detection_card("—", 0.0, "—", 0.0, 0), unsafe_allow_html=True
)
top3_placeholder.markdown(render_top3_card([]), unsafe_allow_html=True)
sequence_placeholder.markdown(
    render_sequence(st.session_state.detected_sequence), unsafe_allow_html=True
)


# ============================================================
# BOUCLE WEBCAM
# ============================================================

if st.session_state.camera_active:
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        st.error("❌ Impossible d'ouvrir la webcam. Vérifie les permissions macOS.")
        st.session_state.camera_active = False
    else:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        cap.set(cv2.CAP_PROP_FPS, 30)

        sequence_buf = deque(maxlen=TARGET_FRAMES)
        stability_history = deque(maxlen=25)
        last_added_sign = None
        cooldown_remaining = 0
        current_pred_idx = -1
        current_pred_conf = 0.0
        current_top_3 = []
        frame_count = 0
        current_vote_ratio = 0.0
        current_vote_label = "?"

        try:
            while st.session_state.camera_active:
                ret, frame = cap.read()
                if not ret:
                    continue

                frame = cv2.flip(frame, 1)
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = holistic.process(rgb)

                vector = landmarks_to_vector_no_pose(results)
                sequence_buf.append(vector)

                if cooldown_remaining > 0:
                    cooldown_remaining -= 1

                # === Prédiction LSTM ===
                if (
                    len(sequence_buf) == TARGET_FRAMES
                    and frame_count % predict_every_n_frames == 0
                ):
                    sequence = np.array(sequence_buf, dtype=np.float32)
                    batch = sequence[np.newaxis, ...]
                    probs = model.predict(batch, verbose=0)[0]

                    top_3_idx = np.argsort(probs)[::-1][:3]
                    current_pred_idx = int(top_3_idx[0])
                    current_pred_conf = float(probs[current_pred_idx])
                    current_top_3 = [
                        (idx_to_sign[int(i)], float(probs[i])) for i in top_3_idx
                    ]

                    # === Vote majoritaire ===
                    if current_pred_conf >= confidence_threshold:
                        stability_history.append(current_pred_idx)
                    else:
                        stability_history.append(-1)

                    recent = list(stability_history)[-stability_frames:]
                    if len(recent) == stability_frames:
                        valid_candidates = [idx for idx in recent if idx != -1]
                        if valid_candidates:
                            most_common_idx = max(
                                set(valid_candidates), key=valid_candidates.count
                            )
                            current_vote_ratio = (
                                recent.count(most_common_idx) / stability_frames
                            )
                            current_vote_label = idx_to_sign[most_common_idx]

                            if (
                                current_vote_ratio >= vote_ratio_threshold
                                and cooldown_remaining == 0
                            ):
                                validated_sign = current_vote_label
                                if skip_inconnu and validated_sign == "inconnu":
                                    stability_history.clear()
                                    cooldown_remaining = 15
                                elif validated_sign != last_added_sign:
                                    st.session_state.detected_sequence.append(validated_sign)
                                    last_added_sign = validated_sign
                                    cooldown_remaining = cooldown_frames
                                    stability_history.clear()
                        else:
                            current_vote_ratio = 0.0
                            current_vote_label = "?"

                # === Affichage frame webcam ===
                annotated_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                annotated_rgb = draw_landmarks(annotated_rgb, results, mp_holistic, mp_drawing)
                frame_placeholder.image(annotated_rgb, channels="RGB", use_column_width=True)

                # === Mise à jour des cartes infos ===
                instant_label = (
                    idx_to_sign[current_pred_idx] if current_pred_idx >= 0 else "—"
                )
                detection_placeholder.markdown(
                    render_detection_card(
                        instant_label, current_pred_conf,
                        current_vote_label, current_vote_ratio,
                        cooldown_remaining,
                    ),
                    unsafe_allow_html=True,
                )
                top3_placeholder.markdown(
                    render_top3_card(current_top_3), unsafe_allow_html=True
                )
                sequence_placeholder.markdown(
                    render_sequence(st.session_state.detected_sequence),
                    unsafe_allow_html=True,
                )

                frame_count += 1
        finally:
            cap.release()
else:
    frame_placeholder.info(
        "👆 Clique **▶ Démarrer** pour activer la webcam. "
        "Fais tes signes face à la caméra, puis **🔊 Traduire**."
    )


# ============================================================
# MODE TEST — Simulation manuelle (backup démo)
# ============================================================

with st.expander("🧪 Mode Test — Simulation manuelle (backup démo)"):
    st.caption(
        "Si la webcam déconne le jour J, utilise ce mode pour démontrer le "
        "pipeline NLP + TTS avec des glosses choisis à la main."
    )

    glosses_input = st.multiselect(
        label="Glosses (simulation)",
        options=[
            "AIDER", "AMELIORER", "AMI", "AUJOURD_HUI", "BONJOUR",
            "COMMUNIQUER", "ENTENDANTS", "CONTENT", "JE_SUIS", "JE_VEUX",
            "LANGUE_DES_SIGNES", "MERCI", "OUTIL_POINTAGE", "OUTIL",
            "PRESENTER", "PROJET", "SOURD_POINTAGE", "SOURD",
            "TRADUCTION", "VOCAL",
        ],
        default=["BONJOUR", "JE_SUIS", "CONTENT", "PRESENTER", "PROJET"],
    )

    if glosses_input:
        st.caption(f"Séquence : {' → '.join(glosses_input)}")

    if st.button("🎤 Traduire (simulation)", disabled=not glosses_input, key="test_translate"):
        with st.spinner("Traduction en cours…"):
            try:
                response = requests.post(
                    f"{API_URL}/predict/sentence/audio",
                    json={"glosses": glosses_input},
                    timeout=60,
                )
                if response.status_code == 200:
                    phrase = response.headers.get("x-phrase", "")
                    st.success(f"**{phrase}**")
                    st.audio(response.content, format="audio/mpeg", autoplay=True)
                else:
                    st.error(f"Erreur API : {response.status_code}")
            except requests.exceptions.ConnectionError:
                st.error("API injoignable.")
            except requests.exceptions.Timeout:
                st.error("L'API met trop de temps à répondre.")
