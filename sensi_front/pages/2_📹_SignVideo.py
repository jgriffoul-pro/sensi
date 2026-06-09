"""
SignVideo — Traduction LSF à partir d'une vidéo pré-enregistrée.

Layout demo day : sidebar masquée, vue split preview/infos, zéro scroll.
Lit la vidéo de démo configurée dans DEMO_VIDEO_PATH OU une vidéo uploadée.
Pipeline identique à SignLive.
"""

import json
import os
import tempfile
import time
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
    page_title="SignVideo — Sensi",
    page_icon="📹",
    layout="wide",
    initial_sidebar_state="collapsed",
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
MODEL_DIR = PROJECT_ROOT / "models"
MODEL_NAME = "sensi_team_v6.keras"
METADATA_NAME = "sensi_team_v6_metadata.json"

DEMO_VIDEO_PATH = Path(
    "/Users/jean-christophebertincourt/Desktop/videos_soeur/phrase_complete.mov"
)

TARGET_FRAMES = 60
FACE_LANDMARKS_SELECTED = [13, 14, 61, 291, 159, 386, 70, 300]


# ============================================================
# CSS — identique à SignLive pour cohérence visuelle
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
# CHARGEMENT MODÈLE
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
# FEATURES + ROTATION
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


ROTATION_MAP = {
    "0° (aucune)": 0,
    "90° horaire": 90,
    "180°": 180,
    "90° anti-horaire": 270,
}


def apply_rotation(frame, deg):
    if deg == 90:
        return cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
    elif deg == 180:
        return cv2.rotate(frame, cv2.ROTATE_180)
    elif deg == 270:
        return cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
    return frame


# ============================================================
# HELPERS d'affichage (identiques à SignLive)
# ============================================================

def render_detection_card(instant_label, instant_conf, cooldown):
    return (
        '<div class="sensi-card">'
        '<div class="sensi-eyebrow">Détection en cours</div>'
        f'<div class="sensi-big">{instant_label}</div>'
        '<div class="sensi-meta">'
        f'<span><b>{instant_conf:.0%}</b> instant</span>'
        f'<span>cooldown <b>{cooldown}</b></span>'
        '</div>'
        '</div>'
    )


def render_top3_card(top_3):
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
    if not seq:
        inner = '<div class="seq-empty">aucun signe — lance l\'analyse</div>'
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
        f'<div class="sensi-eyebrow">Séquence détectée · {len(seq)} signe{plural}</div>'
        f'{inner}'
        '</div>'
    )


def render_video_meta_card(total_frames, fps_video, duration, w, h, det_parts=None):
    det_html = ''
    if det_parts:
        det_html = (
            '<div class="sensi-meta" style="margin-top:0.7rem">'
            f'<span>{" · ".join(det_parts)}</span>'
            '</div>'
        )
    return (
        '<div class="sensi-card">'
        '<div class="sensi-eyebrow">Vidéo</div>'
        '<div class="sensi-meta" style="margin-top:0.2rem">'
        f'<span><b>{total_frames}</b> frames</span>'
        f'<span><b>{fps_video:.0f}</b> fps</span>'
        f'<span><b>{duration:.1f}s</b></span>'
        f'<span><b>{w}×{h}</b></span>'
        '</div>'
        f'{det_html}'
        '</div>'
    )


# ============================================================
# SESSION STATE
# ============================================================

defaults = {
    "video_detected_sequence": [],
    "video_analysis_done": False,
    "video_report": [],
    "video_path": None,
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v


# ============================================================
# HEADER + BOUTONS (ligne du haut)
# ============================================================

head_col, btn_col = st.columns([2, 3], gap="medium")

with head_col:
    st.markdown(
        '<div class="sensi-title">📹 <span class="accent">SignVideo</span></div>'
        '<div class="sensi-subtitle">Traduction LSF à partir d\'une vidéo · LSTM v6</div>',
        unsafe_allow_html=True,
    )

with btn_col:
    b1, b2, b3, b4 = st.columns(4, gap="small")
    has_video = st.session_state.video_path is not None
    has_seq = bool(st.session_state.video_detected_sequence)

    with b1:
        if st.button("📂 Démo", type="primary",
                     use_container_width=True, key="load_demo_btn"):
            if DEMO_VIDEO_PATH.exists():
                st.session_state.video_path = str(DEMO_VIDEO_PATH)
                st.session_state.video_analysis_done = False
                st.session_state.video_detected_sequence = []
                st.session_state.video_report = []
                st.rerun()
            else:
                st.error("Fichier démo introuvable")
    with b2:
        start_analysis = st.button("▶ Analyser", disabled=not has_video,
                                   use_container_width=True, key="start_analysis")
    with b3:
        if st.button("🔄 Reset", use_container_width=True, key="reset_analysis"):
            st.session_state.video_detected_sequence = []
            st.session_state.video_analysis_done = False
            st.session_state.video_report = []
            st.rerun()
    with b4:
        translate_clicked = st.button(
            "🔊 Traduire", disabled=not has_seq,
            use_container_width=True, key="translate_video",
        )


# ============================================================
# CHARGEMENT MODÈLE
# ============================================================

with st.spinner("Chargement du modèle Sensi v6…"):
    model, idx_to_sign, holistic, mp_holistic, mp_drawing, metadata = load_model_and_holistic()


# ============================================================
# SIDEBAR — paramètres avancés
# ============================================================

with st.sidebar:
    st.header("🎬 Affichage")
    rotation_label = st.selectbox(
        "Rotation vidéo",
        list(ROTATION_MAP.keys()),
        index=2,
        help="Corrige l'orientation (typique .MOV iPhone).",
    )
    rotation_deg = ROTATION_MAP[rotation_label]

    playback_realtime = st.checkbox(
        "Lecture vitesse réelle",
        value=True,
        help="Décocher pour traiter aussi vite que possible (debug).",
    )

    st.divider()
    st.header("⚙️ Détection")
    confidence_threshold = st.slider("Seuil de confiance modèle", 0.10, 0.95, 0.50, 0.05)
    stability_frames = st.slider("Frames stables pour valider", 2, 15, 4, 1)
    cooldown_frames = st.slider("Cooldown après validation (frames)", 5, 60, 20, 5)
    predict_every_n_frames = st.slider("Prédire toutes les N frames", 1, 15, 3, 1)
    skip_inconnu = st.checkbox("Ignorer 'inconnu'", value=True)

    st.divider()
    with st.expander("📁 Uploader une autre vidéo"):
        uploaded_file = st.file_uploader(
            "Choisis une vidéo (.mp4, .mov, .webm, .avi)",
            type=["mp4", "mov", "webm", "avi"],
        )
        if uploaded_file is not None:
            suffix = Path(uploaded_file.name).suffix
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp.write(uploaded_file.read())
                uploaded_path = tmp.name
            if st.button("Utiliser cette vidéo", type="primary",
                         use_container_width=True, key="load_upload"):
                st.session_state.video_path = uploaded_path
                st.session_state.video_analysis_done = False
                st.session_state.video_detected_sequence = []
                st.session_state.video_report = []
                st.rerun()

    st.divider()
    st.caption(
        f"📊 Modèle : `{MODEL_NAME}`  \n"
        f"{len(idx_to_sign)} signes · "
        f"val acc {metadata.get('best_val_accuracy', 0):.0%}"
    )


# ============================================================
# RENDU AUDIO (clic Traduire)
# ============================================================

if translate_clicked:
    with st.spinner("Traduction & synthèse vocale…"):
        try:
            response = requests.post(
                f"{API_URL}/predict/sentence/audio",
                json={"glosses": st.session_state.video_detected_sequence},
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
# LAYOUT PRINCIPAL — preview à gauche, infos à droite
# ============================================================

col_video, col_infos = st.columns([2, 1], gap="medium")
with col_video:
    frame_placeholder = st.empty()
    progress_placeholder = st.empty()
with col_infos:
    meta_placeholder = st.empty()
    detection_placeholder = st.empty()
    top3_placeholder = st.empty()

# Séquence + rapport en bas
sequence_placeholder = st.empty()
report_placeholder = st.empty()

# Affichage initial
detection_placeholder.markdown(
    render_detection_card("—", 0.0, 0), unsafe_allow_html=True
)
top3_placeholder.markdown(render_top3_card([]), unsafe_allow_html=True)
sequence_placeholder.markdown(
    render_sequence(st.session_state.video_detected_sequence),
    unsafe_allow_html=True,
)


# ============================================================
# PREVIEW VIDÉO + INFOS
# ============================================================

if st.session_state.video_path is not None:
    video_path = st.session_state.video_path

    cap_info = cv2.VideoCapture(video_path)
    if not cap_info.isOpened():
        frame_placeholder.error(f"❌ Impossible d'ouvrir : `{video_path}`")
        st.stop()

    total_frames = int(cap_info.get(cv2.CAP_PROP_FRAME_COUNT))
    fps_video = cap_info.get(cv2.CAP_PROP_FPS) or 30.0
    width_video = int(cap_info.get(cv2.CAP_PROP_FRAME_WIDTH))
    height_video = int(cap_info.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration = total_frames / fps_video if fps_video > 0 else 0

    ret_preview, first_frame = cap_info.read()
    cap_info.release()

    if ret_preview:
        first_frame_rot = apply_rotation(first_frame, rotation_deg)
        preview_rgb = cv2.cvtColor(first_frame_rot, cv2.COLOR_BGR2RGB)
        frame_placeholder.image(preview_rgb, use_column_width=True,
                                caption="Aperçu — change la rotation dans la sidebar si besoin")

    meta_placeholder.markdown(
        render_video_meta_card(total_frames, fps_video, duration, width_video, height_video),
        unsafe_allow_html=True,
    )


# ============================================================
# ANALYSE VIDÉO
# ============================================================

if 'start_analysis' in locals() and start_analysis and st.session_state.video_path:
    video_path = st.session_state.video_path
    st.session_state.video_detected_sequence = []
    st.session_state.video_analysis_done = False
    st.session_state.video_report = []

    cap_info = cv2.VideoCapture(video_path)
    total_frames = int(cap_info.get(cv2.CAP_PROP_FRAME_COUNT))
    fps_video = cap_info.get(cv2.CAP_PROP_FPS) or 30.0
    width_video = int(cap_info.get(cv2.CAP_PROP_FRAME_WIDTH))
    height_video = int(cap_info.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration = total_frames / fps_video if fps_video > 0 else 0
    cap_info.release()

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        frame_placeholder.error("❌ Impossible d'ouvrir la vidéo pour l'analyse.")
    else:
        sequence_buf = deque(maxlen=TARGET_FRAMES)
        stability_history = deque(maxlen=stability_frames)
        last_added_sign = None
        cooldown_remaining = 0
        current_pred_idx = -1
        current_pred_conf = 0.0
        current_top_3 = []
        frame_count = 0
        frames_with_hands = 0
        frames_with_face = 0
        target_frame_time = 1.0 / fps_video if fps_video > 0 else 1.0 / 30

        progress_bar = progress_placeholder.progress(0.0, text="Analyse en cours…")

        try:
            while True:
                frame_start = time.time()
                ret, frame = cap.read()
                if not ret:
                    break

                frame = apply_rotation(frame, rotation_deg)
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = holistic.process(rgb)

                has_face = results.face_landmarks is not None
                has_left = results.left_hand_landmarks is not None
                has_right = results.right_hand_landmarks is not None
                if has_left or has_right:
                    frames_with_hands += 1
                if has_face:
                    frames_with_face += 1

                vector = landmarks_to_vector_no_pose(results)
                sequence_buf.append(vector)

                if cooldown_remaining > 0:
                    cooldown_remaining -= 1

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

                    if current_pred_conf >= confidence_threshold:
                        stability_history.append(current_pred_idx)
                    else:
                        stability_history.clear()

                    if (
                        len(stability_history) == stability_frames
                        and len(set(stability_history)) == 1
                        and cooldown_remaining == 0
                    ):
                        validated_sign = idx_to_sign[stability_history[0]]
                        timestamp = frame_count / fps_video if fps_video > 0 else 0

                        if skip_inconnu and validated_sign == "inconnu":
                            stability_history.clear()
                            cooldown_remaining = 10
                        elif validated_sign != last_added_sign:
                            st.session_state.video_detected_sequence.append(validated_sign)
                            st.session_state.video_report.append({
                                "sign": validated_sign,
                                "conf": current_pred_conf,
                                "frame": frame_count,
                                "timestamp": timestamp,
                            })
                            last_added_sign = validated_sign
                            cooldown_remaining = cooldown_frames
                            stability_history.clear()

                # Affichage frame
                annotated_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                annotated_rgb = draw_landmarks(annotated_rgb, results, mp_holistic, mp_drawing)
                frame_placeholder.image(annotated_rgb, channels="RGB", use_column_width=True)

                # Progress
                progress = frame_count / total_frames if total_frames > 0 else 0
                progress_bar.progress(
                    min(progress, 1.0),
                    text=f"{frame_count}/{total_frames} frames ({progress*100:.0f}%)",
                )

                # Cartes infos
                instant_label = idx_to_sign[current_pred_idx] if current_pred_idx >= 0 else "—"

                det_parts = []
                det_parts.append("😀 visage" if has_face else "❌ visage")
                if has_left or has_right:
                    sides = []
                    if has_left: sides.append("G")
                    if has_right: sides.append("D")
                    det_parts.append(f"✋ mains [{'+'.join(sides)}]")
                else:
                    det_parts.append("❌ mains")

                meta_placeholder.markdown(
                    render_video_meta_card(total_frames, fps_video, duration,
                                           width_video, height_video, det_parts),
                    unsafe_allow_html=True,
                )
                detection_placeholder.markdown(
                    render_detection_card(instant_label, current_pred_conf, cooldown_remaining),
                    unsafe_allow_html=True,
                )
                top3_placeholder.markdown(
                    render_top3_card(current_top_3), unsafe_allow_html=True
                )
                sequence_placeholder.markdown(
                    render_sequence(st.session_state.video_detected_sequence),
                    unsafe_allow_html=True,
                )

                if playback_realtime:
                    elapsed = time.time() - frame_start
                    sleep_time = target_frame_time - elapsed
                    if sleep_time > 0:
                        time.sleep(sleep_time)

                frame_count += 1

        finally:
            cap.release()
            progress_placeholder.empty()
            st.session_state.video_analysis_done = True

        # Diagnostic final
        hands_rate = frames_with_hands / frame_count if frame_count > 0 else 0
        face_rate = frames_with_face / frame_count if frame_count > 0 else 0
        if hands_rate < 0.3:
            st.warning(
                f"⚠️ Mains détectées sur seulement {hands_rate:.0%} des frames. "
                "Probable problème de **rotation** (sidebar)."
            )
        else:
            st.success(
                f"✅ Analyse terminée — **{len(st.session_state.video_detected_sequence)} signes** "
                f"(mains {hands_rate:.0%} · visage {face_rate:.0%})"
            )


# ============================================================
# RAPPORT (en bas, dans un expander pour économiser l'écran)
# ============================================================

if st.session_state.video_analysis_done and st.session_state.video_report:
    with st.expander(f"📋 Chronologie ({len(st.session_state.video_report)} signes détectés)"):
        report_rows = [
            {
                "#": i + 1,
                "Timestamp": f"{r['timestamp']:.1f}s",
                "Signe": r["sign"],
                "Confiance": f"{r['conf']:.0%}",
                "Frame": r["frame"],
            }
            for i, r in enumerate(st.session_state.video_report)
        ]
        st.table(report_rows)


# Message initial si rien chargé
if st.session_state.video_path is None:
    frame_placeholder.info(
        "👆 Clique **📂 Démo** pour charger la vidéo de démo, "
        "ou utilise l'uploader dans la sidebar."
    )
