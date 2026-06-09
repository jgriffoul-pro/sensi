"""
Landing page Sensi — 3 modules accessibles depuis la sidebar
ou via les cartes ci-dessous.
"""

import streamlit as st


st.set_page_config(
    page_title="Sensi — Accessibilité",
    page_icon="🤟",
    layout="wide",
)


# ============================================================
# HEADER
# ============================================================

st.title("🤟 Sensi")
st.subheader("Plateforme d'accessibilité pour personnes sourdes et malentendantes")

st.markdown(
    """
Sensi propose trois modules pour rendre la communication accessible :
**traduction de la langue des signes** (en direct ou sur vidéo)
et **détection des sons d'alerte** du quotidien.
"""
)

st.divider()


# ============================================================
# 3 CARTES MODULES
# ============================================================

col1, col2, col3 = st.columns(3)

with col1:
    with st.container(border=True):
        st.markdown("### 🎥 SignLive")
        st.markdown(
            "**Traduction LSF en temps réel via webcam.**\n\n"
            "MediaPipe Holistic → LSTM bidirectionnel → synthèse vocale.\n\n"
            "Idéal pour une conversation en direct face caméra."
        )
        st.page_link(
            "pages/1_🎥_SignLive.py",
            label="Lancer SignLive",
            icon="🎥",
        )

with col2:
    with st.container(border=True):
        st.markdown("### 📹 SignVideo")
        st.markdown(
            "**Traduction LSF à partir d'une vidéo pré-enregistrée.**\n\n"
            "Vidéo de démo intégrée ou upload de ta propre vidéo.\n\n"
            "Idéal pour analyser un message ou pour la démonstration."
        )
        st.page_link(
            "pages/2_📹_SignVideo.py",
            label="Lancer SignVideo",
            icon="📹",
        )

with col3:
    with st.container(border=True):
        st.markdown("### 🔊 SoundWatch")
        st.markdown(
            "**Détection de sons d'alerte en temps réel.**\n\n"
            "YAMNet (transfer learning) + MLP → alertes visuelles.\n\n"
            "Sonnette, alarme, sirène, pleurs de bébé."
        )
        st.page_link(
            "pages/3_🔊_SoundWatch.py",
            label="Lancer SoundWatch",
            icon="🔊",
        )


st.divider()


# ============================================================
# FOOTER
# ============================================================

with st.expander("ℹ️ À propos du projet"):
    st.markdown(
        """
**Sensi** est le projet final du bootcamp Data Science Le Wagon (Batch 2288).

**Stack technique :**
- Détection des landmarks : MediaPipe Holistic (Google)
- Traduction LSF : LSTM bidirectionnel entraîné sur dataset auto-constitué (18 signes × 4 personnes × 10 prises)
- Détection sonore : YAMNet (AudioSet) + classifieur MLP sur ESC-50
- Backend : FastAPI · Frontend : Streamlit · Inference : TensorFlow
- LLM local (Ollama) pour la reformulation des séquences de signes en phrases naturelles

**Modèles entraînés sur Mac M3 Max** — pyenv Python 3.10.6.
"""
    )

st.caption("Sensi · Le Wagon Data Science Batch 2288 · 2026")
