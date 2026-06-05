import streamlit as st
import requests
import os
from dotenv import load_dotenv

# Chargement des variables d'environnement
load_dotenv()
API_URL = os.getenv("API_URL", "http://localhost:8000/api/v1")

# ============================================================
# Configuration
# ============================================================

st.set_page_config(
    page_title="Sensi — Traduction LSF",
    page_icon="🤝",
    layout="centered"
)

# ============================================================
# Header
# ============================================================

st.title("🤝 Sensi")
st.subheader("Transcription oral de la Langue des Signes Française")
st.divider()

# ============================================================
# Simulation LSTM — à remplacer par la vraie webcam
# ============================================================

st.markdown("### Signes détectés par le modèle")

glosses_input = st.multiselect(
    label="Glosses détectées (simulation LSTM)",
    options=[
        "AIDER", "AMELIORER", "AMI", "AUJOURD_HUI", "BONJOUR",
        "COMMUNIQUER", "ENTENDANTS", "CONTENT", "JE_SUIS", "JE_VEUX",
        "LANGUE_DES_SIGNES", "MERCI", "OUTIL_POINTAGE", "OUTIL",
        "PRESENTER", "PROJET", "SOURD_POINTAGE", "SOURD",
        "TRADUCTION", "VOCAL"
    ],
    default=["BONJOUR", "JE_SUIS", "CONTENT", "PRESENTER", "PROJET"],
    help="En production, ces glosses seront détectées automatiquement par le modèle LSTM"
)

if glosses_input:
    st.caption(f"Séquence : {' → '.join(glosses_input)}")

st.divider()

# ============================================================
# Traduction
# ============================================================

st.markdown("### Traduction")

if st.button("🎤 Traduire en français", type="primary", disabled=not glosses_input):
    with st.spinner("Traduction en cours..."):
        try:
            response = requests.post(
                f"{API_URL}/predict/sentence/audio",
                json={"glosses": glosses_input},
                timeout=60
            )

            if response.status_code == 200:
                phrase = response.headers.get("x-phrase", "")
                st.success(f"**{phrase}**")
                st.audio(response.content, format="audio/mp3")

            else:
                st.error(f"Erreur API : {response.status_code}")

        except requests.exceptions.ConnectionError:
            st.error("Impossible de contacter l'API. Vérifie que uvicorn tourne sur le port 8000.")

        except requests.exceptions.Timeout:
            st.error("L'API met trop de temps à répondre.")

st.divider()

# ============================================================
# Footer
# ============================================================

st.caption("Projet Sensi — Le Wagon Data Science Batch 2288 Bootcamp 2026")
