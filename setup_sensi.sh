#!/bin/bash
# ============================================================
# Sensi — Script de setup multi-pages Streamlit
# À lancer depuis la racine du projet sensi/
# Usage : bash setup_sensi.sh
# ============================================================

set -e

# Vérif qu'on est au bon endroit
if [ ! -d "sensi_front" ] || [ ! -d "app" ]; then
    echo "❌ Erreur : ce script doit être lancé depuis la racine de ton projet sensi/"
    echo "   (le dossier qui contient sensi_front/ ET app/)"
    echo ""
    echo "   Tu es actuellement dans : $(pwd)"
    exit 1
fi

echo ""
echo "🤝 Setup Sensi multi-pages — démarrage"
echo "    Répertoire : $(pwd)"
echo ""

# ============================================================
# 1. BACKUP des fichiers existants
# ============================================================

cd sensi_front

if [ -f streamlit_app.py ]; then
    mv streamlit_app.py streamlit_app.py.bak
    echo "✅ Backup : streamlit_app.py → streamlit_app.py.bak"
fi

# ============================================================
# 2. Création du dossier pages/
# ============================================================

mkdir -p pages
echo "✅ Dossier pages/ prêt"

# ============================================================
# 3. Déplacer soundwatch_app.py (code du collègue) dans pages/
# ============================================================

if [ -f soundwatch_app.py ]; then
    mv soundwatch_app.py "pages/2_🔊_SoundWatch.py"
    echo "✅ soundwatch_app.py → pages/2_🔊_SoundWatch.py (code du collègue préservé)"
fi

# ============================================================
# 4. Création de streamlit_app.py (page d'accueil)
# ============================================================

cat > streamlit_app.py << 'PYEOF'
"""
Sensi — Plateforme d'accessibilité pour personnes sourdes et malentendantes.
Le Wagon Data Science — Batch 2288 — Bootcamp 2026.

Page d'accueil (entry point).
Lancer : streamlit run sensi_front/streamlit_app.py
"""

import streamlit as st

st.set_page_config(
    page_title="Sensi — Plateforme d'accessibilité",
    page_icon="🤝",
    layout="centered",
    initial_sidebar_state="expanded",
)

st.title("🤝 Sensi")
st.subheader("Plateforme d'accessibilité pour personnes sourdes et malentendantes")

st.divider()

col1, col2 = st.columns(2, gap="large")

with col1:
    st.markdown("### 🎥 SignLive")
    st.markdown(
        "Reconnaissance en temps réel de la Langue des Signes Française.  \n"
        "**Webcam → MediaPipe → LSTM → phrase → voix synthétique**"
    )
    st.page_link("pages/1_🎥_SignLive.py", label="Lancer SignLive", icon="🎥")

with col2:
    st.markdown("### 🔊 SoundWatch")
    st.markdown(
        "Détection en temps réel de sons d'alerte du quotidien.  \n"
        "**Micro → YAMNet → alerte visuelle**"
    )
    st.page_link("pages/2_🔊_SoundWatch.py", label="Lancer SoundWatch", icon="🔊")

st.divider()

st.markdown(
    """
    #### Le problème
    En France, **plus de 5 millions de personnes** sont sourdes ou malentendantes.
    L'accès à l'information sonore et la communication orale restent un défi quotidien.

    #### Notre réponse
    Sensi combine deux modules d'IA conçus pour s'intégrer naturellement dans
    la vie de tous les jours : un **traducteur LSF → voix** pour faire entendre
    sa voix, et un **détecteur de sons critiques** pour ne plus rater une sonnette,
    une alarme ou les pleurs d'un bébé.
    """
)

st.divider()

st.caption(
    "Projet final — Le Wagon Data Science Batch 2288 — Bootcamp 2026"
)
PYEOF
echo "✅ streamlit_app.py créé (page d'accueil)"

# ============================================================
# 5. Création de pages/1_🎥_SignLive.py
# ============================================================

cat > "pages/1_🎥_SignLive.py" << 'PYEOF'
"""
SignLive — Reconnaissance LSF en temps réel.

Mode Live : lit la séquence détectée par le script LSTM webcam.
Mode Test : simulation manuelle des glosses (dev/démo de secours).
"""

import streamlit as st
import requests
import os
from dotenv import load_dotenv

load_dotenv()
API_URL = os.getenv("API_URL", "http://localhost:8000/api/v1")

st.set_page_config(
    page_title="SignLive — Sensi",
    page_icon="🎥",
    layout="centered",
)

st.title("🎥 SignLive")
st.caption("Reconnaissance temps réel de la Langue des Signes Française")
st.divider()

# Mode Live — Lecture depuis output/sequence.txt
st.markdown("### Mode Live")
st.caption(
    "Lance `python scripts/test_team_live.py` dans un terminal séparé, "
    "fais tes signes devant la webcam, puis clique Traduire."
)

if st.button("🎤 Traduire la séquence détectée", type="primary", key="live"):
    with st.spinner("Lecture de la séquence et traduction..."):
        try:
            response = requests.post(
                f"{API_URL}/predict/from-sequence",
                timeout=60,
            )

            if response.status_code == 200:
                error = response.headers.get("x-error", "")
                if error:
                    st.warning(
                        "Aucune séquence détectée. "
                        "Lance le script LSTM et fais des signes."
                    )
                else:
                    phrase = response.headers.get("x-phrase", "")
                    glosses = response.headers.get("x-glosses", "")
                    st.caption(f"Séquence : {' → '.join(glosses.split())}")
                    st.success(f"**{phrase}**")
                    st.audio(response.content, format="audio/mpeg", autoplay=True)
            else:
                st.error(f"Erreur API : {response.status_code}")

        except requests.exceptions.ConnectionError:
            st.error(
                "Impossible de contacter l'API. "
                "Vérifie que uvicorn tourne sur le port 8000."
            )
        except requests.exceptions.Timeout:
            st.error("L'API met trop de temps à répondre.")

st.divider()

# Mode Test — Simulation manuelle (planqué dans un expander)
with st.expander("🧪 Mode Test — Simulation manuelle (dev only)"):
    st.caption(
        "Sélectionne des glosses manuellement pour tester le pipeline NLP + TTS "
        "sans webcam. Utile en démo de secours si la webcam déconne."
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

    if st.button("🎤 Traduire (simulation)", disabled=not glosses_input, key="test"):
        with st.spinner("Traduction en cours..."):
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
                st.error("Impossible de contacter l'API.")
            except requests.exceptions.Timeout:
                st.error("L'API met trop de temps à répondre.")
PYEOF
echo "✅ pages/1_🎥_SignLive.py créé (audio corrigé)"

# ============================================================
# 6. Création de pages/3_ℹ️_À_propos.py
# ============================================================

cat > "pages/3_ℹ️_À_propos.py" << 'PYEOF'
"""À propos — Équipe, projet, technologies."""

import streamlit as st

st.set_page_config(
    page_title="À propos — Sensi",
    page_icon="ℹ️",
    layout="centered",
)

st.title("ℹ️ À propos de Sensi")
st.divider()

st.markdown(
    """
    ### 🎯 Mission

    Rendre le quotidien plus accessible aux personnes sourdes et malentendantes
    grâce à deux modules d'IA complémentaires :

    - **SignLive** transcrit la Langue des Signes Française en voix synthétique,
      permettant à une personne signante de se faire comprendre par une personne
      entendante non-signante.
    - **SoundWatch** détecte les sons critiques du quotidien (sirène, alarme,
      pleurs de bébé, coup à la porte, applaudissements) et alerte visuellement
      l'utilisateur.
    """
)

st.divider()

st.markdown("### 🛠️ Stack technique")

col1, col2 = st.columns(2)

with col1:
    st.markdown(
        """
        **Computer Vision & Audio**
        - MediaPipe (landmarks main + pose)
        - YAMNet (transfer learning AudioSet)
        - OpenCV, sounddevice

        **Deep Learning**
        - LSTM Bidirectionnel (Keras/TensorFlow)
        - MLP custom + YAMNet embeddings
        - Transformers (HuggingFace)
        """
    )

with col2:
    st.markdown(
        """
        **Backend & Frontend**
        - FastAPI (API REST)
        - Streamlit (UI multi-pages)
        - gTTS (synthèse vocale)

        **Infra & Outils**
        - Python 3.10
        - pyenv + venv
        - Git / GitHub
        """
    )

st.divider()

st.markdown(
    """
    ### 📊 Données

    - **SignLive** : dataset custom auto-enregistré — 18 signes × 4 personnes
      × 10 prises = **720 échantillons** au format `(60, 282)`.
    - **SoundWatch** : **ESC-50** restreint à 5 classes cibles, filtré via les
      scores natifs YAMNet sur AudioSet pour écarter la parole, le silence et le bruit.
    """
)

st.divider()

st.markdown(
    """
    ### 👥 Équipe

    Projet final — **Le Wagon Data Science Batch 2288 — Bootcamp 2026**

    - Jean-Christophe Bertincourt
    - *[Prénom 2 à compléter]*
    - *[Prénom 3 à compléter]*
    - *[Prénom 4 à compléter]*
    """
)

st.divider()

st.caption(
    "Code source : *[lien GitHub à ajouter]*  \n"
    "Démo réalisée — Le Wagon Bordeaux"
)
PYEOF
echo "✅ pages/3_ℹ️_À_propos.py créé"

# ============================================================
# 7. Mise à jour du Makefile à la racine
# ============================================================

cd ..

cat > Makefile << 'MAKEEOF'
# ============================================================
# Sensi — Makefile
# Le Wagon Data Science Batch 2288
# ============================================================

.PHONY: help api front lstm demo install clean lint test

help:
	@echo ""
	@echo "🤝 Sensi — Cibles disponibles :"
	@echo ""
	@echo "  make api       — lance l'API FastAPI sur http://localhost:8000"
	@echo "  make front     — lance l'app Streamlit sur http://localhost:8501"
	@echo "  make lstm      — lance le script de capture webcam LSTM (interactif)"
	@echo "  make demo      — lance API + Streamlit en parallèle"
	@echo "                   (lance le LSTM séparément avec 'make lstm' pour la démo)"
	@echo ""
	@echo "  make install   — installe les dépendances Python"
	@echo "  make clean     — nettoie caches Python et fichiers temporaires"
	@echo ""

api:
	uvicorn app.main:app --reload --port 8000

front:
	streamlit run sensi_front/streamlit_app.py

lstm:
	python scripts/test_team_live.py

demo:
	@echo "🚀 Lancement de Sensi (API + Streamlit)..."
	@echo "   Lance 'make lstm' dans un autre terminal pour la webcam."
	@echo "   Ctrl+C ici stoppe API + Streamlit."
	@echo ""
	@trap 'kill 0' INT TERM EXIT; \
		uvicorn app.main:app --port 8000 & \
		sleep 3 && streamlit run sensi_front/streamlit_app.py & \
		wait

install:
	pip install -r requirements.txt

clean:
	@find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	@find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	@rm -f output/*.txt
	@echo "✨ Caches et fichiers temporaires nettoyés."
MAKEEOF
echo "✅ Makefile mis à jour à la racine"

# ============================================================
# Vérification finale
# ============================================================

echo ""
echo "📁 Arbo finale de sensi_front/ :"
ls -la sensi_front/ | grep -v "^total"
echo ""
echo "📁 Contenu de sensi_front/pages/ :"
ls -la sensi_front/pages/ | grep -v "^total"
echo ""
echo "🎉 Setup terminé avec succès."
echo ""
echo "🚀 Pour lancer : make demo"
echo "   Puis ouvre http://localhost:8501 dans ton navigateur."
echo ""
