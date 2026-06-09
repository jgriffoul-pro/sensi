# ============================================================
# MODIF 003 — Dockerfile optimise et corrige
# Réf : modifs/003_dockerfile_optimise.txt
# ============================================================
# Changements :
#   - Copie uniquement barthez_sensi_final/ (pas les .keras)
#   - Crée le répertoire output/ + fichier last_phrase.txt vide
#   - Ajoute OUTPUT_DIR pour preprocessing.py (modif 002)
#   - Exclut tensorflow/keras/mediapipe (inutiles en API-only)
# ============================================================

FROM python:3.10.6-slim

WORKDIR /app

# Dependances systeme minimales
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Installation des dependances Python
# MODIF 007 — requirements-docker.txt (CPU-only, sans tensorflow/keras/mediapipe)
# Réf : modifs/007_requirements_docker.txt
COPY requirements-docker.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Code applicatif
COPY app/ ./app/
COPY config/ ./config/

# Modele NLP BARThez uniquement (pas les modeles LSTM .keras)
COPY models/barthez_sensi_final/ ./models/barthez_sensi_final/

# MODIF 002 — Création du repertoire output/ pour preprocessing.py
RUN mkdir -p /app/output && \
    touch /app/output/last_phrase.txt

# Variables d'environnement
ENV MODEL_BASE_DIR=/app/models
ENV OUTPUT_DIR=/app/output

EXPOSE 8080

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8080"]
