# ============================================================
# SENSI — Dockerfile
# Image de base Python 3.10 allégée
# Déploiement : GCP Cloud Run (port 8080)
# ============================================================

FROM python:3.10-slim

# Répertoire de travail dans le conteneur
WORKDIR /app

# Installer les dépendances système nécessaires à OpenCV et MediaPipe
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copier et installer les dépendances Python
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copier le code de l'application
COPY app/ ./app/
COPY config/ ./config/

# Copier les modèles
COPY models/ ./models/

# Créer le dossier output (nécessaire pour last_phrase.txt)
RUN mkdir -p output

# Port exposé par Cloud Run
EXPOSE 8080

# Lancement de l'API
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8080"]
