# Sensi — Traduction de la Langue des Signes Française vers le français oral

> Webcam → MediaPipe → BiLSTM → BARThez → gTTS

**Projet final — Le Wagon Data Science Bootcamp 2026**

---

## Présentation

Sensi est un pipeline complet qui reconnaît les gestes de la Langue des Signes Française (LSF) depuis une webcam et les convertit en phrases françaises parlées. Il combine vision par ordinateur, deep learning, NLP et synthèse vocale dans un système deployable sur Google Cloud Run.

---

## Pipeline complet

```
Webcam (cv2)
    ↓
MediaPipe Holistic — 150 features (visage + mains, sans pose)
    ↓
BiLSTM v6 — 21 signes, 96.2% de précision
    ↓
["BONJOUR", "JE_SUIS", "CONTENT", "PRESENTER", "PROJET"]
    ↓
BARThez (fine-tuné) — glosses → phrase française
    ↓
"Bonjour, je suis content de vous présenter ce projet."
    ↓
gTTS — audio MP3
    ↓
Interface Streamlit — lecture audio dans le navigateur
```

---

## Structure du projet

```
sensi/
├── app/                        Backend FastAPI
│   ├── main.py                 Point d'entrée de l'API
│   ├── schemas.py              Modèles Pydantic
│   ├── routers/
│   │   └── predict.py          Routes HTTP
│   └── services/
│       ├── nlp.py              Inférence BARThez
│       ├── tts.py              Synthèse vocale gTTS
│       ├── pipeline.py         Orchestrateur NLP + TTS
│       ├── model.py            Lecteur de glosses LSTM
│       └── preprocessing.py    Lecteur de fichier séquence
│
├── sensi_front/                Frontend Streamlit
│   ├── streamlit_app.py
│   ├── requirements.txt
│   └── .env
│
├── models/
│   ├── sensi_team_v6.keras     Modèle BiLSTM production
│   ├── sensi_team_v6_metadata.json
│   └── barthez_sensi_final/    BARThez fine-tuné
│
├── scripts/
│   ├── train_nlp.py            Script d'entraînement BARThez
│   ├── train_lstm.py           Script d'entraînement LSTM
│   ├── evaluate_nlp.py         Script d'évaluation NLP
│   └── test_team_live(llm).py  Démo live LSTM
│
├── notebooks/
│   ├── lstm_best_version.ipynb Entraînement BiLSTM v6
│   ├── lstm_model.ipynb        Expérimentations LSTM
│   └── nlp_dataset_generation.ipynb
│
├── capture/                    Scripts d'acquisition de données
│   ├── video_vectorizer.py     MP4 → .npy (MediaPipe)
│   └── prompteur/              Vidéos de référence par signe (.mp4)
│
├── utils/                      Utilitaires partagés
│   ├── utils_mediapipe.py
│   ├── utils_sauvegarde.py
│   └── utils_source.py
│
├── tests/                      Tests unitaires
├── config/
│   └── config.yaml             Configuration centrale
├── data/nlp/                   Dataset NLP (glosses → phrases)
├── output/                     Fichiers générés en temps réel
├── Dockerfile
├── Makefile
└── requirements.txt
```

---

## Démarrage rapide

### Prérequis

- Python 3.10.6
- pyenv + virtualenv
- macOS (GPU MPS Apple Silicon) ou Linux (CPU/CUDA)

### Installation

```bash
# Cloner le repo
git clone https://github.com/jgriffoul-pro/sensi.git
cd sensi

# Créer et activer le virtualenv backend
pyenv virtualenv 3.10.6 sensi
pyenv local sensi
pip install -r requirements.txt

# Frontend (virtualenv séparé — évite le conflit starlette)
pyenv virtualenv 3.10.6 sensi-front
cd sensi_front && pyenv local sensi-front
pip install -r requirements.txt
cd ..
```

### Lancer le projet

```bash
# Terminal 1 — Détection LSTM live
make lstm

# Terminal 2 — Backend FastAPI
make run

# Terminal 3 — Frontend Streamlit
make front
```

Sans Make :

```bash
python "scripts/test_team_live(llm).py"    # Terminal 1
uvicorn app.main:app --reload --port 8000  # Terminal 2
cd sensi_front && streamlit run streamlit_app.py  # Terminal 3
```

### Utilisation

1. Lancer les 3 terminaux ci-dessus
2. Faire des signes devant la webcam — les glosses s'accumulent à l'écran
3. Appuyer sur `V` pour valider la séquence
4. Cliquer sur **"Traduire la séquence détectée"** dans Streamlit
5. La phrase française s'affiche et l'audio se lance automatiquement

---

## API

URL de base : `http://localhost:8000/api/v1`

| Méthode | Route | Entrée | Sortie |
|---|---|---|---|
| `GET` | `/` | — | Health check |
| `POST` | `/predict/sentence` | `{"glosses": [...]}` | `{"phrase": "..."}` |
| `POST` | `/predict/sentence/audio` | `{"glosses": [...]}` | MP3 + header `X-Phrase` |
| `POST` | `/predict/from-sequence` | — | MP3 (lit `output/last_phrase.txt`) |

Documentation interactive disponible sur `http://localhost:8000/docs`

---

## Modèles

### BiLSTM v6 — Reconnaissance des signes

| Propriété | Valeur |
|---|---|
| Fichier | `models/sensi_team_v6.keras` |
| Classes | 21 (20 signes + "inconnu") |
| Features | 150 (visage + mains, sans pose) |
| Frames | 60 par séquence |
| Précision validation | **96.2%** |

### BARThez — Glosses vers français

| Propriété | Valeur |
|---|---|
| Modèle de base | `moussaKam/barthez` |
| Fine-tuné sur | 678 paires glosses/phrases synthétiques |
| Architecture | mBART, 6 couches encodeur/décodeur, 768 dim |
| Génération | Beam search (4 beams), max 40 tokens |

---

## Vocabulaire — 20 signes LSF

| | | | |
|---|---|---|---|
| AIDER | AMELIORER | AMI | AUJOURD_HUI |
| BONJOUR | COMMUNIQUER | CONTENT | ENTENDANTS |
| JE_SUIS | JE_VEUX | LANGUE_DES_SIGNES | MERCI |
| OUTIL | OUTIL_POINTAGE | PRESENTER | PROJET |
| SOURD | SOURD_POINTAGE | TRADUCTION | VOCAL |

---

## Développement

### Commandes Make

```bash
make help           # Liste toutes les commandes
make install        # Installe les dépendances
make run            # Lance l'API (port 8000)
make front          # Lance le frontend Streamlit
make lstm           # Lance la démo LSTM live
make test           # Lance tous les tests unitaires
make train-nlp      # Entraîne BARThez
make train-lstm     # Entraîne le LSTM
make evaluate       # Évalue le modèle NLP
make docker-build   # Build l'image Docker
make clean          # Supprime __pycache__ et .pyc
```

### Configuration

Tous les paramètres sont centralisés dans `config/config.yaml` — chemins des modèles, hyperparamètres de génération, paramètres d'entraînement et seuils LSTM. Aucune valeur en dur dans le code source.

### Tests

```bash
make test                         # Tous les tests
pytest tests/test_nlp.py -v       # NLP uniquement
pytest tests/test_tts.py -v       # TTS uniquement
pytest tests/test_pipeline.py -v  # Pipeline uniquement
```

### Entraînement

```bash
# Réentraîner le modèle NLP BARThez
python scripts/train_nlp.py --epochs 20

# Réentraîner le LSTM (nécessite data/lstm/ avec séquences .npy)
python scripts/train_lstm.py --epochs 50

# Évaluer le NLP sur le dataset
python scripts/evaluate_nlp.py --examples 20
python scripts/evaluate_nlp.py --manual
```

---

## Déploiement

### Docker

```bash
make docker-build
make docker-run
# API disponible sur http://localhost:8080
```

### GCP Cloud Run

```bash
# Authentification
gcloud auth configure-docker europe-west1-docker.pkg.dev

# Build et push
docker tag sensi-api europe-west1-docker.pkg.dev/PROJECT_ID/sensi-repo/sensi-api:v1
docker push europe-west1-docker.pkg.dev/PROJECT_ID/sensi-repo/sensi-api:v1

# Déploiement
gcloud run deploy sensi-api \
  --image europe-west1-docker.pkg.dev/PROJECT_ID/sensi-repo/sensi-api:v1 \
  --platform managed \
  --region europe-west1 \
  --allow-unauthenticated \
  --memory 4Gi \
  --cpu 2 \
  --timeout 120
```

---

## Prochaines étapes

- [ ] Augmenter le vocabulaire au-delà de 20 signes
- [ ] Construire un vrai dataset NLP à partir des prédictions LSTM
- [ ] Réentraîner BARThez sur ce vrai dataset
- [ ] Intégrer la webcam directement dans l'API (`/predict/live`, `/predict/video`)
- [ ] Déployer sur GCP Cloud Run
- [ ] Ajouter une authentification pour l'API en production

---

## Équipe

| Membre | Rôle |
|---|---|
| Jérôme | Chef de projet, capture de données, expérimentations modèle |
| Vincent | Capture de données, pipeline MediaPipe |
| JC | Modèle BiLSTM, script démo live |
| Franck | Infrastructure API, NLP (BARThez), TTS |

---

## Stack technique

`Python 3.10` · `FastAPI` · `Streamlit` · `PyTorch` · `TensorFlow` · `HuggingFace Transformers` · `MediaPipe` · `gTTS` · `Docker` · `GCP Cloud Run`
