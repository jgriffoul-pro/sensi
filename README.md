# Sensi — LSF to Speech Translation

> Translate French Sign Language (LSF) into spoken French in real time.
> Webcam → MediaPipe → BiLSTM → BARThez → gTTS

**Le Wagon Data Science Bootcamp 2026 — Final Project**

---

## Overview

Sensi is an end-to-end pipeline that recognizes French Sign Language gestures from a webcam feed and converts them into spoken French sentences. It combines computer vision, deep learning, NLP, and text-to-speech into a unified system deployable on Google Cloud Run.

---

## Pipeline

```
Webcam (cv2)
    ↓
MediaPipe Holistic — 150 features (face + hands, no pose)
    ↓
BiLSTM v6 — 21 signs, 96.2% val accuracy
    ↓
["BONJOUR", "JE_SUIS", "CONTENT", "PRESENTER", "PROJET"]
    ↓
BARThez (fine-tuned) — glosses → French sentence
    ↓
"Bonjour, je suis content de vous présenter ce projet."
    ↓
gTTS — MP3 audio
    ↓
Streamlit Frontend — plays audio in browser
```

---

## Project Structure

```
sensi/
├── app/                        FastAPI backend
│   ├── main.py                 API entry point
│   ├── schemas.py              Pydantic models
│   ├── routers/
│   │   └── predict.py          HTTP routes
│   └── services/
│       ├── nlp.py              BARThez inference
│       ├── tts.py              gTTS text-to-speech
│       ├── pipeline.py         NLP + TTS orchestrator
│       ├── model.py            LSTM gloss reader
│       └── preprocessing.py    Sequence file reader
│
├── sensi_front/                Streamlit frontend
│   ├── streamlit_app.py
│   ├── requirements.txt
│   └── .env
│
├── models/
│   ├── sensi_team_v6.keras     BiLSTM production model
│   ├── sensi_team_v6_metadata.json
│   └── barthez_sensi_final/    Fine-tuned BARThez
│
├── scripts/
│   ├── train_nlp.py            BARThez training script
│   ├── train_lstm.py           LSTM training script
│   ├── evaluate_nlp.py         NLP evaluation script
│   └── test_team_live(llm).py  Live LSTM demo
│
├── notebooks/
│   ├── lstm_best_version.ipynb BiLSTM v6 training
│   ├── lstm_model.ipynb        LSTM experiments
│   └── nlp_dataset_generation.ipynb
│
├── capture/                    Data acquisition scripts
│   ├── video_vectorizer.py     MP4 → .npy (MediaPipe)
│   └── prompteur/              Reference sign videos (.mp4)
│
├── utils/                      Shared utilities
│   ├── utils_mediapipe.py
│   ├── utils_sauvegarde.py
│   └── utils_source.py
│
├── tests/                      Unit tests
├── config/
│   └── config.yaml             Central configuration
├── data/nlp/                   NLP dataset (glosses → phrases)
├── output/                     Runtime files (sequence, last_phrase)
├── Dockerfile
├── Makefile
└── requirements.txt
```

---

## Quick Start

### Prerequisites

- Python 3.10.6
- pyenv + virtualenv
- macOS (MPS GPU) or Linux (CPU/CUDA)

### Installation

```bash
# Clone the repo
git clone https://github.com/jgriffoul-pro/sensi.git
cd sensi

# Create and activate virtualenv
pyenv virtualenv 3.10.6 sensi
pyenv local sensi
pip install -r requirements.txt

# Frontend (separate virtualenv — avoids starlette conflict)
pyenv virtualenv 3.10.6 sensi-front
cd sensi_front && pyenv local sensi-front
pip install -r requirements.txt
cd ..
```

### Running the project

```bash
# Terminal 1 — LSTM live detection
make lstm

# Terminal 2 — FastAPI backend
make run

# Terminal 3 — Streamlit frontend
make front
```

Or without Make:

```bash
python "scripts/test_team_live(llm).py"   # Terminal 1
uvicorn app.main:app --reload --port 8000  # Terminal 2
cd sensi_front && streamlit run streamlit_app.py  # Terminal 3
```

### Usage

1. Launch all three terminals above
2. Make signs in front of the webcam — glosses accumulate on screen
3. Press `V` to validate the sequence
4. Click **"Traduire la séquence détectée"** in Streamlit
5. The French sentence appears and audio plays automatically

---

## API

Base URL: `http://localhost:8000/api/v1`

| Method | Route | Input | Output |
|---|---|---|---|
| `GET` | `/` | — | Health check |
| `POST` | `/predict/sentence` | `{"glosses": [...]}` | `{"phrase": "..."}` |
| `POST` | `/predict/sentence/audio` | `{"glosses": [...]}` | MP3 + `X-Phrase` header |
| `POST` | `/predict/from-sequence` | — | MP3 (reads `output/last_phrase.txt`) |

Interactive docs available at `http://localhost:8000/docs`

---

## Models

### BiLSTM v6 — Sign Recognition

| Property | Value |
|---|---|
| File | `models/sensi_team_v6.keras` |
| Classes | 21 (20 signs + "inconnu") |
| Features | 150 (face + hands, no pose) |
| Frames | 60 per sequence |
| Val accuracy | **96.2%** |

### BARThez — Glosses to French

| Property | Value |
|---|---|
| Base model | `moussaKam/barthez` |
| Fine-tuned on | 678 synthetic gloss/phrase pairs |
| Architecture | mBART, 6 encoder/decoder layers, 768 dim |
| Generation | Beam search (4 beams), max 40 tokens |

---

## Vocabulary — 20 LSF signs

| | | | |
|---|---|---|---|
| AIDER | AMELIORER | AMI | AUJOURD_HUI |
| BONJOUR | COMMUNIQUER | CONTENT | ENTENDANTS |
| JE_SUIS | JE_VEUX | LANGUE_DES_SIGNES | MERCI |
| OUTIL | OUTIL_POINTAGE | PRESENTER | PROJET |
| SOURD | SOURD_POINTAGE | TRADUCTION | VOCAL |

---

## Development

### Make commands

```bash
make help           # List all commands
make install        # Install dependencies
make run            # Start API (port 8000)
make front          # Start Streamlit frontend
make lstm           # Start live LSTM demo
make test           # Run all unit tests
make train-nlp      # Train BARThez
make train-lstm     # Train LSTM
make evaluate       # Evaluate NLP model
make docker-build   # Build Docker image
make clean          # Remove __pycache__ and .pyc files
```

### Configuration

All parameters are centralized in `config/config.yaml` — model paths, generation hyperparameters, training settings, and LSTM thresholds. No hardcoded values in source files.

### Tests

```bash
make test                        # All tests
pytest tests/test_nlp.py -v      # NLP only
pytest tests/test_tts.py -v      # TTS only
pytest tests/test_pipeline.py -v # Pipeline only
```

### Training

```bash
# Retrain BARThez NLP model
python scripts/train_nlp.py --epochs 20

# Retrain LSTM (requires data/lstm/ with .npy sequences)
python scripts/train_lstm.py --epochs 50

# Evaluate NLP on dataset
python scripts/evaluate_nlp.py --examples 20
python scripts/evaluate_nlp.py --manual
```

---

## Deployment

### Docker

```bash
make docker-build
make docker-run
# API available at http://localhost:8080
```

### GCP Cloud Run

```bash
# Authenticate
gcloud auth configure-docker europe-west1-docker.pkg.dev

# Build and push
docker tag sensi-api europe-west1-docker.pkg.dev/PROJECT_ID/sensi-repo/sensi-api:v1
docker push europe-west1-docker.pkg.dev/PROJECT_ID/sensi-repo/sensi-api:v1

# Deploy
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

## Roadmap

- [ ] Expand vocabulary beyond 20 signs
- [ ] Build real NLP dataset from LSTM predictions
- [ ] Retrain BARThez on real data
- [ ] Integrate webcam directly into API (`/predict/live`, `/predict/video`)
- [ ] Deploy to GCP Cloud Run
- [ ] Add authentication for production API

---

## Team

| Member | Role |
|---|---|
| Jérôme | Project lead, data capture, model experiments |
| Vincent | Data capture, MediaPipe pipeline |
| JC | BiLSTM model, live demo script |
| Franck | API infrastructure, NLP (BARThez), TTS |

---

## Tech Stack

`Python 3.10` · `FastAPI` · `Streamlit` · `PyTorch` · `TensorFlow` · `HuggingFace Transformers` · `MediaPipe` · `gTTS` · `Docker` · `GCP Cloud Run`
