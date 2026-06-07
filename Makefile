# ============================================================
# SENSI — Makefile
# Commandes pratiques pour le développement et la production
# Usage : make <commande>
# ============================================================

.PHONY: help install run dev test train-nlp train-lstm evaluate lint clean docker-build docker-run


# ============================================================
# AIDE
# ============================================================

help:
	@echo ""
	@echo "╔══════════════════════════════════════════════╗"
	@echo "║           SENSI — Commandes Make             ║"
	@echo "╠══════════════════════════════════════════════╣"
	@echo "║  SETUP                                       ║"
	@echo "║    make install        Installe les deps     ║"
	@echo "║                                              ║"
	@echo "║  DÉVELOPPEMENT                               ║"
	@echo "║    make run            Lance l'API           ║"
	@echo "║    make front          Lance Streamlit       ║"
	@echo "║    make lstm           Lance le script LSTM  ║"
	@echo "║                                              ║"
	@echo "║  TESTS                                       ║"
	@echo "║    make test           Lance tous les tests  ║"
	@echo "║    make test-nlp       Tests NLP             ║"
	@echo "║    make test-tts       Tests TTS             ║"
	@echo "║    make test-pipeline  Tests pipeline        ║"
	@echo "║                                              ║"
	@echo "║  ML                                          ║"
	@echo "║    make train-nlp      Entraîne BARThez      ║"
	@echo "║    make train-lstm     Entraîne le LSTM      ║"
	@echo "║    make evaluate       Évalue le NLP         ║"
	@echo "║                                              ║"
	@echo "║  DOCKER                                      ║"
	@echo "║    make docker-build   Build l'image         ║"
	@echo "║    make docker-run     Lance le conteneur    ║"
	@echo "║                                              ║"
	@echo "║    make clean          Nettoie les caches    ║"
	@echo "╚══════════════════════════════════════════════╝"
	@echo ""


# ============================================================
# SETUP
# ============================================================

install:
	pip install -r requirements.txt

install-front:
	cd sensi_front && pip install -r requirements.txt


# ============================================================
# DÉVELOPPEMENT
# ============================================================

run:
	uvicorn app.main:app --reload --port 8000

front:
	cd sensi_front && pyenv local sensi-front && streamlit run streamlit_app.py

lstm:
	python "scripts/test_team_live(llm).py"


# ============================================================
# TESTS
# ============================================================

test:
	pytest tests/ -v --tb=short

test-nlp:
	pytest tests/test_nlp.py -v

test-tts:
	pytest tests/test_tts.py -v

test-pipeline:
	pytest tests/test_pipeline.py -v

test-preprocessing:
	pytest tests/test_preprocessing.py -v


# ============================================================
# ML — ENTRAÎNEMENT
# ============================================================

train-nlp:
	python scripts/train_nlp.py

train-nlp-quick:
	python scripts/train_nlp.py --epochs 5 --batch-size 4

train-lstm:
	python scripts/train_lstm.py

evaluate:
	python scripts/evaluate_nlp.py --examples 20

evaluate-manual:
	python scripts/evaluate_nlp.py --manual


# ============================================================
# DOCKER
# ============================================================

docker-build:
	docker build -t sensi-api .

docker-run:
	docker run -p 8080:8080 sensi-api

docker-build-run: docker-build docker-run


# ============================================================
# NETTOYAGE
# ============================================================

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	@echo "Cache nettoyé ✅"
