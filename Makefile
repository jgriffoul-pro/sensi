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
