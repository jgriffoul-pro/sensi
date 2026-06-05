from pydantic import BaseModel


# ============================================================
# SCHEMAS ACTIFS — API santé
# ============================================================

class HealthResponse(BaseModel):
    """Réponse du endpoint racine — vérifie que l'API tourne."""
    status: str
    message: str


# ============================================================
# SCHEMAS ACTIFS — NLP + TTS
# ============================================================

class PredictRequest(BaseModel):
    """Entrée : liste de glosses prédites par le LSTM."""
    glosses: list[str]
    # ex: ["BONJOUR", "JE_SUIS", "CONTENT", "PRESENTER", "PROJET"]


class PredictResponse(BaseModel):
    """Sortie : phrase française générée par le NLP."""
    phrase: str
    # ex: "Bonjour, je suis content de vous présenter ce projet."


# ============================================================
# SCHEMAS EN ATTENTE — LSTM
# ============================================================

# class PredictionResponse(BaseModel):
#     """Sortie : signe prédit par le LSTM avec confiance."""
#     sign: str
#     confidence: float
#     message: str
