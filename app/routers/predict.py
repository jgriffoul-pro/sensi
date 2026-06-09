"""
Router FastAPI — Routes de prédiction Sensi.
Prefix : /api/v1
"""

import logging
from fastapi import APIRouter, Response, HTTPException
from app.schemas import PredictRequest, PredictResponse
from app.services.pipeline import pipeline_sensi

# ============================================================
# LOGGING
# ============================================================

logger = logging.getLogger(__name__)

router = APIRouter()


# ============================================================
# ROUTES ACTIVES — NLP + TTS
# ============================================================

@router.post("/predict/sentence", response_model=PredictResponse)
async def predict_sentence(request: PredictRequest):
    """
    Reçoit une liste de glosses et retourne la phrase française.
    ex: {"glosses": ["BONJOUR", "JE_SUIS", "CONTENT"]}
    """
    logger.info(f"POST /predict/sentence — glosses : {request.glosses}")

    try:
        phrase, _ = pipeline_sensi(request.glosses)
        return PredictResponse(phrase=phrase)

    except ValueError as e:
        logger.warning(f"Données invalides : {e}")
        raise HTTPException(status_code=422, detail=str(e))

    except Exception as e:
        logger.error(f"Erreur pipeline : {e}")
        raise HTTPException(status_code=500, detail=f"Erreur interne : {e}")


@router.post("/predict/sentence/audio", response_class=Response)
async def predict_sentence_audio(request: PredictRequest):
    """
    Reçoit une liste de glosses et retourne le fichier MP3.
    La phrase française est disponible dans le header X-Phrase.
    Appelé directement par scripts/test_team_live(llm).py (keypress V).
    ex: {"glosses": ["BONJOUR", "JE_SUIS", "CONTENT"]}
    """
    logger.info(f"POST /predict/sentence/audio — glosses : {request.glosses}")

    try:
        phrase, audio = pipeline_sensi(request.glosses)
        return Response(
            content=audio,
            media_type="audio/mp3",
            headers={"X-Phrase": phrase}
        )

    except ValueError as e:
        logger.warning(f"Données invalides : {e}")
        raise HTTPException(status_code=422, detail=str(e))

    except Exception as e:
        logger.error(f"Erreur pipeline : {e}")
        raise HTTPException(status_code=500, detail=f"Erreur interne : {e}")


# ============================================================
# ROUTES EN ATTENTE — Webcam directe (option future)
# À décommenter si on intègre la webcam dans l'API
# ============================================================

# from fastapi import UploadFile, File
# from app.schemas import PredictionResponse
# from app.services.preprocessing import (
#     extract_keypoints_from_frame,
#     extract_keypoints_from_video,
# )
# from app.services.model import predict_sign


# @router.post("/predict/live", response_model=PredictionResponse)
# async def predict_from_live(frame: UploadFile = File(...)):
#     """Reçoit une frame (image) depuis la webcam et retourne le signe prédit."""
#     ...

# @router.post("/predict/video", response_model=PredictionResponse)
# async def predict_from_video(video: UploadFile = File(...)):
#     """Reçoit une vidéo complète et retourne le signe prédit."""
#     ...
