"""
Router FastAPI — Routes de prédiction Sensi.
Prefix : /api/v1
"""

import logging
from fastapi import APIRouter, Response, HTTPException
from app.schemas import PredictRequest, PredictResponse
from app.services.pipeline import pipeline_sensi
from app.services.model import get_glosses_from_sequence

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
    """
    logger.info(f"POST /predict/sentence/audio — glosses : {request.glosses}")

    try:
        phrase, audio = pipeline_sensi(request.glosses)
        return Response(
            content=audio,
            media_type="audio/mpeg",
            headers={"X-Phrase": phrase}
        )

    except ValueError as e:
        logger.warning(f"Données invalides : {e}")
        raise HTTPException(status_code=422, detail=str(e))

    except Exception as e:
        logger.error(f"Erreur pipeline : {e}")
        raise HTTPException(status_code=500, detail=f"Erreur interne : {e}")


@router.post("/predict/from-sequence", response_class=Response)
async def predict_from_sequence():
    """
    Lit output/last_phrase.txt écrit par scripts/test_team_live(llm).py,
    génère la phrase française et retourne le MP3.
    Appelé par Streamlit quand l'utilisateur clique "Traduire".
    """
    logger.info("POST /predict/from-sequence")

    try:
        glosses = get_glosses_from_sequence()
        logger.info(f"Glosses lues : {glosses}")

        if not glosses:
            logger.warning("Séquence vide — aucun signe détecté")
            return Response(
                content=b"",
                media_type="audio/mpeg",
                headers={"X-Phrase": "", "X-Error": "Séquence vide"}
            )

        phrase, audio = pipeline_sensi(glosses)
        logger.info(f"Phrase générée : {phrase}")

        return Response(
            content=audio,
            media_type="audio/mpeg",
            headers={
                "X-Phrase": phrase,
                "X-Glosses": " ".join(glosses)
            }
        )

    except Exception as e:
        logger.error(f"Erreur from-sequence : {e}")
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
#     """
#     Reçoit une frame (image) depuis la webcam et retourne le signe prédit.
#     Formats acceptés : image/jpeg, image/png
#     """
#     logger.info(f"POST /predict/live — content_type : {frame.content_type}")
#
#     if frame.content_type not in ["image/jpeg", "image/png"]:
#         raise HTTPException(
#             status_code=400,
#             detail=f"Format non supporté : {frame.content_type}. Utilise jpeg ou png"
#         )
#     try:
#         frame_bytes = await frame.read()
#         keypoints = extract_keypoints_from_frame(frame_bytes)
#         if keypoints is None:
#             raise HTTPException(status_code=422, detail="Aucune main détectée.")
#         sign, confidence = predict_sign(keypoints)
#         glosses = [sign.upper()]
#         phrase, audio = pipeline_sensi(glosses)
#         return PredictionResponse(
#             sign=sign,
#             confidence=confidence,
#             message=f"Signe détecté : {sign} — Phrase : {phrase}"
#         )
#     except HTTPException:
#         raise
#     except Exception as e:
#         logger.error(f"Erreur /predict/live : {e}")
#         raise HTTPException(status_code=500, detail=f"Erreur interne : {e}")


# @router.post("/predict/video", response_model=PredictionResponse)
# async def predict_from_video(video: UploadFile = File(...)):
#     """
#     Reçoit une vidéo complète et retourne le signe prédit.
#     Formats acceptés : .webm, .mp4
#     """
#     logger.info(f"POST /predict/video — content_type : {video.content_type}")
#
#     if video.content_type not in ["video/webm", "video/mp4"]:
#         raise HTTPException(
#             status_code=400,
#             detail=f"Format non supporté : {video.content_type}. Utilise .webm ou .mp4"
#         )
#     try:
#         video_bytes = await video.read()
#         keypoints = extract_keypoints_from_video(video_bytes)
#         if keypoints is None:
#             raise HTTPException(status_code=422, detail="Impossible d'extraire les keypoints.")
#         sign, confidence = predict_sign(keypoints)
#         glosses = [sign.upper()]
#         phrase, audio = pipeline_sensi(glosses)
#         return PredictionResponse(
#             sign=sign,
#             confidence=confidence,
#             message=f"Signe détecté : {sign} — Phrase : {phrase}"
#         )
#     except HTTPException:
#         raise
#     except Exception as e:
#         logger.error(f"Erreur /predict/video : {e}")
#         raise HTTPException(status_code=500, detail=f"Erreur interne : {e}")
