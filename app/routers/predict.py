from fastapi import APIRouter, Response
# from fastapi import UploadFile, File, HTTPException  # décommenté quand LSTM branché
from app.schemas import PredictRequest, PredictResponse
# from app.schemas import PredictionResponse           # décommenté quand LSTM branché
from app.services.pipeline import pipeline_sensi
# from app.services.preprocessing import (            # décommenté quand LSTM branché
#     extract_keypoints_from_video,
#     extract_keypoints_from_frame
# )
# from app.services.model import predict_sign         # décommenté quand LSTM branché

router = APIRouter()


# ============================================================
# ROUTES ACTIVES — NLP + TTS (Franck)
# ============================================================

@router.post("/predict/sentence")
async def predict_sentence(request: PredictRequest):
    """
    Reçoit une liste de glosses prédites par le LSTM
    et retourne la phrase française.
    ex: {"glosses": ["BONJOUR", "JE_SUIS", "CONTENT", "PRESENTER", "PROJET"]}
    """
    phrase, audio = pipeline_sensi(request.glosses)
    return PredictResponse(phrase=phrase)


@router.post("/predict/sentence/audio", response_class=Response)
async def predict_sentence_audio(request: PredictRequest):
    """
    Reçoit une liste de glosses prédites par le LSTM
    et retourne directement le fichier MP3.
    Le header X-Phrase contient la phrase française générée.
    """
    phrase, audio = pipeline_sensi(request.glosses)
    return Response(
        content=audio,
        media_type="audio/mp3",
        headers={"X-Phrase": phrase}
    )


# ============================================================
# ROUTES EN ATTENTE — LSTM (coéquipiers)
# À décommenter quand preprocessing.py et model.py sont prêts
# ============================================================

# @router.post("/predict/live", response_model=PredictionResponse)
# async def predict_from_live(frame: UploadFile = File(...)):
#     """
#     Reçoit une frame (image) depuis la webcam et retourne le signe prédit.
#     Formats acceptés : image/jpeg, image/png
#     """
#     if frame.content_type not in ["image/jpeg", "image/png"]:
#         raise HTTPException(
#             status_code=400,
#             detail=f"Format non supporté : {frame.content_type}. Utilise jpeg ou png"
#         )
#     frame_bytes = await frame.read()
#     keypoints = extract_keypoints_from_frame(frame_bytes)
#     if keypoints is None:
#         raise HTTPException(
#             status_code=422,
#             detail="Aucune main détectée dans cette frame."
#         )
#     sign, confidence = predict_sign(keypoints)
#     glosses = [sign]                          # à adapter selon format LSTM
#     phrase, audio = pipeline_sensi(glosses)
#     return PredictionResponse(
#         sign=sign,
#         confidence=confidence,
#         message=f"Signe détecté : {sign} — Phrase : {phrase}"
#     )


# @router.post("/predict/video", response_model=PredictionResponse)
# async def predict_from_video(video: UploadFile = File(...)):
#     """
#     Reçoit une vidéo complète et retourne le signe prédit.
#     Formats acceptés : .webm, .mp4
#     """
#     if video.content_type not in ["video/webm", "video/mp4"]:
#         raise HTTPException(
#             status_code=400,
#             detail=f"Format non supporté : {video.content_type}. Utilise .webm ou .mp4"
#         )
#     video_bytes = await video.read()
#     keypoints = extract_keypoints_from_video(video_bytes)
#     if keypoints is None:
#         raise HTTPException(
#             status_code=422,
#             detail="Impossible d'extraire les keypoints. Vérifie que la vidéo contient des mains visibles."
#         )
#     sign, confidence = predict_sign(keypoints)
#     glosses = [sign]                          # à adapter selon format LSTM
#     phrase, audio = pipeline_sensi(glosses)
#     return PredictionResponse(
#         sign=sign,
#         confidence=confidence,
#         message=f"Signe détecté : {sign} — Phrase : {phrase}"
#     )
