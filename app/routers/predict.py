from fastapi import APIRouter, Response
# from fastapi import UploadFile, File, HTTPException  # décommenté si webcam dans API
from app.schemas import PredictRequest, PredictResponse
# from app.schemas import PredictionResponse           # décommenté si webcam dans API
from app.services.pipeline import pipeline_sensi
from app.services.model import get_glosses_from_sequence
# from app.services.preprocessing import (            # décommenté si webcam dans API
#     extract_keypoints_from_frame,
#     extract_keypoints_from_video,
# )
# from app.services.model import predict_sign         # décommenté si webcam dans API

router = APIRouter()


# ============================================================
# ROUTES ACTIVES — NLP + TTS
# ============================================================

@router.post("/predict/sentence")
async def predict_sentence(request: PredictRequest):
    """
    Reçoit une liste de glosses et retourne la phrase française.
    ex: {"glosses": ["BONJOUR", "JE_SUIS", "CONTENT"]}
    """
    phrase, audio = pipeline_sensi(request.glosses)
    return PredictResponse(phrase=phrase)


@router.post("/predict/sentence/audio", response_class=Response)
async def predict_sentence_audio(request: PredictRequest):
    """
    Reçoit une liste de glosses et retourne le fichier MP3.
    La phrase française est disponible dans le header X-Phrase.
    """
    phrase, audio = pipeline_sensi(request.glosses)
    return Response(
        content=audio,
        media_type="audio/mp3",
        headers={"X-Phrase": phrase}
    )


@router.post("/predict/from-sequence", response_class=Response)
async def predict_from_sequence():
    """
    Lit output/sequence.txt écrit par test_team_live(llm).py,
    génère la phrase française et retourne le MP3.
    Appelé par Streamlit quand l'utilisateur clique "Traduire".
    """
    glosses = get_glosses_from_sequence()

    if not glosses:
        return Response(
            content=b"",
            media_type="audio/mp3",
            headers={"X-Phrase": "", "X-Error": "Séquence vide"}
        )

    phrase, audio = pipeline_sensi(glosses)
    return Response(
        content=audio,
        media_type="audio/mp3",
        headers={
            "X-Phrase": phrase,
            "X-Glosses": " ".join(glosses),
        }
    )


# ============================================================
# ROUTES EN ATTENTE — Webcam directe (option future)
# À décommenter si on intègre la webcam dans l'API
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
#     glosses = [sign.upper()]
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
#             detail="Impossible d'extraire les keypoints."
#         )
#     sign, confidence = predict_sign(keypoints)
#     glosses = [sign.upper()]
#     phrase, audio = pipeline_sensi(glosses)
#     return PredictionResponse(
#         sign=sign,
#         confidence=confidence,
#         message=f"Signe détecté : {sign} — Phrase : {phrase}"
#     )
