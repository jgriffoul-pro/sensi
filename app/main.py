from fastapi import FastAPI
from app.routers import predict
from app.schemas import HealthResponse

app = FastAPI(
    title="Sensi API",
    description="API de traduction du langage des signes (LSF) en français",
    version="0.1.0"
)

app.include_router(predict.router, prefix="/api/v1")

@app.get("/", response_model=HealthResponse)
def root():
    return HealthResponse(
        status="ok",
        message="Sensi API is running"
    )
