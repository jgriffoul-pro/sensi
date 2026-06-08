"""
Point d'entrée de l'API Sensi.
Lance avec : uvicorn app.main:app --reload --port 8000
"""

import logging
import yaml
from pathlib import Path
from fastapi import FastAPI
from app.routers import predict
from app.schemas import HealthResponse

# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent
CONFIG_PATH = BASE_DIR / "config" / "config.yaml"

with open(CONFIG_PATH, encoding="utf-8") as f:
    config = yaml.safe_load(f)

# ============================================================
# LOGGING
# ============================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s — %(name)s — %(levelname)s — %(message)s",
)
logger = logging.getLogger(__name__)

# ============================================================
# APPLICATION
# ============================================================

app = FastAPI(
    title=config["api"]["title"],
    description=config["api"]["description"],
    version=config["api"]["version"],
)

app.include_router(predict.router, prefix="/api/v1")

logger.info("API Sensi démarrée ✅")
logger.info(f"Titre   : {config['api']['title']}")
logger.info(f"Version : {config['api']['version']}")


# ============================================================
# HEALTH CHECK
# ============================================================

@app.get("/", response_model=HealthResponse)
def root():
    """Health check — vérifie que l'API tourne."""
    logger.info("Health check appelé")
    return HealthResponse(
        status="ok",
        message="Sensi API is running"
    )
