"""
backend/main.py - FastAPI Application Entrypoint
Initializes application, database schema, CORS policies, and routing.
"""

import os
import sys

# Ensure project root is in sys.path regardless of directory from which uvicorn is launched
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from contextlib import asynccontextmanager
import logging
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.config import settings
from backend.database.connection import init_db
from backend.api.routes import router, health_check


logging.basicConfig(
    level=getattr(logging, settings.LOG_LEVEL.upper(), logging.INFO),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("backend")

# Initialize database schema
init_db()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan context manager.
    Loads and warms neural intent classification models and semantic embedding engines
    once on startup, completely eliminating per-request loading latency.
    """
    logger.info("Initializing and warming Intent Classifier...")
    try:
        from ai.intent_classification import get_intent_classifier
        clf = get_intent_classifier()
        clf.warmup()
        logger.info(f"Intent Classifier ready (active backend: {clf.active_backend}).")
    except Exception as e:
        logger.warning(f"Intent Classifier startup warmup non-fatal issue: {e}")
    yield


app = FastAPI(
    title="AI-Powered Email Writing Assistant API",
    description="Research-oriented semantic-guided email generation and validation framework.",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

# CORS Middleware (allowing Streamlit and frontend integrations)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register routes
app.include_router(router)


@app.get("/")
def root():
    return {
        "service": "AI-Powered Email Writing Assistant with Semantic Analysis",
        "documentation": "/docs",
        "health": "/api/health"
    }


@app.get("/health")
def root_health():
    return health_check()


# Mount Browser Extension for live web preview
from fastapi.staticfiles import StaticFiles
from fastapi.responses import RedirectResponse

extension_dir = os.path.join(PROJECT_ROOT, "browser_extension")
if os.path.isdir(extension_dir):
    app.mount("/extension", StaticFiles(directory=extension_dir, html=True), name="extension")

data_dir = os.path.join(PROJECT_ROOT, "data")
if os.path.isdir(data_dir):
    app.mount("/data", StaticFiles(directory=data_dir), name="data")


@app.get("/app")
def redirect_to_extension():
    return RedirectResponse(url="/extension/popup.html")




if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "backend.main:app",
        host=settings.BACKEND_HOST,
        port=settings.BACKEND_PORT,
        reload=True
    )
