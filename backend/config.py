"""
backend/config.py - Application Configuration
Reads settings from environment variables with sensible defaults and validation.
"""

import os
from typing import Optional
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )

    # General
    ENVIRONMENT: str = "development"
    LOG_LEVEL: str = "INFO"
    SECRET_KEY: str = "default-insecure-secret-key-change-in-production"
    DISABLE_EMAIL_CONTENT_PERSISTENCE: bool = False

    # LLM Settings
    LLM_PROVIDER: str = "mock"  # "mock", "openai", "local", "gemini"
    OPENAI_API_KEY: Optional[str] = None
    OPENAI_MODEL_NAME: str = "gpt-4o-mini"
    OPENAI_API_BASE: str = "https://api.openai.com/v1"

    LOCAL_LLM_URL: str = "http://localhost:11434/api/generate"
    LOCAL_LLM_MODEL: str = "llama3:8b"

    GEMINI_API_KEY: Optional[str] = None
    GEMINI_MODEL_NAME: str = "gemini-3.5-flash-lite"
    GEMINI_API_BASE: str = "https://generativelanguage.googleapis.com/v1beta"

    DEFAULT_TEMPERATURE: float = 0.7
    DEFAULT_MAX_TOKENS: int = 800

    # NLP & Semantic Embedding
    EMBEDDING_MODEL_NAME: str = "all-MiniLM-L6-v2"
    ENABLE_SPACY_EXTRACTION: bool = True
    SIMILARITY_THRESHOLD_PASS: float = 0.65
    SIMILARITY_THRESHOLD_REVIEW: float = 0.45

    # Intent Classification Architecture
    INTENT_CLASSIFIER_BACKEND: str = "auto"  # "auto", "setfit", "head", "exemplar"
    INTENT_MODEL_PATH: str = "data/models/setfit_intent"
    INTENT_CONFIDENCE_THRESHOLD: float = 0.17
    INTENT_MULTI_THRESHOLD: float = 0.20

    # Database
    DATABASE_URL: str = "sqlite:///./email_assistant.db"

    # Vector Storage
    VECTOR_STORE_TYPE: str = "in_memory"

    # API Host/Port
    BACKEND_HOST: str = "127.0.0.1"
    BACKEND_PORT: int = 8000
    FRONTEND_PORT: int = 8501

    @property
    def active_gemini_key(self) -> Optional[str]:
        """Dynamically retrieves GEMINI_API_KEY from in-memory settings, os.environ, or .env file."""
        if self.GEMINI_API_KEY and self.GEMINI_API_KEY.strip():
            return self.GEMINI_API_KEY.strip()
        env_val = os.getenv("GEMINI_API_KEY")
        if env_val and env_val.strip():
            return env_val.strip()
        try:
            if os.path.exists(".env"):
                with open(".env", "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if line.startswith("GEMINI_API_KEY="):
                            val = line.split("=", 1)[1].strip().strip("\"'")
                            if val:
                                return val
        except Exception:
            pass
        return None


def normalize_gemini_model(model_name: Optional[str]) -> str:
    """
    Normalizes Gemini model names to the canonical Google Generative Language API ID.
    Maps variants like 'gemini-3.5', 'gemini-3.5-lite', 'gemini-3.5-flash-lite' to 'gemini-3.5-flash-lite',
    and 'gemini-3.8' variants to 'gemini-3.8-flash'.
    """
    if not model_name:
        return "gemini-3.5-flash-lite"
    cleaned = model_name.strip()
    lower = cleaned.lower()
    if lower in (
        "gemini-3.5",
        "gemini-3.5-lite",
        "gemini-3.5-flash-lite",
        "gemini 3.5",
        "gemini 3.5 lite",
        "gemini 3.5 flash lite"
    ):
        return "gemini-3.5-flash-lite"
    if lower in (
        "gemini-3.8",
        "gemini-3.8-live",
        "gemini-3.8-flash",
        "gemini 3.8",
        "gemini 3.8 live",
        "gemini 3.8 flash",
        "gemini-3.8-live-extended-thinking"
    ):
        return "gemini-3.8-flash"
    return cleaned


# Global cached settings instance
settings = Settings()

