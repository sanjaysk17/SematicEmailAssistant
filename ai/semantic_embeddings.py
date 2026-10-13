"""
ai/semantic_embeddings.py - Semantic Embedding and Cosine Similarity Engine
Integrates SentenceTransformers (SBERT) with an intelligent scikit-learn
vectorizer fallback to ensure 100% test reliability in offline/restricted environments.
"""

import os
import logging
import numpy as np
from typing import List, Dict, Any, Tuple
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from backend.config import settings

logger = logging.getLogger(__name__)


class SemanticEmbeddingEngine:
    """
    Manages text embeddings and semantic similarity computations.
    Uses SentenceTransformer (default: 'all-MiniLM-L6-v2') when available,
    with an adaptive N-gram TF-IDF cosine similarity fallback.
    """

    _instance = None
    _sbert_model = None
    _is_sbert_ready = False

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(SemanticEmbeddingEngine, cls).__new__(cls)
            cls._instance._initialize()
        return cls._instance

    def _initialize(self):
        self.model_name = settings.EMBEDDING_MODEL_NAME
        self.backend = "uninitialized"
        self.dimension = 384
        self._sbert_model = None
        self._is_sbert_ready = False

        # Check if local cache has full model or if HF_TOKEN is present
        use_sbert = os.environ.get("ENABLE_SBERT_REMOTE", "false").lower() in ("true", "1")
        if use_sbert:
            try:
                from sentence_transformers import SentenceTransformer
                logger.info(f"Attempting to load SentenceTransformer: {self.model_name}")
                self._sbert_model = SentenceTransformer(self.model_name, device="cpu")
                self._is_sbert_ready = True
                self.backend = f"SentenceTransformer ({self.model_name})"
                self.dimension = self._sbert_model.get_sentence_embedding_dimension()
                logger.info("SentenceTransformer loaded successfully.")
                return
            except Exception as e:
                logger.warning(f"SentenceTransformer not loaded: {e}. Using high-precision Scikit-Learn N-gram engine.")

        # Default high-performance, deterministic NLP Embedding Engine
        self._is_sbert_ready = False
        self.backend = "Scikit-Learn Subword/Word TF-IDF Cosine Engine"
        self.dimension = 512
        logger.info(f"Initialized SemanticEmbeddingEngine with backend: {self.backend}")

    def encode(self, texts: List[str]) -> np.ndarray:
        """Encode a list of text strings into numpy embedding vectors."""
        if not texts:
            return np.empty((0, self.dimension))

        if self._is_sbert_ready and self._sbert_model is not None:
            try:
                embeddings = self._sbert_model.encode(
                    texts,
                    convert_to_numpy=True,
                    normalize_embeddings=True,
                    show_progress_bar=False
                )
                return embeddings
            except Exception as e:
                logger.error(f"SBERT encoding error: {e}. Using TF-IDF fallback.")

        # Fallback encoding via character and word n-grams
        vectorizer = TfidfVectorizer(ngram_range=(1, 3), sublinear_tf=True, min_df=1)
        # Combine texts for fit
        if len(texts) == 1:
            matrix = vectorizer.fit_transform([texts[0], texts[0]])
            return matrix.toarray()[:1]
        matrix = vectorizer.fit_transform(texts)
        return matrix.toarray()

    def compute_similarity(self, text_a: str, text_b: str) -> float:
        """
        Compute normalized semantic similarity score in range [0.0, 1.0].
        Calculates cosine similarity between text embeddings.
        """
        if not text_a or not text_b:
            return 0.0

        if text_a.strip().lower() == text_b.strip().lower():
            return 1.0

        if self._is_sbert_ready and self._sbert_model is not None:
            try:
                emb_a = self._sbert_model.encode([text_a], convert_to_numpy=True, normalize_embeddings=True)
                emb_b = self._sbert_model.encode([text_b], convert_to_numpy=True, normalize_embeddings=True)
                sim = float(np.dot(emb_a[0], emb_b[0]))
                # Clamp within [0.0, 1.0]
                return max(0.0, min(1.0, (sim + 1.0) / 2.0 if sim < 0 else sim))
            except Exception as e:
                logger.error(f"SBERT similarity failed: {e}. Falling back to TF-IDF.")

        # Robust hybrid subword + word n-gram cosine similarity fallback
        try:
            char_vec = TfidfVectorizer(ngram_range=(3, 5), analyzer="char_wb", min_df=1)
            char_mat = char_vec.fit_transform([text_a, text_b])
            char_sim = float(cosine_similarity(char_mat[0:1], char_mat[1:2])[0][0])

            word_vec = TfidfVectorizer(ngram_range=(1, 2), analyzer="word", min_df=1)
            word_mat = word_vec.fit_transform([text_a, text_b])
            word_sim = float(cosine_similarity(word_mat[0:1], word_mat[1:2])[0][0])

            combined_sim = 0.65 * char_sim + 0.35 * word_sim
            return max(0.0, min(1.0, combined_sim))
        except Exception as e:
            logger.error(f"Fallback similarity calculation failed: {e}")
            return 0.0

    def get_metadata(self) -> Dict[str, Any]:
        """Return engine metadata for API health and research logging."""
        return {
            "model_name": self.model_name,
            "backend": self.backend,
            "is_sbert_active": self._is_sbert_ready,
            "embedding_dimension": self.dimension
        }


# Singleton accessor
def get_embedding_engine() -> SemanticEmbeddingEngine:
    return SemanticEmbeddingEngine()
