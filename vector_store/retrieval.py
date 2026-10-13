"""
vector_store/retrieval.py - Optional Vector Retrieval and Context Store
Independent extension providing user-isolated vector retrieval of approved templates
and authorized prior email context. Does not affect core pipeline if disabled.
"""

import logging
from typing import List, Dict, Any, Optional
from ai.semantic_embeddings import get_embedding_engine

logger = logging.getLogger(__name__)


class ContextRecord:
    def __init__(self, record_id: str, text: str, user_id: str, category: str = "template"):
        self.record_id = record_id
        self.text = text
        self.user_id = user_id
        self.category = category


class ContextRetrievalService:
    """
    User-isolated vector retriever.
    Supports in-memory cosine matching across authorized templates and records.
    Protects against cross-user leakage by enforcing strict user_id filtering.
    """

    def __init__(self):
        self.embedding_engine = get_embedding_engine()
        # In-memory document storage indexed by user_id
        self._store: Dict[str, List[ContextRecord]] = {}
        self._initialize_sample_templates()

    def _initialize_sample_templates(self):
        """Seed sample approved organization templates for default demo user."""
        default_templates = [
            ContextRecord("tmpl-1", "Standard academic extension policy permits up to 3 days for medical or technical justifications.", "default_user", "policy"),
            ContextRecord("tmpl-2", "Corporate meeting reschedule format: Propose 2 alternative time slots and state reason succinctly.", "default_user", "template"),
            ContextRecord("tmpl-3", "Refund authorization template: Always request transaction ID and verify duplicate charge date.", "default_user", "guideline")
        ]
        self._store["default_user"] = default_templates

    def add_context_record(self, user_id: str, record_id: str, text: str, category: str = "template"):
        """Add an explicitly authorized user context record."""
        if user_id not in self._store:
            self._store[user_id] = []
        self._store[user_id].append(ContextRecord(record_id, text, user_id, category))

    def retrieve_relevant_context(
        self,
        query: str,
        user_id: str = "default_user",
        top_k: int = 2,
        similarity_threshold: float = 0.15
    ) -> List[Dict[str, Any]]:
        """
        Retrieves authorized relevant records strictly belonging to the calling user.
        """
        user_records = self._store.get(user_id, [])
        if not user_records or not query:
            return []

        scored_records = []
        for rec in user_records:
            sim = self.embedding_engine.compute_similarity(query, rec.text)
            if sim >= similarity_threshold:
                scored_records.append({
                    "record_id": rec.record_id,
                    "text": rec.text,
                    "category": rec.category,
                    "similarity": round(sim, 3)
                })

        scored_records.sort(key=lambda x: x["similarity"], reverse=True)
        return scored_records[:top_k]
