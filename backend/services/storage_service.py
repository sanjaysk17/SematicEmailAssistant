"""
backend/services/storage_service.py - Database Persistence Service
Manages storage of requests, analyses, generated emails, validations, and experiments.
Enforces privacy settings by filtering or omitting private text when requested.
"""

import logging
from typing import Dict, Any, Optional, List
from sqlalchemy import or_
from sqlalchemy.orm import Session

from backend.models.db_models import (
    EmailRequest, SemanticAnalysisRecord, GeneratedEmailRecord,
    ValidationResultRecord, Experiment, ExperimentResultRecord,
    UserDraft
)
from backend.security.sanitizer import SecuritySanitizer

logger = logging.getLogger(__name__)


class StorageService:
    """Provides typed database persistence methods."""

    @staticmethod
    def save_request(
        db: Session,
        raw_requirement: str,
        context: Optional[str] = None,
        recipient_hint: Optional[str] = None,
        tone_hint: Optional[str] = None
    ) -> EmailRequest:
        sanitized_req = SecuritySanitizer.filter_content_for_storage(raw_requirement)
        sanitized_ctx = SecuritySanitizer.filter_content_for_storage(context) if context else None

        req = EmailRequest(
            raw_requirement=sanitized_req,
            context=sanitized_ctx,
            recipient_hint=recipient_hint,
            tone_hint=tone_hint
        )
        db.add(req)
        db.commit()
        db.refresh(req)
        return req

    @staticmethod
    def save_analysis(
        db: Session,
        request_id: str,
        structured_representation: Dict[str, Any],
        intent: str,
        tone: str,
        uncertainties: List[str],
        model_identifier: str,
        confidence: float
    ) -> SemanticAnalysisRecord:
        record = SemanticAnalysisRecord(
            request_id=request_id,
            structured_representation=structured_representation,
            intent=intent,
            tone=tone,
            uncertainties=uncertainties,
            model_identifier=model_identifier,
            confidence=confidence
        )
        db.add(record)
        db.commit()
        db.refresh(record)
        return record

    @staticmethod
    def save_generation(
        db: Session,
        request_id: str,
        analysis_id: Optional[str],
        subject: str,
        body: str,
        system_version: str,
        model_identifier: str,
        generation_params: Dict[str, Any]
    ) -> GeneratedEmailRecord:
        stored_body = SecuritySanitizer.filter_content_for_storage(body)
        record = GeneratedEmailRecord(
            request_id=request_id,
            analysis_id=analysis_id,
            subject=subject,
            body=stored_body,
            system_version=system_version,
            model_identifier=model_identifier,
            generation_params=generation_params
        )
        db.add(record)
        db.commit()
        db.refresh(record)
        return record

    @staticmethod
    def save_validation(
        db: Session,
        generated_email_id: str,
        overall_status: str,
        semantic_similarity: float,
        intent_preserved: bool,
        tone_suitable: bool,
        completeness_score: float,
        missing_details: List[str],
        unsupported_details: List[str],
        contradictions: List[str],
        raw_report: Dict[str, Any]
    ) -> ValidationResultRecord:
        record = ValidationResultRecord(
            generated_email_id=generated_email_id,
            overall_status=overall_status,
            semantic_similarity=semantic_similarity,
            intent_preserved=intent_preserved,
            tone_suitable=tone_suitable,
            completeness_score=completeness_score,
            missing_details=missing_details,
            unsupported_details=unsupported_details,
            contradictions=contradictions,
            raw_report=raw_report
        )
        db.add(record)
        db.commit()
        db.refresh(record)
        return record

    @staticmethod
    def save_experiment(
        db: Session,
        experiment_name: str,
        description: str,
        baseline_type: str,
        dataset_name: str,
        sample_count: int,
        config_params: Dict[str, Any]
    ) -> Experiment:
        exp = Experiment(
            experiment_name=experiment_name,
            description=description,
            baseline_type=baseline_type,
            dataset_name=dataset_name,
            sample_count=sample_count,
            config_params=config_params
        )
        db.add(exp)
        db.commit()
        db.refresh(exp)
        return exp

    @staticmethod
    def save_experiment_result(
        db: Session,
        experiment_id: str,
        example_id: str,
        method: str,
        subject: str,
        body: str,
        intent_score: float,
        completeness_score: float,
        semantic_similarity: float,
        tone_score: float,
        unsupported_count: int,
        latency_ms: float,
        status: str
    ) -> ExperimentResultRecord:
        rec = ExperimentResultRecord(
            experiment_id=experiment_id,
            example_id=example_id,
            method=method,
            generated_subject=subject,
            generated_body=body,
            intent_score=intent_score,
            completeness_score=completeness_score,
            semantic_similarity=semantic_similarity,
            tone_score=tone_score,
            unsupported_count=unsupported_count,
            latency_ms=latency_ms,
            status=status
        )
        db.add(rec)
        db.commit()
        db.refresh(rec)
        return rec

    @staticmethod
    def save_draft(
        db: Session,
        prompt: str,
        subject: str,
        body: str,
        intent: Optional[str] = None,
        tone: Optional[str] = None,
        recipient: Optional[str] = None,
        user_id: Optional[str] = None,
        user_email: Optional[str] = None,
        draft_id: Optional[str] = None
    ) -> UserDraft:
        """Persists or updates a user draft with its generating prompt in database."""
        sanitized_prompt = SecuritySanitizer.filter_content_for_storage(prompt)
        sanitized_body = SecuritySanitizer.filter_content_for_storage(body)
        sanitized_sub = SecuritySanitizer.filter_content_for_storage(subject)

        if draft_id:
            existing = db.query(UserDraft).filter(UserDraft.id == draft_id).first()
            if existing:
                # If existing draft has an owner and incoming user differs, block unauthorized update
                if user_id and existing.user_id and existing.user_id != user_id:
                    raise ValueError("Cannot update draft belonging to another user.")
                if user_email and existing.user_email and existing.user_email != user_email:
                    raise ValueError("Cannot update draft belonging to another user.")
                existing.prompt = sanitized_prompt
                existing.subject = sanitized_sub
                existing.body = sanitized_body
                if intent is not None:
                    existing.intent = intent
                if tone is not None:
                    existing.tone = tone
                if recipient is not None:
                    existing.recipient = recipient
                if user_id:
                    existing.user_id = user_id
                if user_email:
                    existing.user_email = user_email
                db.commit()
                db.refresh(existing)
                return existing

        draft = UserDraft(
            user_id=user_id,
            user_email=user_email,
            prompt=sanitized_prompt,
            subject=sanitized_sub,
            body=sanitized_body,
            intent=intent,
            tone=tone,
            recipient=recipient
        )
        db.add(draft)
        db.commit()
        db.refresh(draft)
        return draft

    @staticmethod
    def get_drafts(
        db: Session,
        user_id: Optional[str] = None,
        user_email: Optional[str] = None,
        limit: int = 100,
        strict_user: bool = False
    ) -> List[UserDraft]:
        """
        Retrieves saved drafts ordered by creation time descending.
        If strict_user is True, strictly returns drafts belonging to the specified user.
        """
        query = db.query(UserDraft)
        if strict_user:
            conditions = []
            if user_id:
                conditions.append(UserDraft.user_id == user_id)
            if user_email:
                conditions.append(UserDraft.user_email == user_email)
            if conditions:
                query = query.filter(or_(*conditions))
            else:
                return []
        else:
            if user_id:
                query = query.filter((UserDraft.user_id == user_id) | (UserDraft.user_id.is_(None)))
            elif user_email:
                query = query.filter((UserDraft.user_email == user_email) | (UserDraft.user_email.is_(None)))
        return query.order_by(UserDraft.created_at.desc()).limit(limit).all()

    @staticmethod
    def get_draft(
        db: Session,
        draft_id: str,
        user_id: Optional[str] = None,
        user_email: Optional[str] = None
    ) -> Optional[UserDraft]:
        """Retrieves a single saved draft by ID with optional owner verification."""
        draft = db.query(UserDraft).filter(UserDraft.id == draft_id).first()
        if not draft:
            return None
        if user_id and draft.user_id and draft.user_id != user_id:
            return None
        if user_email and draft.user_email and draft.user_email != user_email:
            return None
        return draft

    @staticmethod
    def delete_draft(
        db: Session,
        draft_id: str,
        user_id: Optional[str] = None,
        user_email: Optional[str] = None
    ) -> bool:
        """Deletes a draft by ID with optional owner verification."""
        draft = db.query(UserDraft).filter(UserDraft.id == draft_id).first()
        if not draft:
            return False
        if user_id and draft.user_id and draft.user_id != user_id:
            return False
        if user_email and draft.user_email and draft.user_email != user_email:
            return False
        db.delete(draft)
        db.commit()
        return True
