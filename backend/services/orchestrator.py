"""
backend/services/orchestrator.py - Core Orchestration Pipeline Service
Connects Semantic Interpretation, Generation, Output Validation, and Storage
providing a unified API abstraction.
"""

import logging
from typing import Dict, Any, Optional
from sqlalchemy.orm import Session

from backend.schemas.email_schemas import (
    AnalyzeRequest, AnalyzeResponse,
    GenerateRequest, GenerateResponse,
    ValidateRequest, OutputValidationReport,
    StructuredRequirement
)
from backend.services.storage_service import StorageService
from backend.security.sanitizer import SecuritySanitizer
from ai.semantic_interpretation import SemanticInterpreter
from ai.generation import EmailGenerationService
from ai.validation import OutputValidator

logger = logging.getLogger(__name__)


class PipelineOrchestrator:
    """Coordinates the UNDERSTAND -> GENERATE -> VERIFY workflow."""

    def __init__(self):
        self.interpreter = SemanticInterpreter()
        self.generation_service = EmailGenerationService()
        self.validator = OutputValidator()

    def analyze(self, req: AnalyzeRequest, db: Optional[Session] = None) -> AnalyzeResponse:
        """Stage 1: UNDERSTAND - interprets natural language requirements into structured representation."""
        sanitized_req = SecuritySanitizer.sanitize_input(req.requirement)
        sanitized_ctx = SecuritySanitizer.sanitize_input(req.context) if req.context else None

        user_preferences = {
            "recipient_hint": req.recipient_hint,
            "tone_hint": req.tone_hint,
            "desired_length": req.desired_length,
            "language": req.language,
            "explicit_restrictions": req.explicit_restrictions,
            "api_key": req.api_key,
            "llm_provider": req.llm_provider
        }

        result = self.interpreter.interpret(
            raw_requirement=sanitized_req,
            context=sanitized_ctx,
            user_preferences=user_preferences
        )

        representation: StructuredRequirement = result["representation"]

        request_id = "temp-request-id"
        if db:
            db_req = StorageService.save_request(
                db=db,
                raw_requirement=sanitized_req,
                context=sanitized_ctx,
                recipient_hint=req.recipient_hint,
                tone_hint=req.tone_hint
            )
            request_id = db_req.id
            StorageService.save_analysis(
                db=db,
                request_id=request_id,
                structured_representation=representation.model_dump(),
                intent=representation.intent,
                tone=representation.tone,
                uncertainties=representation.uncertainties,
                model_identifier=result["model_identifier"],
                confidence=result["confidence"]
            )

        return AnalyzeResponse(
            request_id=request_id,
            original_requirement=req.requirement,
            representation=representation,
            model_identifier=result["model_identifier"],
            confidence=result["confidence"],
            analysis_time_ms=result["elapsed_ms"],
            warnings=result["warnings"]
        )

    def generate(
        self,
        req: GenerateRequest,
        db: Optional[Session] = None,
        user_id: Optional[str] = None,
        user_email: Optional[str] = None
    ) -> GenerateResponse:
        sanitized_req = SecuritySanitizer.sanitize_input(req.requirement)
        sanitized_ctx = SecuritySanitizer.sanitize_input(req.context) if req.context else None
        sanitized_revision = SecuritySanitizer.sanitize_input(req.revision_instruction) if req.revision_instruction else None

        gen_result = self.generation_service.generate(
            requirement=sanitized_req,
            representation=req.representation,
            context=sanitized_ctx,
            desired_length=req.desired_length or "medium",
            language=req.language or "English",
            system_version=req.system_version or "proposed",
            sender_name=req.sender_name,
            sender_role=req.sender_role,
            sender_organization=req.sender_organization,
            sender_sign_off=req.sender_sign_off,
            sender_about_me=req.sender_about_me,
            provider_override=req.llm_provider,
            api_key_override=req.api_key,
            model_name_override=req.model_name,
            revision_instruction=sanitized_revision,
            previous_draft=req.previous_draft,
            temperature=req.temperature
        )

        email_id = None
        draft_id = None
        if db:
            if req.request_id:
                try:
                    db_gen = StorageService.save_generation(
                        db=db,
                        request_id=req.request_id,
                        analysis_id=None,
                        subject=gen_result["subject"],
                        body=gen_result["body"],
                        system_version=gen_result["system_version"],
                        model_identifier=gen_result["model_identifier"],
                        generation_params={
                            "length": req.desired_length,
                            "language": req.language,
                            "revision_instruction": sanitized_revision
                        }
                    )
                    email_id = db_gen.id
                except Exception as e:
                    logger.warning(f"Could not persist generation record: {e}")

            # Auto-save as user draft with its prompt in database
            try:
                draft_prompt = req.requirement
                if sanitized_revision:
                    draft_prompt = f"{req.requirement} [Revised: {sanitized_revision}]"
                db_draft = StorageService.save_draft(
                    db=db,
                    prompt=draft_prompt,
                    subject=gen_result["subject"],
                    body=gen_result["body"],
                    intent=req.representation.intent if req.representation else None,
                    tone=req.representation.tone if req.representation else None,
                    recipient=req.representation.recipient if req.representation else None,
                    user_id=user_id,
                    user_email=user_email
                )
                draft_id = db_draft.id
            except Exception as e:
                logger.warning(f"Could not auto-save draft to database: {e}")

        return GenerateResponse(
            subject=gen_result["subject"],
            body=gen_result["body"],
            system_version=gen_result["system_version"],
            model_identifier=gen_result["model_identifier"],
            generation_time_ms=gen_result["generation_time_ms"],
            raw_prompt_used=gen_result["raw_prompt_used"],
            email_id=email_id,
            draft_id=draft_id
        )

    def validate(self, req: ValidateRequest, db: Optional[Session] = None) -> OutputValidationReport:
        """Stage 3: VERIFY - evaluates generated email against user requirements."""
        sanitized_req = SecuritySanitizer.sanitize_input(req.requirement)

        report = self.validator.validate(
            requirement=sanitized_req,
            representation=req.representation,
            subject=req.email_subject,
            body=req.email_body
        )

        if db and req.generated_email_id:
            StorageService.save_validation(
                db=db,
                generated_email_id=req.generated_email_id,
                overall_status=report.overall_status.value,
                semantic_similarity=report.semantic_similarity,
                intent_preserved=report.intent_preserved,
                tone_suitable=report.tone_suitable,
                completeness_score=report.completeness_score,
                missing_details=report.missing_details,
                unsupported_details=report.unsupported_details,
                contradictions=report.contradictions,
                raw_report=report.model_dump()
            )

        return report
