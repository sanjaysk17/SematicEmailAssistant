"""
backend/schemas/email_schemas.py - Pydantic Request, Response, and Semantic Schemas
Implements the dynamic structured communication representation, generation payloads,
and validation report schemas.
"""

from typing import List, Optional, Dict, Any, Union
from enum import Enum
from pydantic import BaseModel, Field


class ValidationStatusEnum(str, Enum):
    PASS = "PASS"
    NEEDS_REVIEW = "NEEDS_REVIEW"
    VALIDATION_ERROR = "VALIDATION_ERROR"


class FactStatusEnum(str, Enum):
    PRESENT = "Present"
    MISSING = "Missing"
    CONTRADICTED = "Contradicted"
    NOT_APPLICABLE = "Not applicable"
    UNCERTAIN = "Uncertain"


class StructuredRequirement(BaseModel):
    """
    Dynamic structured communication representation of user requirements.
    Extensible with custom domain attributes and uncertainties.
    """
    intent: str = Field(..., description="Primary intended communication action or goal")
    tone: str = Field(default="polite and professional", description="Intended tone and register")
    recipient: Optional[str] = Field(default=None, description="Audience or recipient designation")
    topic: Optional[str] = Field(default=None, description="Core subject matter of the email")
    required_action: Optional[str] = Field(default=None, description="Specific action requested from the recipient or taken by sender")
    reason: Optional[str] = Field(default=None, description="Justification or rationale provided by user")
    deadline: Optional[str] = Field(default=None, description="Specific deadline date or time if present")
    requested_duration: Optional[str] = Field(default=None, description="Requested duration or timeline extension")
    required_facts: List[str] = Field(default_factory=list, description="Explicit factual statements to preserve")
    constraints: List[str] = Field(default_factory=list, description="Negative constraints, boundaries, or instructions not to invent details")
    uncertainties: List[str] = Field(default_factory=list, description="Unresolved ambiguities or missing critical fields")
    inferred_details: Dict[str, Any] = Field(default_factory=dict, description="Inferred attributes not stated explicitly")
    clarification_questions: List[str] = Field(default_factory=list, description="Questions suggested to user if critical info is missing")
    user_confirmed: bool = Field(default=False, description="Whether the user reviewed and confirmed this representation")
    custom_attributes: Dict[str, Any] = Field(default_factory=dict, description="Arbitrary domain-specific metadata")


# API Schemas
class AnalyzeRequest(BaseModel):
    requirement: str = Field(..., min_length=3, max_length=5000, description="Natural language email instruction")
    context: Optional[str] = Field(default=None, max_length=5000, description="Optional background context or prior thread")
    recipient_hint: Optional[str] = Field(default=None, max_length=200, description="Explicit recipient preference")
    tone_hint: Optional[str] = Field(default=None, max_length=100, description="Explicit tone preference")
    desired_length: Optional[str] = Field(default="medium", description="Length: short, medium, long")
    language: Optional[str] = Field(default="English", description="Target language")
    explicit_restrictions: List[str] = Field(default_factory=list, description="Explicit negative constraints")
    api_key: Optional[str] = Field(default=None, description="Optional API key override for AI-driven analysis")
    llm_provider: Optional[str] = Field(default=None, description="Optional provider override")


class AnalyzeResponse(BaseModel):
    request_id: str
    original_requirement: str
    representation: StructuredRequirement
    model_identifier: str
    confidence: float
    analysis_time_ms: float
    warnings: List[str] = Field(default_factory=list)


class GenerateRequest(BaseModel):
    requirement: str = Field(..., min_length=3, description="Original requirement")
    representation: StructuredRequirement = Field(..., description="Confirmed or edited structured representation")
    context: Optional[str] = Field(default=None, description="Optional user context")
    desired_length: Optional[str] = Field(default="medium")
    language: Optional[str] = Field(default="English")
    system_version: Optional[str] = Field(default="proposed", description="proposed, baseline_a, or baseline_b")
    request_id: Optional[str] = None
    sender_name: Optional[str] = Field(default=None, description="Sender's real name to personalize signature")
    sender_role: Optional[str] = Field(default=None, description="Sender's title, major or role")
    sender_organization: Optional[str] = Field(default=None, description="Sender's company or university")
    sender_sign_off: Optional[str] = Field(default=None, description="Preferred closing/sign-off")
    sender_about_me: Optional[str] = Field(default=None, description="Personal background notes to remember about sender")
    llm_provider: Optional[str] = Field(default=None, description="Optional override provider: 'gemini', 'openai', 'local', 'mock'")
    api_key: Optional[str] = Field(default=None, description="Optional API key override")
    model_name: Optional[str] = Field(default=None, description="Optional model name override")
    revision_instruction: Optional[str] = Field(default=None, description="Specific revision instructions if user wants to redo/refine the draft")
    previous_draft: Optional[str] = Field(default=None, description="Previous draft subject and body that user was dissatisfied with")
    temperature: Optional[float] = Field(default=None, description="Optional temperature parameter for generation variation")


class ConfigUpdateRequest(BaseModel):
    llm_provider: Optional[str] = None
    gemini_api_key: Optional[str] = None
    gemini_model_name: Optional[str] = None
    openai_api_key: Optional[str] = None
    openai_model_name: Optional[str] = None


class ConfigResponse(BaseModel):
    llm_provider: str
    gemini_model_name: str
    has_gemini_key: bool
    has_openai_key: bool
    openai_model_name: str


class GenerateResponse(BaseModel):
    subject: str
    body: str
    system_version: str
    model_identifier: str
    generation_time_ms: float
    raw_prompt_used: Optional[str] = None
    email_id: Optional[str] = None
    draft_id: Optional[str] = None


class FactValidationItem(BaseModel):
    fact: str
    status: FactStatusEnum
    explanation: str


class OutputValidationReport(BaseModel):
    overall_status: ValidationStatusEnum
    intent_preserved: bool
    intent_explanation: str
    semantic_similarity: float
    similarity_model: str
    tone_suitable: bool
    tone_detected: str
    tone_explanation: str
    completeness_score: float
    fact_evaluations: List[FactValidationItem] = Field(default_factory=list)
    missing_details: List[str] = Field(default_factory=list)
    contradictions: List[str] = Field(default_factory=list)
    unsupported_details: List[str] = Field(default_factory=list)
    language_quality_issues: List[str] = Field(default_factory=list)
    actionable_warnings: List[str] = Field(default_factory=list)
    metric_definitions: Dict[str, str] = Field(default_factory=dict)
    validation_time_ms: float = 0.0


class ValidateRequest(BaseModel):
    requirement: str
    representation: StructuredRequirement
    email_subject: str
    email_body: str
    generated_email_id: Optional[str] = None


class HealthResponse(BaseModel):
    status: str
    version: str
    llm_provider: str
    database_status: str
    embedding_engine: str


# Authentication & User Persona Schemas
class DemoLoginRequest(BaseModel):
    username: Optional[str] = None
    email: Optional[str] = None
    name: Optional[str] = None


class RegisterRequest(BaseModel):
    email: str
    password: str
    name: str = "User"
    job_title: Optional[str] = None
    organization: Optional[str] = None
    preferred_sign_off: Optional[str] = None
    about_me: Optional[str] = None


class RegisterResponse(BaseModel):
    user_id: str
    email: str
    name: str
    totp_secret: str
    qr_code_data_url: str
    message: str


class Verify2FARequest(BaseModel):
    email: str
    totp_code: str


class LoginRequest(BaseModel):
    email: str
    password: str


class UserProfileResponse(BaseModel):
    id: str
    email: str
    name: str
    job_title: Optional[str] = None
    organization: Optional[str] = None
    preferred_sign_off: Optional[str] = None
    about_me: Optional[str] = None


class LoginResponse(BaseModel):
    requires_2fa: bool
    email: str
    access_token: Optional[str] = None
    user: Optional[UserProfileResponse] = None
    message: Optional[str] = None


class UpdateProfileRequest(BaseModel):
    name: Optional[str] = None
    job_title: Optional[str] = None
    organization: Optional[str] = None
    preferred_sign_off: Optional[str] = None
    about_me: Optional[str] = None


# User Draft Schemas
class CreateDraftRequest(BaseModel):
    prompt: str = Field(..., min_length=1, description="Original user requirement/prompt")
    subject: str = Field(..., description="Draft subject line")
    body: str = Field(..., description="Draft body text")
    intent: Optional[str] = None
    tone: Optional[str] = None
    recipient: Optional[str] = None
    user_id: Optional[str] = None
    user_email: Optional[str] = None
    draft_id: Optional[str] = None


class UpdateDraftRequest(BaseModel):
    prompt: Optional[str] = None
    subject: Optional[str] = None
    body: Optional[str] = None
    intent: Optional[str] = None
    tone: Optional[str] = None
    recipient: Optional[str] = None


class DraftResponse(BaseModel):
    id: str
    user_id: Optional[str] = None
    user_email: Optional[str] = None
    prompt: str
    subject: str
    body: str
    intent: Optional[str] = None
    tone: Optional[str] = None
    recipient: Optional[str] = None
    created_at: str
    updated_at: Optional[str] = None


class DraftListResponse(BaseModel):
    drafts: List[DraftResponse]
    total: int

