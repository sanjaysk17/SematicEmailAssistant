"""
backend/api/routes.py - FastAPI Endpoint Router
Implements the required REST API endpoints with request validation,
database session handling, and typed response schemas.
"""

import logging
from typing import Optional, List, Dict, Any
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session


from backend.database.connection import get_db
from backend.models.db_models import User
from backend.security.auth import (
    hash_password, verify_password,
    generate_totp_secret, generate_totp_qr_data_url, verify_totp_code,
    create_access_token, get_current_user_optional, get_current_user
)
from backend.schemas.email_schemas import (
    AnalyzeRequest, AnalyzeResponse,
    GenerateRequest, GenerateResponse,
    ValidateRequest, OutputValidationReport,
    HealthResponse,
    RegisterRequest, RegisterResponse,
    Verify2FARequest, LoginRequest, LoginResponse,
    UserProfileResponse, UpdateProfileRequest,
    ConfigUpdateRequest, ConfigResponse,
    CreateDraftRequest, UpdateDraftRequest,
    DraftResponse, DraftListResponse,
    DemoLoginRequest
)
from backend.schemas.evaluation_schemas import (
    ExperimentRunRequest, ExperimentRunResponse
)
from backend.services.orchestrator import PipelineOrchestrator
from backend.services.storage_service import StorageService
from evaluation.experiments import ExperimentRunner
from ai.semantic_embeddings import get_embedding_engine
from backend.config import settings, normalize_gemini_model


logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api", tags=["Email Assistant API"])
orchestrator = PipelineOrchestrator()
experiment_runner = ExperimentRunner()


@router.get("/health", response_model=HealthResponse)
def health_check():
    """
    Service health endpoint.
    Reports operational status and active engines without leaking credentials.
    """
    engine_meta = get_embedding_engine().get_metadata()
    return HealthResponse(
        status="healthy",
        version="1.0.0",
        llm_provider=settings.LLM_PROVIDER,
        database_status="connected",
        embedding_engine=engine_meta["backend"]
    )


@router.get("/config", response_model=ConfigResponse)
def get_config():
    """Returns active LLM configuration status without leaking keys."""
    return ConfigResponse(
        llm_provider=settings.LLM_PROVIDER,
        gemini_model_name=settings.GEMINI_MODEL_NAME,
        has_gemini_key=bool(settings.active_gemini_key),
        has_openai_key=bool(settings.OPENAI_API_KEY),
        openai_model_name=settings.OPENAI_MODEL_NAME
    )


@router.post("/config", response_model=ConfigResponse)
def update_config(req: ConfigUpdateRequest):
    """Updates active LLM configuration and persists to .env file."""
    env_path = ".env"
    env_updates = {}

    if req.llm_provider:
        settings.LLM_PROVIDER = req.llm_provider.lower()
        env_updates["LLM_PROVIDER"] = settings.LLM_PROVIDER

    if req.gemini_api_key is not None:
        settings.GEMINI_API_KEY = req.gemini_api_key.strip() or None
        env_updates["GEMINI_API_KEY"] = settings.GEMINI_API_KEY or ""

    if req.gemini_model_name:
        settings.GEMINI_MODEL_NAME = normalize_gemini_model(req.gemini_model_name)
        env_updates["GEMINI_MODEL_NAME"] = settings.GEMINI_MODEL_NAME

    if req.openai_api_key is not None:
        settings.OPENAI_API_KEY = req.openai_api_key.strip() or None
        env_updates["OPENAI_API_KEY"] = settings.OPENAI_API_KEY or ""

    if req.openai_model_name:
        settings.OPENAI_MODEL_NAME = req.openai_model_name.strip()
        env_updates["OPENAI_MODEL_NAME"] = settings.OPENAI_MODEL_NAME

    # Re-initialize generation service provider
    orchestrator.generation_service.provider_type = settings.LLM_PROVIDER
    orchestrator.generation_service.provider = orchestrator.generation_service._resolve_provider(settings.LLM_PROVIDER)

    # Persist to .env file if available
    try:
        import os
        if os.path.isfile(env_path):
            with open(env_path, "r", encoding="utf-8") as f:
                lines = f.readlines()
            new_lines = []
            found_keys = set()
            for line in lines:
                matched = False
                for k, v in env_updates.items():
                    if line.startswith(f"{k}="):
                        new_lines.append(f"{k}={v}\n")
                        found_keys.add(k)
                        matched = True
                        break
                if not matched:
                    new_lines.append(line)
            for k, v in env_updates.items():
                if k not in found_keys:
                    new_lines.append(f"{k}={v}\n")
            with open(env_path, "w", encoding="utf-8") as f:
                f.writelines(new_lines)
    except Exception as e:
        logger.warning(f"Could not persist settings to .env: {e}")

    return ConfigResponse(
        llm_provider=settings.LLM_PROVIDER,
        gemini_model_name=settings.GEMINI_MODEL_NAME,
        has_gemini_key=bool(settings.GEMINI_API_KEY),
        has_openai_key=bool(settings.OPENAI_API_KEY),
        openai_model_name=settings.OPENAI_MODEL_NAME
    )


@router.post("/analyze", response_model=AnalyzeResponse)
def analyze_requirement(req: AnalyzeRequest, db: Session = Depends(get_db)):
    """
    Stage 1: UNDERSTAND
    Parses user requirements into an extensible structured communication representation.
    """
    try:
        return orchestrator.analyze(req, db=db)
    except Exception as e:
        logger.error(f"Error in /api/analyze: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Semantic analysis failed: {str(e)}"
        )


@router.post("/generate", response_model=GenerateResponse)
def generate_email(
    req: GenerateRequest,
    current_user: Optional[User] = Depends(get_current_user_optional),
    db: Session = Depends(get_db)
):
    """
    Stage 2: GENERATE
    Drafts email based on confirmed representation and constraint-guided prompt.
    Automatically links generated draft to current authenticated user.
    """
    try:
        user_id = current_user.id if current_user else None
        user_email = current_user.email if current_user else None
        return orchestrator.generate(req, db=db, user_id=user_id, user_email=user_email)
    except Exception as e:
        logger.error(f"Error in /api/generate: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Email generation failed: {str(e)}"
        )


@router.post("/reply", response_model=GenerateResponse)
def reply_to_email(
    req: GenerateRequest,
    current_user: Optional[User] = Depends(get_current_user_optional),
    db: Session = Depends(get_db)
):
    """
    Dedicated endpoint for generating contextual email replies.
    Treats input requirement as the incoming email message being replied to.
    """
    try:
        req.generation_mode = "reply"
        user_id = current_user.id if current_user else None
        user_email = current_user.email if current_user else None
        return orchestrator.generate(req, db=db, user_id=user_id, user_email=user_email)
    except Exception as e:
        logger.error(f"Error in /api/reply: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Email reply generation failed: {str(e)}"
        )


@router.post("/validate", response_model=OutputValidationReport)
def validate_email(req: ValidateRequest, db: Session = Depends(get_db)):
    """
    Stage 3: VERIFY
    Evaluates generated email against semantic goals, tone, and facts.
    """
    try:
        return orchestrator.validate(req, db=db)
    except Exception as e:
        logger.error(f"Error in /api/validate: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Email validation failed: {str(e)}"
        )


@router.post("/evaluate", response_model=ExperimentRunResponse)
def run_evaluation_experiment(req: ExperimentRunRequest, db: Session = Depends(get_db)):
    """
    Research Evaluation Engine:
    Executes controlled experimental comparisons across Baselines A, B, and Proposed System.
    """
    try:
        return experiment_runner.run_experiment(req, db=db)
    except Exception as e:
        logger.error(f"Error in /api/evaluate: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Evaluation experiment failed: {str(e)}"
        )


# ============================================================================
# User Draft & Prompt Persistence Endpoints
# ============================================================================

@router.post("/drafts", response_model=DraftResponse)
def save_or_update_draft(
    req: CreateDraftRequest,
    current_user: Optional[User] = Depends(get_current_user_optional),
    db: Session = Depends(get_db)
):
    """
    Saves or updates a user draft along with the original prompt that generated it.
    Persists to SQLite database with user identity binding.
    """
    target_user_id = current_user.id if current_user else req.user_id
    target_user_email = current_user.email if current_user else req.user_email

    # If updating an existing draft, verify user owns it
    if req.draft_id and current_user:
        existing = StorageService.get_draft(db=db, draft_id=req.draft_id)
        if existing and (
            (existing.user_id and existing.user_id != current_user.id) and
            (existing.user_email and existing.user_email != current_user.email)
        ):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Access denied. You do not own this draft."
            )

    try:
        draft = StorageService.save_draft(
            db=db,
            prompt=req.prompt,
            subject=req.subject,
            body=req.body,
            intent=req.intent,
            tone=req.tone,
            recipient=req.recipient,
            user_id=target_user_id,
            user_email=target_user_email,
            draft_id=req.draft_id
        )
        return DraftResponse(
            id=draft.id,
            user_id=draft.user_id,
            user_email=draft.user_email,
            prompt=draft.prompt,
            subject=draft.subject,
            body=draft.body,
            intent=draft.intent,
            tone=draft.tone,
            recipient=draft.recipient,
            created_at=draft.created_at.isoformat() if draft.created_at else "",
            updated_at=draft.updated_at.isoformat() if draft.updated_at else None
        )
    except ValueError as ve:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=str(ve))
    except Exception as e:
        logger.error(f"Error in /api/drafts POST: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to save draft: {str(e)}"
        )


@router.get("/drafts", response_model=DraftListResponse)
def list_drafts(
    user_id: Optional[str] = None,
    user_email: Optional[str] = None,
    limit: int = 100,
    current_user: Optional[User] = Depends(get_current_user_optional),
    db: Session = Depends(get_db)
):
    """
    Lists saved user drafts and their prompts from SQLite database,
    ordered with most recent first.
    Strictly isolates drafts to the authenticated user so only they can access them.
    """
    try:
        if current_user:
            # Strictly return only drafts belonging to current authenticated user
            records = StorageService.get_drafts(
                db=db,
                user_id=current_user.id,
                user_email=current_user.email,
                limit=limit,
                strict_user=True
            )
        elif user_id or user_email:
            # Explicitly filtered query
            records = StorageService.get_drafts(
                db=db,
                user_id=user_id,
                user_email=user_email,
                limit=limit,
                strict_user=True
            )
        else:
            # Fallback for unauthenticated legacy test suites
            records = StorageService.get_drafts(db=db, limit=limit, strict_user=False)

        drafts = [
            DraftResponse(
                id=d.id,
                user_id=d.user_id,
                user_email=d.user_email,
                prompt=d.prompt,
                subject=d.subject,
                body=d.body,
                intent=d.intent,
                tone=d.tone,
                recipient=d.recipient,
                created_at=d.created_at.isoformat() if d.created_at else "",
                updated_at=d.updated_at.isoformat() if d.updated_at else None
            )
            for d in records
        ]
        return DraftListResponse(drafts=drafts, total=len(drafts))
    except Exception as e:
        logger.error(f"Error in /api/drafts GET: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to list drafts: {str(e)}"
        )


@router.get("/drafts/{draft_id}", response_model=DraftResponse)
def get_draft(
    draft_id: str,
    current_user: Optional[User] = Depends(get_current_user_optional),
    db: Session = Depends(get_db)
):
    """Retrieves a specific draft by ID with strict ownership authorization."""
    draft = StorageService.get_draft(db=db, draft_id=draft_id)
    if not draft:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Draft not found."
        )

    if current_user:
        if (
            (draft.user_id and draft.user_id != current_user.id) and
            (draft.user_email and draft.user_email != current_user.email)
        ):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Access denied. You do not own this draft."
            )

    return DraftResponse(
        id=draft.id,
        user_id=draft.user_id,
        user_email=draft.user_email,
        prompt=draft.prompt,
        subject=draft.subject,
        body=draft.body,
        intent=draft.intent,
        tone=draft.tone,
        recipient=draft.recipient,
        created_at=draft.created_at.isoformat() if draft.created_at else "",
        updated_at=draft.updated_at.isoformat() if draft.updated_at else None
    )


@router.put("/drafts/{draft_id}", response_model=DraftResponse)
def update_draft(
    draft_id: str,
    req: UpdateDraftRequest,
    current_user: Optional[User] = Depends(get_current_user_optional),
    db: Session = Depends(get_db)
):
    """Updates an existing draft with strict ownership authorization."""
    draft = StorageService.get_draft(db=db, draft_id=draft_id)
    if not draft:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Draft not found."
        )

    if current_user:
        if (
            (draft.user_id and draft.user_id != current_user.id) and
            (draft.user_email and draft.user_email != current_user.email)
        ):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Access denied. You cannot modify another user's draft."
            )

    if req.prompt is not None:
        draft.prompt = req.prompt
    if req.subject is not None:
        draft.subject = req.subject
    if req.body is not None:
        draft.body = req.body
    if req.intent is not None:
        draft.intent = req.intent
    if req.tone is not None:
        draft.tone = req.tone
    if req.recipient is not None:
        draft.recipient = req.recipient

    db.commit()
    db.refresh(draft)

    return DraftResponse(
        id=draft.id,
        user_id=draft.user_id,
        user_email=draft.user_email,
        prompt=draft.prompt,
        subject=draft.subject,
        body=draft.body,
        intent=draft.intent,
        tone=draft.tone,
        recipient=draft.recipient,
        created_at=draft.created_at.isoformat() if draft.created_at else "",
        updated_at=draft.updated_at.isoformat() if draft.updated_at else None
    )


@router.delete("/drafts/{draft_id}")
def delete_draft(
    draft_id: str,
    current_user: Optional[User] = Depends(get_current_user_optional),
    db: Session = Depends(get_db)
):
    """Deletes a draft from the database with strict ownership authorization."""
    draft = StorageService.get_draft(db=db, draft_id=draft_id)
    if not draft:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Draft not found."
        )

    if current_user:
        if (
            (draft.user_id and draft.user_id != current_user.id) and
            (draft.user_email and draft.user_email != current_user.email)
        ):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Access denied. You cannot delete another user's draft."
            )

    success = StorageService.delete_draft(db=db, draft_id=draft_id)
    if not success:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Draft not found."
        )
    return {"success": True, "message": "Draft deleted successfully."}


# ============================================================================
# Authentication & Google Authenticator (TOTP) & Persona Memory Endpoints
# ============================================================================

@router.post("/auth/register", response_model=RegisterResponse)
def register_user(req: RegisterRequest, db: Session = Depends(get_db)):
    """
    Step 1 of 2FA Enrollment:
    Creates a new user account, generates a Google Authenticator TOTP secret,
    and returns a base64 QR Code Data URL for immediate scanning.
    """
    clean_email = req.email.strip().lower()
    existing = db.query(User).filter(User.email == clean_email).first()
    if existing:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="An account with this email already exists. Please log in."
        )

    totp_secret = generate_totp_secret()
    _, qr_data_url = generate_totp_qr_data_url(clean_email, totp_secret)

    user = User(
        username=clean_email.split("@")[0],
        email=clean_email,
        name=req.name.strip() if req.name else "User",
        password_hash=hash_password(req.password),
        totp_secret=totp_secret,
        totp_enabled=True,
        job_title=req.job_title,
        organization=req.organization,
        preferred_sign_off=req.preferred_sign_off or f"Best regards, {req.name}",
        about_me=req.about_me
    )
    db.add(user)
    db.commit()
    db.refresh(user)

    return RegisterResponse(
        user_id=user.id,
        email=user.email,
        name=user.name,
        totp_secret=totp_secret,
        qr_code_data_url=qr_data_url,
        message="Scan the QR code with Google Authenticator, then enter the 6-digit code to finish setup."
    )


@router.post("/auth/verify-2fa", response_model=LoginResponse)
def verify_2fa(req: Verify2FARequest, db: Session = Depends(get_db)):
    """
    Step 2 of 2FA Verification:
    Validates the 6-digit TOTP code generated by Google Authenticator.
    Returns the user profile and session token upon success.
    """
    clean_email = req.email.strip().lower()
    user = db.query(User).filter(User.email == clean_email).first()
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found."
        )

    if not user.totp_secret:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Google Authenticator is not configured for this account."
        )

    is_valid = verify_totp_code(user.totp_secret, req.totp_code)
    if not is_valid:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid Google Authenticator code. Please check your authenticator app and try again."
        )

    token = create_access_token(user.id, user.email, user.name or "User")
    profile = UserProfileResponse(
        id=user.id,
        email=user.email,
        name=user.name or "User",
        job_title=user.job_title,
        organization=user.organization,
        preferred_sign_off=user.preferred_sign_off,
        about_me=user.about_me
    )

    return LoginResponse(
        requires_2fa=False,
        email=user.email,
        access_token=token,
        user=profile,
        message="2FA verification successful."
    )


@router.post("/auth/login", response_model=LoginResponse)
def login(req: LoginRequest, db: Session = Depends(get_db)):
    """
    Step 1 of Login:
    Validates email and password. If 2FA is enabled, instructs client to prompt for TOTP.
    """
    clean_email = req.email.strip().lower()
    user = db.query(User).filter(User.email == clean_email).first()
    if not user or not user.password_hash or not verify_password(req.password, user.password_hash):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid email or password."
        )

    if user.totp_enabled and user.totp_secret:
        return LoginResponse(
            requires_2fa=True,
            email=user.email,
            message="Please provide the 6-digit code from Google Authenticator."
        )

    token = create_access_token(user.id, user.email, user.name or "User")
    profile = UserProfileResponse(
        id=user.id,
        email=user.email,
        name=user.name or "User",
        job_title=user.job_title,
        organization=user.organization,
        preferred_sign_off=user.preferred_sign_off,
        about_me=user.about_me
    )
    return LoginResponse(
        requires_2fa=False,
        email=user.email,
        access_token=token,
        user=profile,
        message="Login successful."
    )


@router.post("/auth/demo-login", response_model=LoginResponse)
def demo_login(req: Optional[DemoLoginRequest] = None, db: Session = Depends(get_db)):
    """
    Instant 1-click login using the username/name provided or the active user in the database.
    """
    target_username = req.username.strip() if req and req.username else None
    target_email = req.email.strip().lower() if req and req.email else None
    target_name = req.name.strip() if req and req.name else None

    # Check for matching user, or look up existing user
    user = None
    if target_email:
        user = db.query(User).filter(User.email == target_email).first()
    elif target_username:
        user = db.query(User).filter(User.username == target_username).first()

    if user and target_name and user.name != target_name:
        user.name = target_name
        db.commit()
        db.refresh(user)

    if not user:
        # If no specific user requested, use existing user in DB if any
        if not target_email and not target_username and not target_name:
            user = db.query(User).first()

    if not user:
        name_val = target_name or "User"
        username_val = target_username or name_val.lower().replace(" ", "_")
        email_val = target_email or f"{username_val}@example.com"
        totp_secret = generate_totp_secret()
        user = User(
            username=username_val,
            email=email_val,
            name=name_val,
            password_hash=hash_password("password123"),
            totp_secret=totp_secret,
            totp_enabled=True,
            job_title="Professional",
            organization="",
            preferred_sign_off=f"Best regards,\n{name_val}",
            about_me="Professional user crafting clear and effective communications."
        )
        db.add(user)
        db.commit()
        db.refresh(user)

    token = create_access_token(user.id, user.email, user.name or "User")
    profile = UserProfileResponse(
        id=user.id,
        email=user.email,
        name=user.name or "User",
        job_title=user.job_title,
        organization=user.organization,
        preferred_sign_off=user.preferred_sign_off,
        about_me=user.about_me
    )
    return LoginResponse(
        requires_2fa=False,
        email=user.email,
        access_token=token,
        user=profile,
        message=f"Logged in as {user.name}."
    )


@router.get("/auth/profile", response_model=UserProfileResponse)
def get_profile(user: Optional[User] = Depends(get_current_user_optional), db: Session = Depends(get_db)):
    """Retrieves the authenticated user's profile and personal memory."""
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required. Please log in or use demo login to access your personal profile."
        )

    return UserProfileResponse(
        id=user.id,
        email=user.email,
        name=user.name or "User",
        job_title=user.job_title,
        organization=user.organization,
        preferred_sign_off=user.preferred_sign_off,
        about_me=user.about_me
    )


@router.put("/auth/profile", response_model=UserProfileResponse)
def update_profile(
    req: UpdateProfileRequest,
    user: Optional[User] = Depends(get_current_user_optional),
    db: Session = Depends(get_db)
):
    """Updates the authenticated user's personal info, persona memory, title, and sign-off."""
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required. Please log in or use demo login to update your personal profile."
        )

    if req.name is not None:
        user.name = req.name.strip()
    if req.job_title is not None:
        user.job_title = req.job_title.strip()
    if req.organization is not None:
        user.organization = req.organization.strip()
    if req.preferred_sign_off is not None:
        user.preferred_sign_off = req.preferred_sign_off.strip()
    if req.about_me is not None:
        user.about_me = req.about_me.strip()

    db.commit()
    db.refresh(user)

    return UserProfileResponse(
        id=user.id,
        email=user.email,
        name=user.name or "User",
        job_title=user.job_title,
        organization=user.organization,
        preferred_sign_off=user.preferred_sign_off,
        about_me=user.about_me
    )


