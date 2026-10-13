"""
backend/models/db_models.py - SQLAlchemy Relational Models
Defines tables for users, preferences, requests, analyses, generations,
validations, experiments, and experiment results.
"""

import uuid
from datetime import datetime, timezone
from sqlalchemy import (
    Column, String, Text, Boolean, Float, Integer, DateTime, ForeignKey, JSON
)
from sqlalchemy.orm import relationship
from backend.database.connection import Base


def get_utc_now():
    return datetime.now(timezone.utc)


def generate_uuid() -> str:
    return str(uuid.uuid4())


class User(Base):
    __tablename__ = "users"

    id = Column(String(36), primary_key=True, default=generate_uuid)
    username = Column(String(100), unique=True, nullable=False)
    email = Column(String(255), unique=True, nullable=False)
    name = Column(String(150), nullable=True, default="User")
    password_hash = Column(String(255), nullable=True)
    totp_secret = Column(String(64), nullable=True)
    totp_enabled = Column(Boolean, default=True)
    job_title = Column(String(150), nullable=True)
    organization = Column(String(150), nullable=True)
    preferred_sign_off = Column(String(150), nullable=True)
    about_me = Column(Text, nullable=True)
    created_at = Column(DateTime, default=get_utc_now)

    preferences = relationship("UserPreference", back_populates="user", uselist=False, cascade="all, delete-orphan")
    requests = relationship("EmailRequest", back_populates="user", cascade="all, delete-orphan")
    drafts = relationship("UserDraft", back_populates="user", cascade="all, delete-orphan")



class UserPreference(Base):
    __tablename__ = "user_preferences"

    id = Column(String(36), primary_key=True, default=generate_uuid)
    user_id = Column(String(36), ForeignKey("users.id"), nullable=False)
    preferred_tone = Column(String(50), default="polite and professional")
    preferred_language = Column(String(50), default="English")
    preferred_length = Column(String(50), default="medium")
    style_notes = Column(Text, nullable=True)
    updated_at = Column(DateTime, default=get_utc_now, onupdate=get_utc_now)

    user = relationship("User", back_populates="preferences")


class EmailRequest(Base):
    __tablename__ = "email_requests"

    id = Column(String(36), primary_key=True, default=generate_uuid)
    user_id = Column(String(36), ForeignKey("users.id"), nullable=True)
    raw_requirement = Column(Text, nullable=False)
    context = Column(Text, nullable=True)
    recipient_hint = Column(String(255), nullable=True)
    tone_hint = Column(String(100), nullable=True)
    created_at = Column(DateTime, default=get_utc_now)

    user = relationship("User", back_populates="requests")
    analyses = relationship("SemanticAnalysisRecord", back_populates="request", cascade="all, delete-orphan")
    generated_emails = relationship("GeneratedEmailRecord", back_populates="request", cascade="all, delete-orphan")


class SemanticAnalysisRecord(Base):
    __tablename__ = "semantic_analyses"

    id = Column(String(36), primary_key=True, default=generate_uuid)
    request_id = Column(String(36), ForeignKey("email_requests.id"), nullable=False)
    structured_representation = Column(JSON, nullable=False)
    intent = Column(String(100), nullable=False)
    tone = Column(String(100), nullable=False)
    uncertainties = Column(JSON, default=list)
    model_identifier = Column(String(100), default="semantic-engine-v1")
    confidence = Column(Float, default=1.0)
    created_at = Column(DateTime, default=get_utc_now)

    request = relationship("EmailRequest", back_populates="analyses")
    generated_emails = relationship("GeneratedEmailRecord", back_populates="analysis")


class GeneratedEmailRecord(Base):
    __tablename__ = "generated_emails"

    id = Column(String(36), primary_key=True, default=generate_uuid)
    request_id = Column(String(36), ForeignKey("email_requests.id"), nullable=False)
    analysis_id = Column(String(36), ForeignKey("semantic_analyses.id"), nullable=True)
    subject = Column(String(500), nullable=False)
    body = Column(Text, nullable=False)
    system_version = Column(String(50), default="proposed")
    model_identifier = Column(String(100), default="mock-llm")
    generation_params = Column(JSON, default=dict)
    created_at = Column(DateTime, default=get_utc_now)

    request = relationship("EmailRequest", back_populates="generated_emails")
    analysis = relationship("SemanticAnalysisRecord", back_populates="generated_emails")
    validations = relationship("ValidationResultRecord", back_populates="generated_email", cascade="all, delete-orphan")


class ValidationResultRecord(Base):
    __tablename__ = "validation_results"

    id = Column(String(36), primary_key=True, default=generate_uuid)
    generated_email_id = Column(String(36), ForeignKey("generated_emails.id"), nullable=False)
    overall_status = Column(String(50), nullable=False)  # PASS, NEEDS_REVIEW, VALIDATION_ERROR
    semantic_similarity = Column(Float, nullable=True)
    intent_preserved = Column(Boolean, default=True)
    tone_suitable = Column(Boolean, default=True)
    completeness_score = Column(Float, default=1.0)
    missing_details = Column(JSON, default=list)
    unsupported_details = Column(JSON, default=list)
    contradictions = Column(JSON, default=list)
    raw_report = Column(JSON, default=dict)
    created_at = Column(DateTime, default=get_utc_now)

    generated_email = relationship("GeneratedEmailRecord", back_populates="validations")


class Experiment(Base):
    __tablename__ = "experiments"

    id = Column(String(36), primary_key=True, default=generate_uuid)
    experiment_name = Column(String(150), nullable=False)
    description = Column(Text, nullable=True)
    baseline_type = Column(String(50), nullable=False)  # "all", "baseline_a", "baseline_b", "proposed"
    dataset_name = Column(String(100), default="synthetic_benchmark")
    sample_count = Column(Integer, default=0)
    config_params = Column(JSON, default=dict)
    created_at = Column(DateTime, default=get_utc_now)

    results = relationship("ExperimentResultRecord", back_populates="experiment", cascade="all, delete-orphan")


class ExperimentResultRecord(Base):
    __tablename__ = "experiment_results"

    id = Column(String(36), primary_key=True, default=generate_uuid)
    experiment_id = Column(String(36), ForeignKey("experiments.id"), nullable=False)
    example_id = Column(String(50), nullable=False)
    method = Column(String(50), nullable=False)  # "baseline_a", "baseline_b", "proposed"
    generated_subject = Column(String(500), nullable=False)
    generated_body = Column(Text, nullable=False)
    intent_score = Column(Float, default=0.0)
    completeness_score = Column(Float, default=0.0)
    semantic_similarity = Column(Float, default=0.0)
    tone_score = Column(Float, default=0.0)
    unsupported_count = Column(Integer, default=0)
    latency_ms = Column(Float, default=0.0)
    status = Column(String(50), default="PASS")
    created_at = Column(DateTime, default=get_utc_now)

    experiment = relationship("Experiment", back_populates="results")
 
 
class UserDraft(Base):
    __tablename__ = "user_drafts"

    id = Column(String(36), primary_key=True, default=generate_uuid)
    user_id = Column(String(36), ForeignKey("users.id"), nullable=True)
    user_email = Column(String(255), nullable=True)
    prompt = Column(Text, nullable=False)
    subject = Column(String(500), nullable=False)
    body = Column(Text, nullable=False)
    intent = Column(String(100), nullable=True)
    tone = Column(String(100), nullable=True)
    recipient = Column(String(255), nullable=True)
    created_at = Column(DateTime, default=get_utc_now)
    updated_at = Column(DateTime, default=get_utc_now, onupdate=get_utc_now)

    user = relationship("User", back_populates="drafts")
