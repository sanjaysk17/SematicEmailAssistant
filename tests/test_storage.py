"""
tests/test_storage.py - Tests for Database Models and Privacy Persistence
Verifies table creation, relational integrity, and privacy persistence settings.
"""

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from backend.database.connection import Base
from backend.services.storage_service import StorageService
from backend.config import settings


@pytest.fixture
def test_db():
    engine = create_engine("sqlite:///:memory:", echo=False)
    Base.metadata.create_all(bind=engine)
    TestingSessionLocal = sessionmaker(bind=engine)
    session = TestingSessionLocal()
    yield session
    session.close()


def test_save_request_and_analysis(test_db):
    req = StorageService.save_request(
        db=test_db,
        raw_requirement="Write to professor",
        recipient_hint="Professor",
        tone_hint="polite"
    )
    assert req.id is not None

    analysis = StorageService.save_analysis(
        db=test_db,
        request_id=req.id,
        structured_representation={"intent": "request_extension"},
        intent="request_extension",
        tone="polite",
        uncertainties=[],
        model_identifier="test-model",
        confidence=0.9
    )
    assert analysis.id is not None
    assert analysis.request_id == req.id


def test_privacy_persistence_filter(test_db, monkeypatch):
    monkeypatch.setattr(settings, "DISABLE_EMAIL_CONTENT_PERSISTENCE", True)

    req = StorageService.save_request(test_db, raw_requirement="Private secret requirement")
    assert req.raw_requirement == "[PERSISTENCE_DISABLED_FOR_PRIVACY]"

    gen = StorageService.save_generation(
        test_db,
        request_id=req.id,
        analysis_id=None,
        subject="Secret Subject",
        body="Secret private message body",
        system_version="proposed",
        model_identifier="test-model",
        generation_params={}
    )
    assert gen.body == "[PERSISTENCE_DISABLED_FOR_PRIVACY]"


def test_user_draft_storage_lifecycle(test_db):
    # 1. Save draft with user prompt
    draft = StorageService.save_draft(
        db=test_db,
        prompt="Tell manager I will be late by 15 mins due to metro delay",
        subject="Brief delay for morning standup",
        body="Hi Alex, I will be 15 minutes late due to metro delay. Best, Sanjay",
        intent="Notify Delay",
        tone="professional",
        recipient="Alex",
        user_email="sanjay@example.com"
    )
    assert draft.id is not None
    assert "metro delay" in draft.prompt
    assert "Alex" in draft.body

    # 2. Retrieve draft
    drafts = StorageService.get_drafts(test_db, user_email="sanjay@example.com")
    assert len(drafts) == 1
    assert drafts[0].prompt == "Tell manager I will be late by 15 mins due to metro delay"
    assert drafts[0].subject == "Brief delay for morning standup"

    # 3. Update draft
    updated = StorageService.save_draft(
        db=test_db,
        prompt="Tell manager I will be late by 20 mins due to signal failure",
        subject="Updated: Brief delay for morning standup",
        body="Hi Alex, I will be 20 minutes late due to signal failure. Best, Sanjay",
        draft_id=draft.id
    )
    assert updated.id == draft.id
    assert "20 minutes" in updated.body

    # 4. Delete draft
    deleted = StorageService.delete_draft(test_db, draft.id)
    assert deleted is True
    assert StorageService.get_draft(test_db, draft.id) is None
