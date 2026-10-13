"""
tests/test_api.py - Tests for FastAPI Endpoints
Tests health, analyze, generate, validate, and evaluate routes with TestClient.
"""

import pytest
from fastapi.testclient import TestClient
from backend.main import app


@pytest.fixture
def client():
    return TestClient(app)


def test_api_health(client):
    resp = client.get("/api/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "healthy"
    assert "llm_provider" in data


def test_api_analyze(client):
    payload = {
        "requirement": "Please write a polite email to my manager asking for sick leave tomorrow.",
        "recipient_hint": "Sarah"
    }
    resp = client.post("/api/analyze", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert "representation" in data
    assert "request_id" in data
    assert data["representation"]["recipient"] == "Sarah"


def test_api_generate_and_validate(client):
    # Step 1: Analyze
    a_resp = client.post("/api/analyze", json={
        "requirement": "I could not complete my project because of a technical issue. Write a polite email to my professor requesting two additional days."
    })
    assert a_resp.status_code == 200
    rep = a_resp.json()["representation"]

    # Step 2: Generate
    g_resp = client.post("/api/generate", json={
        "requirement": "I could not complete my project because of a technical issue. Write a polite email to my professor requesting two additional days.",
        "representation": rep,
        "system_version": "proposed"
    })
    assert g_resp.status_code == 200
    g_data = g_resp.json()
    assert len(g_data["subject"]) > 0
    assert len(g_data["body"]) > 0

    # Step 3: Validate
    v_resp = client.post("/api/validate", json={
        "requirement": "I could not complete my project because of a technical issue. Write a polite email to my professor requesting two additional days.",
        "representation": rep,
        "email_subject": g_data["subject"],
        "email_body": g_data["body"]
    })
    assert v_resp.status_code == 200
    v_data = v_resp.json()
    assert "overall_status" in v_data
    assert "completeness_score" in v_data


def test_api_invalid_empty_requirement(client):
    resp = client.post("/api/analyze", json={"requirement": "a"})  # min length 3
    assert resp.status_code == 422


def test_api_drafts_crud(client):
    # 1. Create a draft with user prompt
    create_payload = {
        "prompt": "Schedule meeting with John for Q4 project planning on Friday",
        "subject": "Q4 Project Planning Meeting",
        "body": "Hi John,\n\nCould we meet this Friday to discuss Q4 project planning?\n\nBest,\nSanjay",
        "intent": "Schedule Meeting",
        "tone": "polite and professional",
        "recipient": "John",
        "user_email": "sanjay@example.com"
    }
    create_resp = client.post("/api/drafts", json=create_payload)
    assert create_resp.status_code == 200
    draft_data = create_resp.json()
    assert draft_data["id"] is not None
    assert draft_data["prompt"] == create_payload["prompt"]
    assert draft_data["subject"] == create_payload["subject"]
    draft_id = draft_data["id"]

    # 2. List drafts
    list_resp = client.get("/api/drafts")
    assert list_resp.status_code == 200
    list_data = list_resp.json()
    assert list_data["total"] >= 1
    found = any(d["id"] == draft_id for d in list_data["drafts"])
    assert found is True

    # 3. Get single draft
    get_resp = client.get(f"/api/drafts/{draft_id}")
    assert get_resp.status_code == 200
    assert get_resp.json()["id"] == draft_id
    assert get_resp.json()["prompt"] == create_payload["prompt"]

    # 4. Update draft
    update_resp = client.put(f"/api/drafts/{draft_id}", json={
        "subject": "Updated: Q4 Project Planning Meeting",
        "body": "Hi John, Updated time: let's meet Friday 2 PM. Best, Sanjay"
    })
    assert update_resp.status_code == 200
    assert update_resp.json()["subject"] == "Updated: Q4 Project Planning Meeting"

    # 5. Delete draft
    del_resp = client.delete(f"/api/drafts/{draft_id}")
    assert del_resp.status_code == 200
    assert del_resp.json()["success"] is True

    # Verify deletion
    assert client.get(f"/api/drafts/{draft_id}").status_code == 404


def test_api_auth_dynamic_username(client):
    # Test demo login with custom username provided by user
    custom_user_payload = {
        "username": "alex_developer",
        "email": "alex@company.com",
        "name": "Alex"
    }
    resp = client.post("/api/auth/demo-login", json=custom_user_payload)
    assert resp.status_code == 200
    data = resp.json()
    assert data["user"]["name"] == "Alex"
    assert data["user"]["email"] == "alex@company.com"
    assert "Alex" in data["message"]

    # Test update profile with another general name
    token = data["access_token"]
    update_resp = client.put(
        "/api/auth/profile",
        headers={"Authorization": f"Bearer {token}"},
        json={"name": "Morgan Taylor", "job_title": "Lead Researcher"}
    )
    assert update_resp.status_code == 200
    assert update_resp.json()["name"] == "Morgan Taylor"
    assert update_resp.json()["job_title"] == "Lead Researcher"


def test_api_user_isolation_drafts_and_profile(client):
    # 1. Verify unauthenticated profile access is rejected with 401
    unauth_resp = client.get("/api/auth/profile")
    assert unauth_resp.status_code == 401

    # 2. Login as User A
    login_a = client.post("/api/auth/demo-login", json={
        "username": "user_a",
        "email": "user_a@example.com",
        "name": "Alice"
    }).json()
    token_a = login_a["access_token"]
    user_a_id = login_a["user"]["id"]

    # 3. Login as User B
    login_b = client.post("/api/auth/demo-login", json={
        "username": "user_b",
        "email": "user_b@example.com",
        "name": "Bob"
    }).json()
    token_b = login_b["access_token"]
    user_b_id = login_b["user"]["id"]

    # 4. User A's profile only shows Alice's personal info
    prof_a = client.get("/api/auth/profile", headers={"Authorization": f"Bearer {token_a}"}).json()
    assert prof_a["name"] == "Alice"
    assert prof_a["email"] == "user_a@example.com"

    # User B's profile only shows Bob's personal info
    prof_b = client.get("/api/auth/profile", headers={"Authorization": f"Bearer {token_b}"}).json()
    assert prof_b["name"] == "Bob"
    assert prof_b["email"] == "user_b@example.com"

    # 5. User A creates a confidential draft
    draft_a = client.post(
        "/api/drafts",
        headers={"Authorization": f"Bearer {token_a}"},
        json={
            "prompt": "Confidential meeting prompt for Alice",
            "subject": "Alice Confidential Subject",
            "body": "Body only for Alice"
        }
    ).json()
    draft_a_id = draft_a["id"]

    # 6. User B lists drafts - User A's draft MUST NOT appear in User B's list
    list_b = client.get("/api/drafts", headers={"Authorization": f"Bearer {token_b}"}).json()
    assert all(d["id"] != draft_a_id for d in list_b["drafts"])

    # 7. User A lists drafts - User A's draft MUST appear
    list_a = client.get("/api/drafts", headers={"Authorization": f"Bearer {token_a}"}).json()
    assert any(d["id"] == draft_a_id for d in list_a["drafts"])

    # 8. User B attempts to access User A's draft directly -> 403 Forbidden
    direct_get = client.get(f"/api/drafts/{draft_a_id}", headers={"Authorization": f"Bearer {token_b}"})
    assert direct_get.status_code == 403

    # 9. User B attempts to update User A's draft -> 403 Forbidden
    direct_put = client.put(
        f"/api/drafts/{draft_a_id}",
        headers={"Authorization": f"Bearer {token_b}"},
        json={"subject": "Hacked Subject"}
    )
    assert direct_put.status_code == 403

    # 10. User B attempts to delete User A's draft -> 403 Forbidden
    direct_del = client.delete(f"/api/drafts/{draft_a_id}", headers={"Authorization": f"Bearer {token_b}"})
    assert direct_del.status_code == 403

    # Verify draft still exists untouched for User A
    get_alice = client.get(f"/api/drafts/{draft_a_id}", headers={"Authorization": f"Bearer {token_a}"})
    assert get_alice.status_code == 200
    assert get_alice.json()["subject"] == "Alice Confidential Subject"

