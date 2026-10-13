"""
frontend/pages/page_assistant.py - Page 1: Email Assistant
Captures natural-language email requirements, optional user preferences,
and provides primary action controls (Analyze, Generate, Reset).
"""

import streamlit as st
import httpx
from backend.config import settings

API_BASE = f"http://{settings.BACKEND_HOST}:{settings.BACKEND_PORT}/api"


def render_page_assistant():
    st.header("✍️ Email Writing Assistant")
    st.markdown(
        "Enter your natural-language requirements below. The assistant will interpret your communication "
        "intent, construct a structured representation, draft a calibrated email, and validate factual fidelity."
    )

    # Preset examples for rapid demonstration
    example_options = {
        "-- Select an example or write your own --": "",
        "Academic Extension": "I could not complete my project because of a technical issue. Write a polite email to my professor requesting two additional days.",
        "Catching Up with a Friend": "I want to write to my friend Steve asking if he will be free on Friday to go outside and have a conversation about my life.",
        "Sick Leave Notice": "I have a fever and doctor advised rest. Email my manager Sarah informing her I will be taking sick leave today Monday and tomorrow Tuesday.",
        "Meeting Reschedule": "Our team needs to reschedule tomorrow's Q3 budget review call to Thursday 3 PM due to an urgent client audit. Send a courteous email to David.",
        "Billing Dispute": "I was billed twice ($49.99 each) for my monthly cloud subscription on Oct 10. Write a firm but polite message to billing support asking for a refund of the duplicate charge.",
        "Uncertain / Ambiguous Request": "Things happened and stuff is weird. Email someone to fix it ASAP."
    }

    selected_example = st.selectbox("💡 Quick-fill Example Scenarios:", list(example_options.keys()))
    if selected_example and example_options[selected_example]:
        default_req = example_options[selected_example]
    else:
        default_req = st.session_state.get("requirement_text", "")

    # Main natural language requirement
    requirement_text = st.text_area(
        "Natural Language Requirement (Mandatory):",
        value=default_req,
        height=120,
        placeholder="e.g. Write a polite email to my professor requesting two additional days due to a technical issue...",
        help="Specify the desired email in plain English. State recipient, reasons, dates, and actions if relevant."
    )
    st.session_state["requirement_text"] = requirement_text

    # Optional preferences collapsible accordion
    with st.expander("⚙️ Optional Preferences & Writing Constraints (All Optional)", expanded=False):
        col1, col2 = st.columns(2)
        with col1:
            recipient_hint = st.text_input(
                "Target Recipient Hint:",
                value=st.session_state.get("recipient_hint", ""),
                placeholder="e.g. Professor Smith, Sarah, Client"
            )
            tone_hint = st.selectbox(
                "Desired Tone:",
                ["Default (Auto-detect)", "Polite and Professional", "Firm and Polite", "Courteous and Urgent", "Apologetic and Professional", "Appreciative and Formal", "Friendly and Casual"],
                index=0
            )
            desired_length = st.selectbox(
                "Desired Length:",
                ["medium", "short", "long"],
                index=0
            )

        with col2:
            language = st.selectbox(
                "Target Language:",
                ["English", "Spanish", "French", "German"],
                index=0
            )
            context_text = st.text_area(
                "Background Context / Prior Thread:",
                value=st.session_state.get("context_text", ""),
                height=80,
                placeholder="Optional background context (e.g. Prior email thread or project background)..."
            )

        restrictions_input = st.text_input(
            "Explicit Restrictions (Negative Constraints):",
            value=st.session_state.get("restrictions_text", ""),
            placeholder="e.g. Do not invent a recipient name; do not promise weekend work",
            help="Constraints that the LLM must strictly obey."
        )

    # Action buttons bar
    st.markdown("---")
    btn_col1, btn_col2, btn_col3, btn_col4 = st.columns([1.5, 1.5, 1.5, 1])

    with btn_col1:
        if st.button("🔍 1. Analyze Requirement", use_container_width=True, type="secondary"):
            if not requirement_text.strip():
                st.warning("Please enter a requirement first.")
            else:
                with st.spinner("Performing semantic interpretation..."):
                    payload = {
                        "requirement": requirement_text.strip(),
                        "context": context_text.strip() if context_text else None,
                        "recipient_hint": recipient_hint.strip() if recipient_hint else None,
                        "tone_hint": None if tone_hint.startswith("Default") else tone_hint.lower(),
                        "desired_length": desired_length,
                        "language": language,
                        "explicit_restrictions": [r.strip() for r in restrictions_input.split(";") if r.strip()]
                    }
                    try:
                        resp = httpx.post(f"{API_BASE}/analyze", json=payload, timeout=20.0)
                        if resp.status_code == 200:
                            data = resp.json()
                            st.session_state["representation"] = data["representation"]
                            st.session_state["request_id"] = data["request_id"]
                            st.session_state["analysis_metadata"] = data
                            st.success("Semantic analysis complete! Review extracted attributes on the 'Semantic Analysis' tab.")
                        else:
                            st.error(f"Analysis failed: {resp.text}")
                    except Exception as e:
                        st.error(f"Cannot connect to Backend API: {e}")

    with btn_col2:
        if st.button("⚡ 2. Generate Email", use_container_width=True, type="primary"):
            if not requirement_text.strip():
                st.warning("Please enter a requirement first.")
            else:
                with st.spinner("Analyzing requirement and generating email..."):
                    # Step A: ensure representation exists
                    if "representation" not in st.session_state or not st.session_state["representation"]:
                        a_payload = {
                            "requirement": requirement_text.strip(),
                            "context": context_text.strip() if context_text else None,
                            "recipient_hint": recipient_hint.strip() if recipient_hint else None,
                            "tone_hint": None if tone_hint.startswith("Default") else tone_hint.lower(),
                            "desired_length": desired_length,
                            "language": language
                        }
                        try:
                            a_resp = httpx.post(f"{API_BASE}/analyze", json=a_payload, timeout=20.0)
                            if a_resp.status_code == 200:
                                a_data = a_resp.json()
                                st.session_state["representation"] = a_data["representation"]
                                st.session_state["request_id"] = a_data["request_id"]
                                st.session_state["analysis_metadata"] = a_data
                        except Exception as e:
                            st.error(f"Analysis failed: {e}")
                            st.stop()

                    # Step B: Generate email
                    g_payload = {
                        "requirement": requirement_text.strip(),
                        "representation": st.session_state["representation"],
                        "context": context_text.strip() if context_text else None,
                        "desired_length": desired_length,
                        "language": language,
                        "system_version": "proposed",
                        "request_id": st.session_state.get("request_id"),
                        "llm_provider": settings.LLM_PROVIDER,
                        "api_key": settings.GEMINI_API_KEY if settings.LLM_PROVIDER == "gemini" else None,
                        "model_name": settings.GEMINI_MODEL_NAME if settings.LLM_PROVIDER == "gemini" else None
                    }
                    try:
                        g_resp = httpx.post(f"{API_BASE}/generate", json=g_payload, timeout=35.0)
                        if g_resp.status_code == 200:
                            g_data = g_resp.json()
                            st.session_state["generated_subject"] = g_data["subject"]
                            st.session_state["generated_body"] = g_data["body"]
                            st.session_state["original_model_body"] = g_data["body"]
                            st.session_state["generated_email_id"] = g_data.get("email_id")
                            st.session_state["generation_metadata"] = g_data

                            # Step C: Automatically trigger validation
                            v_payload = {
                                "requirement": requirement_text.strip(),
                                "representation": st.session_state["representation"],
                                "email_subject": g_data["subject"],
                                "email_body": g_data["body"],
                                "generated_email_id": g_data.get("email_id")
                            }
                            v_resp = httpx.post(f"{API_BASE}/validate", json=v_payload, timeout=20.0)
                            if v_resp.status_code == 200:
                                st.session_state["validation_report"] = v_resp.json()

                            st.success("Email generated and validated! View output on the 'Generated Email' tab.")
                            st.info("💡 Not completely satisfied? Navigate to the **Generated Email** tab to **Redo & Refine** with 1-click presets or custom feedback.")
                            model_id = g_data.get("model_identifier", "Unknown")
                            if "gemini" in model_id.lower():
                                st.info(f"✨ Generated with Google Gemini (`{model_id}`)")
                            elif settings.LLM_PROVIDER == "gemini" and not settings.GEMINI_API_KEY:
                                st.warning("💡 Note: Generated with semantic fallback. Configure your GEMINI_API_KEY in the sidebar to enable live Gemini AI generation.")
                        else:
                            st.error(f"Generation failed: {g_resp.text}")
                    except Exception as e:
                        st.error(f"Generation error: {e}")

    with btn_col3:
        if st.button("🛡️ 3. Validate Current Email", use_container_width=True):
            if "generated_body" not in st.session_state or not st.session_state["generated_body"]:
                st.warning("Generate an email first or view the Generated Email tab.")
            else:
                with st.spinner("Validating factual and semantic alignment..."):
                    v_payload = {
                        "requirement": requirement_text.strip(),
                        "representation": st.session_state["representation"],
                        "email_subject": st.session_state.get("generated_subject", ""),
                        "email_body": st.session_state.get("generated_body", ""),
                        "generated_email_id": st.session_state.get("generated_email_id")
                    }
                    try:
                        v_resp = httpx.post(f"{API_BASE}/validate", json=v_payload, timeout=20.0)
                        if v_resp.status_code == 200:
                            st.session_state["validation_report"] = v_resp.json()
                            st.success("Validation updated! Inspect metrics on the 'Validation Results' tab.")
                    except Exception as e:
                        st.error(f"Validation request failed: {e}")

    with btn_col4:
        if st.button("🔄 Reset", use_container_width=True):
            st.session_state.clear()
            st.rerun()
