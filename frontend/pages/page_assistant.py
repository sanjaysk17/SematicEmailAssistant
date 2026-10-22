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

    # Professional Template Library & Quick Scenarios
    templates_catalog = {
        "-- Select a Professional Template or write custom prompt --": None,
        "💼 Executive: Project Milestone & Status Update": {
            "req": "Please draft a concise executive progress report to my department head summarizing our team's completion of Phase 2 deliverables, system stability, and upcoming roadmap milestones for next quarter.",
            "recipient": "Department Head",
            "tone": "Polite and Professional",
            "length": "Medium (Standard - Min 130 words / 800 chars)"
        },
        "💼 Executive: Meeting Reschedule with Agenda": {
            "req": "Our team needs to reschedule tomorrow's Q3 budget review call to Thursday 3 PM due to an urgent client audit. Send a courteous email to David.",
            "recipient": "David",
            "tone": "Courteous and Urgent (Deadlines / Critical)",
            "length": "Medium (Standard - Min 130 words / 800 chars)"
        },
        "💼 Executive: Infrastructure Budget Proposal": {
            "req": "Submit a formal proposal to director requesting $15,000 additional budget for cloud infrastructure upgrades to support user growth.",
            "recipient": "Director",
            "tone": "Assertive and Direct (Executive / Clear Boundaries)",
            "length": "Medium (Standard - Min 130 words / 800 chars)"
        },
        "🤝 Client: Billing Duplicate Charge Refund": {
            "req": "I was billed twice ($49.99 each) for my monthly cloud subscription on Oct 10. Write a firm but polite message to billing support asking for a refund of the duplicate charge.",
            "recipient": "Billing Support",
            "tone": "Firm and Polite (Disputes / Follow-ups)",
            "length": "Medium (Standard - Min 130 words / 800 chars)"
        },
        "🤝 Client: Contract Renewal & Multi-Year Discount": {
            "req": "Negotiate annual software contract renewal with account manager Rachel, requesting a 15% multi-year discount and revised SLA terms.",
            "recipient": "Rachel",
            "tone": "Diplomatic and Respectful (Sensitive Discussions)",
            "length": "Medium (Standard - Min 130 words / 800 chars)"
        },
        "🤝 Client: Proposal Follow-up & Walkthrough": {
            "req": "Follow up with client Mark on the enterprise software proposal submitted last Monday. Inquire if they have reviewed it and offer a 15-minute walkthrough.",
            "recipient": "Mark",
            "tone": "Persuasive and Enthusiastic (Proposals / Pitches)",
            "length": "Medium (Standard - Min 130 words / 800 chars)"
        },
        "🏥 Medical / Leave: Sick Leave Notice": {
            "req": "I have a fever and doctor advised rest. Email my manager Sarah informing her I will be taking sick leave today Monday and tomorrow Tuesday.",
            "recipient": "Sarah",
            "tone": "Polite and Professional",
            "length": "Short (Concise - Min 60 words / 350 chars)"
        },
        "🏥 Medical / Leave: Planned Vacation / PTO Notice": {
            "req": "Submit a formal vacation leave notice to manager Michael for next week Monday to Friday, confirming handover of active tasks to Sarah.",
            "recipient": "Michael",
            "tone": "Polite and Professional",
            "length": "Short (Concise - Min 60 words / 350 chars)"
        },
        "🎓 Academic: Project Deadline Extension": {
            "req": "I could not complete my project because of a technical issue. Write a polite email to my professor requesting two additional days.",
            "recipient": "Professor",
            "tone": "Polite and Professional",
            "length": "Medium (Standard - Min 130 words / 800 chars)"
        },
        "🎓 Academic: Letter of Recommendation Request": {
            "req": "Write a respectful email to Professor Davis requesting a letter of recommendation for a graduate fellowship application before November 15th.",
            "recipient": "Professor Davis",
            "tone": "Appreciative and Formal (Thank You / Recognition)",
            "length": "Medium (Standard - Min 130 words / 800 chars)"
        },
        "☕ Networking: Catching Up with a Colleague/Friend": {
            "req": "I want to write to my friend Steve asking if he will be free on Friday to go outside and have a conversation about my life.",
            "recipient": "Steve",
            "tone": "Friendly and Casual (Peers / Friends)",
            "length": "Short (Concise - Min 60 words / 350 chars)"
        },
        "⚠️ Ambiguity / Vague Test: Prompt Under-Specified": {
            "req": "Things happened and stuff is weird. Email someone to fix it ASAP.",
            "recipient": "",
            "tone": "Default (Auto-detect)",
            "length": "Medium (Standard - Min 130 words / 800 chars)"
        }
    }

    selected_template_key = st.selectbox("📋 Professional Scenario Templates:", list(templates_catalog.keys()))
    if selected_template_key and templates_catalog[selected_template_key]:
        tmpl = templates_catalog[selected_template_key]
        if st.session_state.get("_last_template") != selected_template_key:
            st.session_state["_last_template"] = selected_template_key
            st.session_state["requirement_text"] = tmpl["req"]
            st.session_state["recipient_hint"] = tmpl["recipient"]
            st.session_state["selected_tone"] = tmpl["tone"]
            st.session_state["selected_length"] = tmpl["length"]

    default_req = st.session_state.get("requirement_text", "")

    # Main natural language requirement / incoming email
    requirement_text = st.text_area(
        "Natural Language Requirement or Email to Reply To (Mandatory):",
        value=default_req,
        height=130,
        placeholder="e.g. Write a polite email to my professor requesting 2 days... OR paste an incoming message here and click '↩️ Reply to Email'!",
        help="Specify the desired email in plain English, OR paste an incoming message and click '↩️ Reply to Email' to generate an appropriate response."
    )
    st.session_state["requirement_text"] = requirement_text

    # Dynamic prompt specificity evaluation & guidance
    req_words = requirement_text.strip().split()
    if requirement_text.strip():
        uncertainty_triggers = ["weird", "stuff", "asap", "things", "whatever", "someone", "something", "fix it"]
        has_uncertainty_trigger = any(t in requirement_text.lower() for t in uncertainty_triggers)

        if len(req_words) < 8 or has_uncertainty_trigger:
            st.warning(
                "⚠️ **Prompt may be too general!**\n\n"
                "Your prompt is quite brief or lacks specific details. "
                "**Please provide more specific details** (e.g., recipient name, why you are writing, or concrete deadlines) — "
                "**otherwise, a generic placeholder email will be generated.**\n\n"
                "💡 *Helpful details to add:*\n"
                "- **Recipient:** Who is receiving this? *(e.g., Professor Miller, Manager Sarah)*\n"
                "- **Reason / Justification:** Why are you writing? *(e.g., technical failure, doctor's appointment)*\n"
                "- **Action / Deadline:** What is requested or when is the deadline? *(e.g., 2 extra days, Friday at 3 PM)*"
            )
        else:
            st.caption("✅ **Specific Prompt Detected**: Good level of detail provided for calibrated email generation.")

    # Optional preferences collapsible accordion
    with st.expander("⚙️ Optional Preferences & Writing Constraints (All Optional)", expanded=False):
        col1, col2 = st.columns(2)
        with col1:
            recipient_hint = st.text_input(
                "Target Recipient Hint:",
                value=st.session_state.get("recipient_hint", ""),
                placeholder="e.g. Professor Smith, Sarah, Client"
            )
            tone_options = [
                "Default (Auto-detect)",
                "Polite and Professional",
                "Firm and Polite (Disputes / Follow-ups)",
                "Courteous and Urgent (Deadlines / Critical)",
                "Apologetic and Formal (Delays / Issues)",
                "Appreciative and Formal (Thank You / Recognition)",
                "Friendly and Casual (Peers / Friends)",
                "Assertive and Direct (Executive / Clear Boundaries)",
                "Warm and Welcoming (Introductions / Onboarding)",
                "Empathetic and Supportive (HR / Personal Matters)",
                "Persuasive and Enthusiastic (Proposals / Pitches)",
                "Diplomatic and Respectful (Sensitive Discussions)"
            ]
            saved_tone = st.session_state.get("selected_tone", "Default (Auto-detect)")
            tone_idx = tone_options.index(saved_tone) if saved_tone in tone_options else 0
            tone_hint = st.selectbox(
                "Desired Tone:",
                tone_options,
                index=tone_idx
            )
            length_options = [
                "Medium (Standard - Min 130 words / 800 chars)",
                "Short (Concise - Min 60 words / 350 chars)",
                "Long With Detailed (Comprehensive - Min 250 words / 1600 chars)"
            ]
            saved_length = st.session_state.get("selected_length", length_options[0])
            len_idx = length_options.index(saved_length) if saved_length in length_options else 0
            desired_length = st.selectbox(
                "Desired Length (Strict Minimum Floor):",
                length_options,
                index=len_idx
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
    btn_col1, btn_col2, btn_col3, btn_col4 = st.columns([1.3, 1.4, 1.4, 0.9])

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
                        resp = httpx.post(f"{API_BASE}/analyze", json=payload, timeout=120.0)
                        if resp.status_code == 200:
                            data = resp.json()
                            st.session_state["representation"] = data["representation"]
                            st.session_state["request_id"] = data["request_id"]
                            st.session_state["analysis_metadata"] = data
                            if data.get("is_underspecified") or data.get("representation", {}).get("uncertainties"):
                                st.warning(
                                    "⚠️ **Prompt Under-Specified Warning**: Your prompt lacks specific details. "
                                    "Please provide more specifics (recipient name, reason, or deadline) on the **Semantic Analysis** tab "
                                    "or in your prompt; otherwise, the generated email will be generic."
                                )
                            else:
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
                            a_resp = httpx.post(f"{API_BASE}/analyze", json=a_payload, timeout=120.0)
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
                        "generation_mode": "new",
                        "request_id": st.session_state.get("request_id"),
                        "llm_provider": settings.LLM_PROVIDER,
                        "api_key": settings.GEMINI_API_KEY if settings.LLM_PROVIDER == "gemini" else None,
                        "model_name": settings.GEMINI_MODEL_NAME if settings.LLM_PROVIDER == "gemini" else None
                    }
                    try:
                        g_resp = httpx.post(f"{API_BASE}/generate", json=g_payload, timeout=120.0)
                        if g_resp.status_code == 200:
                            g_data = g_resp.json()
                            st.session_state["generated_subject"] = g_data["subject"]
                            st.session_state["generated_body"] = g_data["body"]
                            st.session_state["original_model_body"] = g_data["body"]
                            st.session_state["generated_email_id"] = g_data.get("email_id")
                            st.session_state["generation_metadata"] = g_data

                            if g_data.get("is_general_draft"):
                                st.warning(
                                    "⚠️ **Notice: Generic Draft Generated**\n\n"
                                    "Because your prompt was general or missing key specifics (such as recipient name, exact reason, or dates), "
                                    "this email was drafted with general assumptions. "
                                    "For a more tailored email, add specific details or use **Redo & Refine** on the **Generated Email** tab."
                                )
                            else:
                                st.success("Email generated! View output on the 'Generated Email' tab.")

                            # Project Self-Validation Score & Auto-Redo Feedback
                            score = g_data.get("validation_score")
                            if score is not None:
                                score_pct = int(round(score * 100))
                                if g_data.get("passed_validation"):
                                    st.success(f"🛡️ **Project Self-Validation Score:** {score_pct}% (PASSED ≥80% Quality Threshold)")
                                else:
                                    st.warning(f"⚠️ **Project Self-Validation Score:** {score_pct}% (Below 80% Threshold — manual review recommended)")

                            if g_data.get("auto_redone"):
                                st.info(f"🔄 **Self-Correction Triggered:** Initial score was under 80%. The project automatically redid/refined the draft ({g_data.get('redo_attempts', 1)} attempt(s)) before presenting it.")

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
        if st.button("↩️ Reply to Email", use_container_width=True, type="primary"):
            if not requirement_text.strip():
                st.warning("Please enter or paste the email you want to reply to.")
            else:
                with st.spinner("Analyzing incoming message and drafting reply..."):
                    # Step A: Semantic analysis in reply mode
                    a_payload = {
                        "requirement": requirement_text.strip(),
                        "context": context_text.strip() if context_text else None,
                        "recipient_hint": recipient_hint.strip() if recipient_hint else None,
                        "tone_hint": None if tone_hint.startswith("Default") else tone_hint.lower(),
                        "desired_length": desired_length,
                        "language": language,
                        "generation_mode": "reply"
                    }
                    try:
                        a_resp = httpx.post(f"{API_BASE}/analyze", json=a_payload, timeout=120.0)
                        if a_resp.status_code == 200:
                            a_data = a_resp.json()
                            st.session_state["representation"] = a_data["representation"]
                            st.session_state["request_id"] = a_data["request_id"]
                            st.session_state["analysis_metadata"] = a_data
                    except Exception as e:
                        st.error(f"Analysis failed: {e}")
                        st.stop()

                    # Step B: Generate contextual reply
                    g_payload = {
                        "requirement": requirement_text.strip(),
                        "representation": st.session_state["representation"],
                        "context": context_text.strip() if context_text else None,
                        "desired_length": desired_length,
                        "language": language,
                        "system_version": "proposed",
                        "generation_mode": "reply",
                        "request_id": st.session_state.get("request_id"),
                        "llm_provider": settings.LLM_PROVIDER,
                        "api_key": settings.GEMINI_API_KEY if settings.LLM_PROVIDER == "gemini" else None,
                        "model_name": settings.GEMINI_MODEL_NAME if settings.LLM_PROVIDER == "gemini" else None
                    }
                    try:
                        g_resp = httpx.post(f"{API_BASE}/reply", json=g_payload, timeout=120.0)
                        if g_resp.status_code == 200:
                            g_data = g_resp.json()
                            st.session_state["generated_subject"] = g_data["subject"]
                            st.session_state["generated_body"] = g_data["body"]
                            st.session_state["original_model_body"] = g_data["body"]
                            st.session_state["generated_email_id"] = g_data.get("email_id")
                            st.session_state["generation_metadata"] = g_data

                            st.success("Reply generated! View and edit your reply on the 'Generated Email' tab.")

                            # Project Self-Validation Score & Auto-Redo Feedback
                            score = g_data.get("validation_score")
                            if score is not None:
                                score_pct = int(round(score * 100))
                                if g_data.get("passed_validation"):
                                    st.success(f"🛡️ **Project Self-Validation Score:** {score_pct}% (PASSED ≥80% Quality Threshold)")
                                else:
                                    st.warning(f"⚠️ **Project Self-Validation Score:** {score_pct}% (Below 80% Threshold — manual review recommended)")

                            if g_data.get("auto_redone"):
                                st.info(f"🔄 **Self-Correction Triggered:** Initial reply draft scored under 80%. The project automatically redid/refined the reply ({g_data.get('redo_attempts', 1)} attempt(s)) before presenting it.")

                            st.info("💡 You can review, polish, and edit the drafted reply directly in the **Generated Email** tab.")
                            model_id = g_data.get("model_identifier", "Unknown")
                            if "gemini" in model_id.lower():
                                st.info(f"✨ Reply drafted with Google Gemini (`{model_id}`)")
                        else:
                            st.error(f"Reply generation failed: {g_resp.text}")
                    except Exception as e:
                        st.error(f"Reply generation error: {e}")

    with btn_col4:
        if st.button("🔄 Reset", use_container_width=True):
            st.session_state.clear()
            st.rerun()
