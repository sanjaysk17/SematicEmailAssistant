"""
frontend/pages/page_email_viewer.py - Page 3: Generated Email
Displays generated email subject and body separately, allows editing,
copying, regeneration, and preserves original model output for diffing.
Strict security rule: Never sends email automatically.
"""

import streamlit as st
import httpx
from backend.config import settings

API_BASE = f"http://{settings.BACKEND_HOST}:{settings.BACKEND_PORT}/api"


def render_page_email_viewer():
    st.header("✉️ Generated Email Draft")
    st.markdown(
        "Review your generated email draft below. Subject and body are displayed separately for full inspection. "
        "You may edit the draft directly or regenerate."
    )

    subject = st.session_state.get("generated_subject")
    body = st.session_state.get("generated_body")

    if not body:
        st.info("No generated email available yet. Go to **Email Assistant** and click **Generate Email**.")
        return

    meta = st.session_state.get("generation_metadata", {})
    score = meta.get("validation_score")
    score_str = f"{int(round(score * 100))}%" if score is not None else "N/A"
    passed_val = meta.get("passed_validation", True)
    status_icon = "✅ Passed (≥80%)" if passed_val else "⚠️ Under 80%"

    st.caption(
        f"🤖 **Model Engine:** `{meta.get('model_identifier', 'mock-llm')}` | "
        f"⏱️ **Generation Time:** `{meta.get('generation_time_ms', 0):.1f} ms` | "
        f"🛡️ **Self-Validation Score:** `{score_str}` ({status_icon}) | "
        f"🧬 **Pipeline:** `{meta.get('system_version', 'proposed')}`"
    )

    if meta.get("auto_redone"):
        st.info(
            f"🔄 **Automated Self-Correction:** The project's self-validation engine detected that the initial draft scored below 80%. "
            f"The system automatically redid the task ({meta.get('redo_attempts', 1)} iteration(s)) with targeted corrective feedback before presenting this draft."
        )

    val_rep = meta.get("validation_report")
    if val_rep and isinstance(val_rep, dict):
        with st.expander("🔍 Inspect Internal Self-Validation Quality Report", expanded=False):
            v_col1, v_col2, v_col3 = st.columns(3)
            with v_col1:
                st.metric("Completeness", f"{int(round(val_rep.get('completeness_score', 0) * 100))}%")
            with v_col2:
                st.metric("Semantic Similarity", f"{val_rep.get('semantic_similarity', 0):.2f}")
            with v_col3:
                st.metric("Tone Suitability", "Suitable" if val_rep.get("tone_suitable") else "Divergent")

            if val_rep.get("missing_details"):
                st.warning(f"Missing items identified: {', '.join(val_rep['missing_details'])}")
            if val_rep.get("unsupported_details"):
                st.warning(f"Unverified claims flagged: {'; '.join(val_rep['unsupported_details'])}")

    # Generic Draft Warning if prompt lacked specifics
    rep = st.session_state.get("representation", {})
    req_text = st.session_state.get("requirement_text", "")
    uncertainties = rep.get("uncertainties", []) if isinstance(rep, dict) else getattr(rep, "uncertainties", [])
    is_general = meta.get("is_general_draft") or len(uncertainties) > 0 or (len(req_text.split()) < 8 and len(req_text.strip()) > 0)

    if is_general:
        st.warning(
            "⚠️ **Notice: Generic Draft Generated**  \n"
            "Your original prompt lacked specific details (such as recipient name, concrete reason, or dates), "
            "so this draft was generated using generalized assumptions.  \n"
            "👉 **To get a tailored draft:** Add specific details in the **Redo & Refine** section below or update your prompt on the **Email Assistant** tab."
        )

    # Editable draft fields
    edited_subject = st.text_input("Email Subject Line:", value=subject, key="viewer_subject_input")
    edited_body = st.text_area("Email Message Body:", value=body, height=260, key="viewer_body_input")

    # Check if user made modifications compared to original raw LLM output
    original_body = st.session_state.get("original_model_body", body)
    is_modified = (edited_body != original_body) or (edited_subject != subject)

    if is_modified:
        st.info("ℹ️ User edits detected. The system preserves the original model output for experimental comparison.")
        st.session_state["generated_subject"] = edited_subject
        st.session_state["generated_body"] = edited_body

    # Dedicated Redo & Refine Section (When user is not satisfied)
    st.markdown("---")
    with st.container():
        st.subheader("🔄 Not Satisfied with this Draft? Redo & Refine")
        st.markdown(
            "If the generated draft doesn't fully meet your expectations, instruct the assistant to redo it. "
            "Pick a quick preset or type your custom feedback below."
        )

        preset_col1, preset_col2, preset_col3, preset_col4, preset_col5, preset_col6 = st.columns(6)
        selected_preset = None
        with preset_col1:
            if st.button("✂️ Shorter", use_container_width=True, help="Make the draft shorter and more concise"):
                selected_preset = "Make the email significantly shorter, concise, and straight to the point."
        with preset_col2:
            if st.button("👔 More Formal", use_container_width=True, help="Increase professional formality"):
                selected_preset = "Make the email much more formal, polite, and professional."
        with preset_col3:
            if st.button("🤝 Friendlier", use_container_width=True, help="Adopt a warmer, friendlier register"):
                selected_preset = "Make the tone warmer, friendlier, and more conversational."
        with preset_col4:
            if st.button("⏰ More Urgent", use_container_width=True, help="Highlight deadlines and urgency"):
                selected_preset = "Emphasize immediate urgency and prompt response politely."
        with preset_col5:
            if st.button("🎯 Add Specifics", use_container_width=True, help="Replace general placeholders with concrete specifics"):
                selected_preset = "Replace generalized statements with concrete specifics: explicitly mention the recipient name, why this is requested, and clear deadlines."
        with preset_col6:
            if st.button("🎲 Alternative", use_container_width=True, help="Generate fresh alternative wording"):
                selected_preset = "Regenerate with completely fresh phrasing and alternative sentence structure."

        # Maintain or set revision instructions in session state
        if selected_preset:
            st.session_state["redo_instruction_val"] = selected_preset

        custom_revision = st.text_input(
            "Specific feedback or revision instructions (Optional):",
            value=st.session_state.get("redo_instruction_val", ""),
            placeholder="e.g. Do not apologize, emphasize our deadline at 5 PM, keep it to two brief paragraphs...",
            key="redo_custom_input",
            help="Tell the AI exactly what wasn't satisfactory and what to change."
        )

        r_col1, r_col2 = st.columns([2, 1])
        with r_col1:
            trigger_redo = st.button("⚡ Redo Draft Mail Now", type="primary", use_container_width=True)
        with r_col2:
            if st.button("🧹 Clear Feedback", use_container_width=True):
                st.session_state["redo_instruction_val"] = ""
                st.rerun()

        if trigger_redo or selected_preset:
            revision_text = custom_revision.strip() or selected_preset or "Generate an alternative variation with refined phrasing."
            with st.spinner("Redoing email draft based on your feedback..."):
                req_text = st.session_state.get("requirement_text", "")
                rep = st.session_state.get("representation", {})
                prev_draft_text = f"Subject: {edited_subject}\n\n{edited_body}"
                
                payload = {
                    "requirement": req_text,
                    "representation": rep,
                    "system_version": "proposed",
                    "request_id": st.session_state.get("request_id"),
                    "revision_instruction": revision_text,
                    "previous_draft": prev_draft_text,
                    "llm_provider": settings.LLM_PROVIDER,
                    "api_key": settings.GEMINI_API_KEY if settings.LLM_PROVIDER == "gemini" else None,
                    "model_name": settings.GEMINI_MODEL_NAME if settings.LLM_PROVIDER == "gemini" else None
                }
                try:
                    resp = httpx.post(f"{API_BASE}/generate", json=payload, timeout=35.0)
                    if resp.status_code == 200:
                        data = resp.json()
                        st.session_state["generated_subject"] = data["subject"]
                        st.session_state["generated_body"] = data["body"]
                        st.session_state["original_model_body"] = data["body"]
                        st.session_state["generation_metadata"] = data
                        st.session_state["redo_instruction_val"] = ""
                        st.success(f"✨ Draft successfully redone: '{revision_text}'")
                        st.rerun()
                    else:
                        st.error(f"Redo generation failed: {resp.text}")
                except Exception as e:
                    st.error(f"Redo request error: {e}")

    st.markdown("---")
    # Quick action bar
    col1, col2, col3 = st.columns([1.5, 1.5, 1.5])

    with col1:
        full_text = f"Subject: {edited_subject}\n\n{edited_body}"
        # Code block with copy button
        st.download_button(
            "📋 Download Draft (.txt)",
            data=full_text,
            file_name="email_draft.txt",
            mime="text/plain",
            use_container_width=True
        )

    with col2:
        if st.button("🎲 Fast Variation", use_container_width=True, help="Quickly re-roll an alternative variation"):
            with st.spinner("Generating alternative variation..."):
                req_text = st.session_state.get("requirement_text", "")
                rep = st.session_state.get("representation", {})
                payload = {
                    "requirement": req_text,
                    "representation": rep,
                    "system_version": "proposed",
                    "request_id": st.session_state.get("request_id"),
                    "revision_instruction": "Generate an alternative phrasing variation with different wording.",
                    "previous_draft": f"Subject: {edited_subject}\n\n{edited_body}",
                    "llm_provider": settings.LLM_PROVIDER,
                    "api_key": settings.GEMINI_API_KEY if settings.LLM_PROVIDER == "gemini" else None,
                    "model_name": settings.GEMINI_MODEL_NAME if settings.LLM_PROVIDER == "gemini" else None
                }
                try:
                    resp = httpx.post(f"{API_BASE}/generate", json=payload, timeout=35.0)
                    if resp.status_code == 200:
                        data = resp.json()
                        st.session_state["generated_subject"] = data["subject"]
                        st.session_state["generated_body"] = data["body"]
                        st.session_state["original_model_body"] = data["body"]
                        st.session_state["generation_metadata"] = data
                        st.success("Alternative variation generated!")
                        st.rerun()
                except Exception as e:
                    st.error(f"Regeneration error: {e}")

    with col3:
        if st.button("💾 Save to Drafts Library", use_container_width=True):
            try:
                d_payload = {
                    "prompt": st.session_state.get("requirement_text", edited_subject),
                    "subject": edited_subject,
                    "body": edited_body,
                    "user_email": st.session_state.get("user_email", "user@example.com")
                }
                d_resp = httpx.post(f"{API_BASE}/drafts", json=d_payload, timeout=5.0)
                if d_resp.status_code == 200:
                    st.success("Draft and prompt successfully saved to your database!")
                else:
                    st.warning("Could not persist to database.")
            except Exception as e:
                st.info(f"Saved locally: {e}")

    # Prompt Inspector
    raw_prompt = meta.get("raw_prompt_used")
    if raw_prompt:
        with st.expander("🔍 View Raw Prompt Sent to Model", expanded=False):
            st.code(raw_prompt, language="markdown")
