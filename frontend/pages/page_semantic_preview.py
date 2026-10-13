"""
frontend/pages/page_semantic_preview.py - Page 2: Semantic Analysis
Displays interpreted communication requirements, intents, tone, facts, and uncertainties.
Allows user editing and formal confirmation prior to generation.
"""

import streamlit as st
from frontend.components.cards import render_metric_card


def render_page_semantic_preview():
    st.header("🧠 Semantic Analysis & Representation Preview")
    st.markdown(
        "Inspect the structured interpretation extracted from your requirement. "
        "You can edit any field to refine the instructions before generation."
    )

    representation = st.session_state.get("representation")
    if not representation:
        st.info("No semantic representation currently available. Return to **Email Assistant** and click **Analyze Requirement**.")
        return

    # Confidence and model metadata bar
    meta = st.session_state.get("analysis_metadata", {})
    col_m1, col_m2, col_m3 = st.columns(3)
    with col_m1:
        render_metric_card("Intent Confidence", f"{meta.get('confidence', 1.0) * 100:.1f}%", "Calculated semantic fit")
    with col_m2:
        render_metric_card("Analysis Latency", f"{meta.get('analysis_time_ms', 0):.1f} ms", "Linguistic extraction")
    with col_m3:
        status_color = "warning" if representation.get("uncertainties") else "success"
        status_text = "Requires Review" if representation.get("uncertainties") else "High Confidence"
        render_metric_card("Ambiguity Status", status_text, f"{len(representation.get('uncertainties', []))} uncertainties detected", status_color)

    # Uncertainties and Warnings Box
    uncertainties = representation.get("uncertainties", [])
    if uncertainties:
        st.warning("⚠️ **Ambiguities / Missing Critical Parameters Detected:**")
        for u in uncertainties:
            st.markdown(f"- **{u}**")
        clarifications = representation.get("clarification_questions", [])
        if clarifications:
            st.info("💡 **Recommended Clarification Questions:**\n" + "\n".join(f"- {q}" for q in clarifications))

    st.markdown("### ✏️ Edit or Confirm Representation")
    with st.form("edit_representation_form"):
        col1, col2 = st.columns(2)
        with col1:
            intent_val = st.text_input("Primary Intent:", value=representation.get("intent", ""))
            recipient_val = st.text_input("Recipient / Audience:", value=representation.get("recipient") or "")
            topic_val = st.text_input("Subject Topic:", value=representation.get("topic") or "")
            action_val = st.text_input("Required Action:", value=representation.get("required_action") or "")

        with col2:
            tone_val = st.text_input("Communication Tone:", value=representation.get("tone", "polite and professional"))
            reason_val = st.text_input("Justification / Reason:", value=representation.get("reason") or "")
            duration_val = st.text_input("Requested Duration:", value=representation.get("requested_duration") or "")
            deadline_val = st.text_input("Explicit Deadline / Date:", value=representation.get("deadline") or "")

        # Facts and constraints
        existing_facts = "\n".join(representation.get("required_facts", []))
        facts_text = st.text_area("Required Facts (One per line):", value=existing_facts, height=80)

        existing_constraints = "\n".join(representation.get("constraints", []))
        constraints_text = st.text_area("Strict Negative Constraints (One per line):", value=existing_constraints, height=80)

        user_confirmed = st.checkbox("✅ I have reviewed and confirmed this communication representation", value=representation.get("user_confirmed", True))

        submit_save = st.form_submit_button("💾 Save Changes to Representation", use_container_width=True)

        if submit_save:
            representation["intent"] = intent_val.strip()
            representation["recipient"] = recipient_val.strip() if recipient_val.strip() else None
            representation["topic"] = topic_val.strip() if topic_val.strip() else None
            representation["required_action"] = action_val.strip() if action_val.strip() else None
            representation["tone"] = tone_val.strip()
            representation["reason"] = reason_val.strip() if reason_val.strip() else None
            representation["requested_duration"] = duration_val.strip() if duration_val.strip() else None
            representation["deadline"] = deadline_val.strip() if deadline_val.strip() else None
            representation["required_facts"] = [f.strip() for f in facts_text.splitlines() if f.strip()]
            representation["constraints"] = [c.strip() for c in constraints_text.splitlines() if c.strip()]
            representation["user_confirmed"] = user_confirmed
            st.session_state["representation"] = representation
            st.success("Representation updated! The generator will now use these confirmed attributes.")
            st.rerun()

    # Inferred details breakdown
    inferred = representation.get("inferred_details", {})
    if inferred:
        with st.expander("🔍 Inferred Linguistic Context Details", expanded=False):
            st.json(inferred)
