"""
frontend/pages/page_validation.py - Page 4: Validation Results
Displays rigorous evaluation results: intent preservation, semantic alignment,
tone suitability, key-information completeness, unsupported claims, and warnings.
Strictly distinguishes computed algorithmic measurements from model-generated judgments.
"""

import streamlit as st
import pandas as pd
from frontend.components.cards import render_metric_card, render_status_pill


def render_page_validation():
    st.header("🛡️ Output Validation Report")
    st.markdown(
        "The Output Validation Engine independently verifies whether the generated draft faithfully satisfies "
        "the user's communication requirements without omission, distortion, or unsupported additions."
    )

    report = st.session_state.get("validation_report")
    if not report:
        st.info("No validation report available yet. Draft an email on **Email Assistant** to view validation metrics.")
        return

    # Top Status Bar
    overall_status = report.get("overall_status", "NEEDS_REVIEW")
    st.markdown("### Overall Validation Verdict")
    render_status_pill(overall_status)

    st.caption(f"⏱️ **Validation Evaluation Latency:** `{report.get('validation_time_ms', 0):.1f} ms` | 🧬 **Similarity Model:** `{report.get('similarity_model', 'NLP Engine')}`")

    # High-level metrics row
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        intent_ok = report.get("intent_preserved", False)
        status_type = "success" if intent_ok else "danger"
        render_metric_card("Intent Preserved", "YES" if intent_ok else "NO", "Communication purpose check", status_type)

    with col2:
        comp_score = report.get("completeness_score", 0.0)
        status_type = "success" if comp_score >= 0.8 else ("warning" if comp_score >= 0.4 else "danger")
        render_metric_card("Completeness", f"{comp_score * 100:.1f}%", "Fraction of required facts present", status_type)

    with col3:
        sim_score = report.get("semantic_similarity", 0.0)
        status_type = "success" if sim_score >= 0.50 else "normal"
        render_metric_card("Semantic Cosine", f"{sim_score:.3f}", "Cosine similarity in [0, 1]", status_type)

    with col4:
        tone_ok = report.get("tone_suitable", False)
        status_type = "success" if tone_ok else "warning"
        render_metric_card("Tone Suitable", "YES" if tone_ok else "NO", f"Detected: {report.get('tone_detected', 'N/A')}", status_type)

    # Actionable Warnings & Diagnostic Alerts
    warnings = report.get("actionable_warnings", [])
    if warnings:
        st.markdown("### ⚠️ Actionable Warnings for Review")
        for w in warnings:
            st.warning(f"• {w}")

    # Section 1: Key Information Completeness Details
    st.markdown("### 📋 Key-Information Completeness Breakdown")
    fact_evals = report.get("fact_evaluations", [])
    if fact_evals:
        table_rows = []
        for f in fact_evals:
            status_icon = "✅" if f["status"] == "Present" else ("❌" if f["status"] == "Missing" else "⚠️")
            table_rows.append({
                "Parameter / Fact": f["fact"],
                "Status": f"{status_icon} {f['status']}",
                "Diagnostic Explanation": f["explanation"]
            })
        st.dataframe(pd.DataFrame(table_rows), use_container_width=True, hide_index=True)
    else:
        st.info("No explicit structured facts recorded.")

    # Section 2: Unsupported Information / Hallucination Detection
    unsupported = report.get("unsupported_details", [])
    st.markdown("### 🔍 Unsupported Information & Hallucination Check")
    if unsupported:
        st.error(f"Potential unsupported commitments detected ({len(unsupported)}):")
        for u in unsupported:
            st.markdown(f"- 🚩 **{u}**")
    else:
        st.success("✅ No unsupported commitments, invented attachments, or unverified claims detected.")

    # Section 3: Metric Definitions (Academic Transparency)
    with st.expander("📖 Rigorous Metric Definitions & Calculation Rules", expanded=False):
        metric_defs = report.get("metric_definitions", {})
        for metric_name, definition in metric_defs.items():
            st.markdown(f"**`{metric_name}`**: {definition}")
        st.markdown(
            "> **Research Note on Semantic Similarity:** A high cosine similarity score indicates topical alignment, "
            "but does not guarantee complete omission-free preservation of specific dates, quantities, or actions. "
            "Our multi-aspect validation engine combines semantic embeddings with deterministic factual verification."
        )
