"""
frontend/components/cards.py - UI Card and Badge Components
Provides reusable HTML/CSS components for metrics, status badges, and warnings.
"""

import streamlit as st


def render_metric_card(title: str, value: str, subtitle: str = "", status: str = "normal"):
    """Renders a modern metric card with subtle borders and status tints."""
    color_map = {
        "normal": "#3b82f6",
        "success": "#10b981",
        "warning": "#f59e0b",
        "danger": "#ef4444"
    }
    bg_map = {
        "normal": "rgba(59, 130, 246, 0.08)",
        "success": "rgba(16, 185, 129, 0.08)",
        "warning": "rgba(245, 158, 11, 0.08)",
        "danger": "rgba(239, 68, 68, 0.08)"
    }
    border_color = color_map.get(status, "#3b82f6")
    bg_color = bg_map.get(status, "rgba(59, 130, 246, 0.08)")

    html = f"""
    <div style="
        border: 1px solid {border_color};
        background: {bg_color};
        border-radius: 8px;
        padding: 14px 18px;
        margin-bottom: 12px;
    ">
        <div style="font-size: 13px; color: #64748b; font-weight: 500; text-transform: uppercase; letter-spacing: 0.5px;">{title}</div>
        <div style="font-size: 24px; font-weight: 700; color: #0f172a; margin-top: 4px;">{value}</div>
        {f'<div style="font-size: 12px; color: #64748b; margin-top: 4px;">{subtitle}</div>' if subtitle else ''}
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)


def render_status_pill(status: str):
    """Renders a pill badge for PASS, NEEDS_REVIEW, or VALIDATION_ERROR."""
    styles = {
        "PASS": ("#10b981", "#ecfdf5", "✅ PASS"),
        "NEEDS_REVIEW": ("#d97706", "#fffbeb", "⚠️ NEEDS REVIEW"),
        "VALIDATION_ERROR": ("#dc2626", "#fef2f2", "❌ VALIDATION ERROR")
    }
    text_color, bg_color, label = styles.get(status, ("#64748b", "#f1f5f9", status))

    html = f"""
    <span style="
        background-color: {bg_color};
        color: {text_color};
        padding: 4px 12px;
        border-radius: 16px;
        font-weight: 600;
        font-size: 13px;
        border: 1px solid {text_color};
        display: inline-block;
    ">{label}</span>
    """
    st.markdown(html, unsafe_allow_html=True)
