"""
frontend/app.py - Main Streamlit Web Application Entrypoint
Manages navigation, global styling, backend health polling, and page orchestration.
"""

import streamlit as st
import httpx
from backend.config import settings
from frontend.pages.page_assistant import render_page_assistant
from frontend.pages.page_semantic_preview import render_page_semantic_preview
from frontend.pages.page_email_viewer import render_page_email_viewer
from frontend.pages.page_experiments import render_page_experiments

st.set_page_config(
    page_title="AI Email Writing Assistant with Semantic Analysis",
    page_icon="✉️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Executive Custom Styling with Modern Typography and High-Contrast Design System
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=Inter:wght@400;500;600;700&display=swap');

    html, body, [class*="css"], .stMarkdown, .stText, .stButton, div, span, p {
        font-family: 'Plus Jakarta Sans', 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
    }

    /* High contrast color palette & typography */
    h1, h2, h3, h4 {
        color: #0f172a !important;
        font-weight: 700 !important;
        letter-spacing: -0.02em;
    }

    p, li, label, .stMarkdown p {
        color: #334155;
        font-size: 14.5px;
        line-height: 1.6;
    }

    /* Hide default sidebar nav */
    div[data-testid="stSidebarNav"] {
        display: none;
    }

    /* Executive Button Styling */
    .stButton>button[kind="primary"], .stButton>button[type="primary"] {
        background: linear-gradient(135deg, #1e3a8a, #2563eb) !important;
        color: #ffffff !important;
        border: none !important;
        font-weight: 600 !important;
        border-radius: 8px !important;
        padding: 0.55rem 1.2rem !important;
        box-shadow: 0 2px 5px rgba(37, 99, 235, 0.25) !important;
        transition: all 0.2s cubic-bezier(0.4, 0, 0.2, 1) !important;
    }
    .stButton>button[kind="primary"]:hover, .stButton>button[type="primary"]:hover {
        background: linear-gradient(135deg, #1d4ed8, #1e40af) !important;
        box-shadow: 0 6px 14px rgba(37, 99, 235, 0.35) !important;
        transform: translateY(-1px);
    }
    .stButton>button[kind="secondary"], .stButton>button[type="secondary"] {
        background: #ffffff !important;
        color: #0f172a !important;
        border: 1px solid #cbd5e1 !important;
        font-weight: 600 !important;
        border-radius: 8px !important;
        padding: 0.55rem 1.2rem !important;
        box-shadow: 0 1px 2px rgba(0, 0, 0, 0.04) !important;
        transition: all 0.2s cubic-bezier(0.4, 0, 0.2, 1) !important;
    }
    .stButton>button[kind="secondary"]:hover, .stButton>button[type="secondary"]:hover {
        background: #f8fafc !important;
        border-color: #94a3b8 !important;
        color: #1e3a8a !important;
        transform: translateY(-1px);
    }

    /* Input Fields & Textareas with Executive Polish */
    textarea.stTextArea, input.stTextInput, div[data-baseweb="select"] {
        border-radius: 8px !important;
        border-color: #cbd5e1 !important;
        color: #0f172a !important;
        background-color: #ffffff !important;
    }
    textarea:focus, input:focus {
        border-color: #2563eb !important;
        box-shadow: 0 0 0 3px rgba(37, 99, 235, 0.15) !important;
    }

    /* Executive Sidebar Styling */
    section[data-testid="stSidebar"] {
        background-color: #f8fafc !important;
        border-right: 1px solid #e2e8f0;
    }
    section[data-testid="stSidebar"] h2, section[data-testid="stSidebar"] h3 {
        color: #0f172a !important;
    }

    /* Callout & Alert Polish */
    div[data-testid="stAlert"] {
        border-radius: 8px !important;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.05) !important;
    }

    /* Category Pill Badges */
    .category-badge {
        display: inline-block;
        font-size: 11px;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.6px;
        padding: 3px 8px;
        border-radius: 12px;
        margin-bottom: 6px;
    }
</style>
""", unsafe_allow_html=True)


def check_backend_health():
    """Polls backend health for sidebar badge."""
    try:
        resp = httpx.get(f"http://{settings.BACKEND_HOST}:{settings.BACKEND_PORT}/api/health", timeout=2.0)
        if resp.status_code == 200:
            return True, resp.json()
    except Exception:
        pass
    return False, {}


def main():
    # Sidebar Navigation
    with st.sidebar:
        st.markdown("## ✉️ Email Assistant")
        st.markdown("**Semantic-Guided NLP Framework**")

        st.markdown("---")
        st.markdown("### 🧭 Navigation")
        pages = [
            "1. Email Assistant",
            "2. Semantic Analysis",
            "3. Generated Email",
            "4. Research Evaluation"
        ]
        selected_page = st.radio("Select View:", pages, index=0)

        st.markdown("---")
        st.markdown("### 🤖 LLM & Model Engine")
        with st.expander("Configure LLM / Gemini Key", expanded=(settings.LLM_PROVIDER == "gemini" and not settings.GEMINI_API_KEY)):
            prov_options = ["gemini", "mock", "openai", "local"]
            curr_prov = settings.LLM_PROVIDER.lower()
            prov_idx = prov_options.index(curr_prov) if curr_prov in prov_options else 0
            selected_provider = st.selectbox("Provider:", prov_options, index=prov_idx)

            key_val = settings.GEMINI_API_KEY or ""
            gemini_key_input = st.text_input(
                "Gemini API Key:",
                type="password",
                value=key_val,
                placeholder="Paste your Gemini API Key...",
                help="Get your free key at https://aistudio.google.com"
            )
            models_list = ["gemini-3.5-flash-lite", "gemini-3.8-flash", "gemini-3.8-live", "gemini-2.5-flash", "gemini-2.0-flash", "gemini-1.5-flash"]
            curr_model = settings.GEMINI_MODEL_NAME if settings.GEMINI_MODEL_NAME in models_list else "gemini-3.5-flash-lite"
            model_idx = models_list.index(curr_model)
            gemini_model_input = st.selectbox("Gemini Model:", models_list, index=model_idx, help="Gemini 3.5 Flash Lite delivers ultra-fast neural drafting with fresh quota.")

            if st.button("💾 Apply & Save Settings", use_container_width=True):
                try:
                    payload = {
                        "llm_provider": selected_provider,
                        "gemini_api_key": gemini_key_input,
                        "gemini_model_name": gemini_model_input
                    }
                    resp = httpx.post(
                        f"http://{settings.BACKEND_HOST}:{settings.BACKEND_PORT}/api/config",
                        json=payload,
                        timeout=5.0
                    )
                    if resp.status_code == 200:
                        st.success("Settings updated successfully!")
                        st.rerun()
                    else:
                        st.error(f"Update failed: {resp.text}")
                except Exception as ex:
                    # In case backend is offline, update local settings and .env directly
                    settings.LLM_PROVIDER = selected_provider
                    settings.GEMINI_API_KEY = gemini_key_input.strip() or None
                    settings.GEMINI_MODEL_NAME = gemini_model_input
                    st.success("Saved to local settings!")
                    st.rerun()

        st.markdown("---")
        st.markdown("### ⚙️ System Status")
        is_healthy, health_info = check_backend_health()
        if is_healthy:
            st.success("● Backend API Online")
            st.caption(f"**LLM Provider:** `{health_info.get('llm_provider')}`")
            st.caption(f"**NLP Engine:** `{health_info.get('embedding_engine')}`")
            st.caption(f"**Database:** `{health_info.get('database_status')}`")
        else:
            st.error("○ Backend API Offline")
            st.caption("Start backend: `uvicorn backend.main:app --port 8000`")

        st.markdown("---")
        st.caption("AI-Powered Email Writing Assistant with Semantic Analysis\nResearch Framework v1.0.0")

    # Render Selected Page
    if selected_page == "1. Email Assistant":
        render_page_assistant()
    elif selected_page == "2. Semantic Analysis":
        render_page_semantic_preview()
    elif selected_page == "3. Generated Email":
        render_page_email_viewer()
    elif selected_page == "4. Research Evaluation":
        render_page_experiments()


if __name__ == "__main__":
    main()
