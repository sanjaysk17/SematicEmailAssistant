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
from frontend.pages.page_validation import render_page_validation
from frontend.pages.page_experiments import render_page_experiments

st.set_page_config(
    page_title="AI Email Writing Assistant with Semantic Analysis",
    page_icon="✉️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling for clean modern aesthetics
st.markdown("""
<style>
    .main-title {
        font-size: 28px;
        font-weight: 800;
        color: #0f172a;
        margin-bottom: 2px;
    }
    .sub-title {
        font-size: 14px;
        color: #64748b;
        margin-bottom: 20px;
    }
    div[data-testid="stSidebarNav"] {
        display: none;
    }
    .stButton>button {
        border-radius: 6px;
        font-weight: 500;
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
            "4. Validation Results",
            "5. Research Evaluation"
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
            models_list = ["gemini-1.5-flash", "gemini-2.0-flash", "gemini-1.5-pro"]
            curr_model = settings.GEMINI_MODEL_NAME if settings.GEMINI_MODEL_NAME in models_list else "gemini-1.5-flash"
            model_idx = models_list.index(curr_model)
            gemini_model_input = st.selectbox("Gemini Model:", models_list, index=model_idx)

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
    elif selected_page == "4. Validation Results":
        render_page_validation()
    elif selected_page == "5. Research Evaluation":
        render_page_experiments()


if __name__ == "__main__":
    main()
