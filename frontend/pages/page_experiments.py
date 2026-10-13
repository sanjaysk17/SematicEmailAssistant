"""
frontend/pages/page_experiments.py - Page 5: Research Evaluation & Benchmarking
Provides a controlled experimental harness to evaluate Baseline A (Direct LLM),
Baseline B (Structured Prompt), and Proposed System over benchmark datasets.
Calculates empirical metrics, displays error diagnostics, and exports results to CSV.
"""

import streamlit as st
import pandas as pd
import httpx
from backend.config import settings
from frontend.components.cards import render_metric_card

API_BASE = f"http://{settings.BACKEND_HOST}:{settings.BACKEND_PORT}/api"


def render_page_experiments():
    st.header("📊 Research Evaluation & Experimental Benchmarks")
    st.markdown(
        "Run controlled experimental trials to evaluate the core research question: "
        "*Does explicit semantic interpretation, structured representation, and post-generation validation "
        "improve the preservation of user communication requirements compared with direct LLM generation and structured-prompt-based generation?*"
    )

    # Configuration Controls
    with st.expander("🔬 Experiment Configuration", expanded=True):
        col1, col2, col3 = st.columns(3)
        with col1:
            exp_name = st.text_input("Experiment Title:", value="Ablation: Baselines vs Proposed System")
            split_filter = st.selectbox("Dataset Split:", ["test", "train", "val", "all"], index=0)
        with col2:
            max_examples = st.number_input("Sample Limit (0 for all dataset):", min_value=0, max_value=100, value=0)
        with col3:
            methods_selected = st.multiselect(
                "Systems to Evaluate:",
                ["baseline_a", "baseline_b", "proposed"],
                default=["baseline_a", "baseline_b", "proposed"]
            )

        btn_run = st.button("🚀 Execute Experimental Benchmark", type="primary", use_container_width=True)

    if btn_run:
        if not methods_selected:
            st.error("Please select at least one method to evaluate.")
            return

        with st.spinner(f"Executing trial across '{split_filter}' dataset partition..."):
            payload = {
                "experiment_name": exp_name,
                "methods_to_run": methods_selected,
                "split_filter": split_filter,
                "max_examples": None if max_examples == 0 else int(max_examples)
            }
            try:
                resp = httpx.post(f"{API_BASE}/evaluate", json=payload, timeout=60.0)
                if resp.status_code == 200:
                    st.session_state["experiment_results"] = resp.json()
                    st.success("Benchmark completed successfully!")
                else:
                    st.error(f"Experiment execution failed: {resp.text}")
            except Exception as e:
                st.error(f"Failed to communicate with evaluation API: {e}")

    results_data = st.session_state.get("experiment_results")
    if not results_data:
        st.info("No benchmark results in session. Configure and execute an experiment above to view comparative metrics.")
        return

    # Benchmark summary metadata
    st.markdown("---")
    st.markdown(f"### Results for: `{results_data.get('experiment_name')}`")
    st.caption(f"🧪 **Total Scenarios Evaluated:** `{results_data.get('total_examples_evaluated')}` | ⏱️ **Total Runtime:** `{results_data.get('runtime_seconds'):.2f}s`")

    # Aggregate Metrics Comparison Table
    aggregates = results_data.get("aggregate_metrics", {})
    if aggregates:
        st.markdown("#### 📈 Aggregate Performance Comparison Table")
        table_rows = []
        for method_key, metrics in aggregates.items():
            name_map = {
                "baseline_a": "Baseline A (Direct LLM)",
                "baseline_b": "Baseline B (Structured Prompt)",
                "proposed": "Proposed (Semantic Guided)"
            }
            table_rows.append({
                "System": name_map.get(method_key, method_key),
                "Samples (N)": metrics["sample_count"],
                "Intent Preservation Rate": f"{metrics['intent_preservation_rate'] * 100:.1f}%",
                "Mean Completeness": f"{metrics['mean_completeness'] * 100:.1f}%",
                "Mean Cosine Similarity": f"{metrics['mean_semantic_similarity']:.3f}",
                "Tone Suitability Rate": f"{metrics['tone_suitability_rate'] * 100:.1f}%",
                "Unsupported Info Rate": f"{metrics['unsupported_information_rate'] * 100:.1f}%",
                "Pass Rate": f"{metrics['pass_rate'] * 100:.1f}%",
                "Mean Latency (ms)": f"{metrics['mean_latency_ms']:.1f}"
            })

        df_agg = pd.DataFrame(table_rows)
        st.dataframe(df_agg, use_container_width=True, hide_index=True)

    # Detailed Per-Example Inspection
    st.markdown("#### 🔍 Per-Example Generation & Validation Inspector")
    detailed = results_data.get("detailed_results", [])
    if detailed:
        detailed_df = pd.DataFrame(detailed)
        st.dataframe(
            detailed_df[[
                "example_id", "domain", "method", "intent_preserved",
                "completeness_score", "semantic_similarity", "tone_suitable",
                "unsupported_count", "validation_status", "latency_ms"
            ]],
            use_container_width=True,
            hide_index=True
        )

    # CSV Export Button
    csv_file = results_data.get("export_csv_filename")
    if csv_file:
        try:
            with open(csv_file, "r") as f:
                csv_data = f.read()
            st.download_button(
                "📥 Export Full Results to CSV",
                data=csv_data,
                file_name="benchmark_experiment_results.csv",
                mime="text/csv",
                use_container_width=True
            )
        except Exception:
            st.caption(f"CSV available on disk: `{csv_file}`")
