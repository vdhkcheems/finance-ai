import os
from typing import Any, Dict

import pandas as pd
import streamlit as st

from agents.query_classifier import classify_and_parse_query
from agents.math import MathAgent
from main import build_graph


def render_dataframe_section(title: str, df: pd.DataFrame) -> None:
    if df is not None and isinstance(df, pd.DataFrame) and not df.empty:
        st.subheader(title)
        st.dataframe(df, use_container_width=True)


def render_metadata(state: Dict[str, Any]) -> None:
    st.subheader("Parsed Query Details")
    cols = st.columns(2)
    with cols[0]:
        st.markdown(f"**Classification**: {state.get('classification', 'unknown')}")
        st.markdown(f"**Companies**: {', '.join(state.get('company') or []) or '-'}")
        st.markdown(f"**Sectors**: {', '.join(state.get('sector') or []) or '-'}")
        st.markdown(f"**Indices**: {', '.join(state.get('indices') or []) or '-'}")
    with cols[1]:
        st.markdown(f"**Time Period**: {state.get('time_period', '-')}")
        st.markdown(f"**Period Type**: {state.get('period_type', '-')}")
        aggs = state.get('aggregations', [])
        if aggs:
            aggs_str = ", ".join([f"{a.get('metric')}:{a.get('aggregation')}" for a in aggs])
        else:
            aggs_str = "-"
        st.markdown(f"**Aggregations**: {aggs_str}")


def main():
    st.set_page_config(page_title="Finance-AI", layout="wide")
    st.title("Finance AI Explorer")
    st.caption("Ask about fundamentals; derived metrics are auto-computed from formulas.")

    with st.sidebar:
        st.header("Derived Metrics")
        try:
            derived = MathAgent().available_derived_metrics()
        except Exception:
            derived = []
        if derived:
            st.write(", ".join(sorted(derived)))
        else:
            st.write("No derived formulas found. Add to data/formulas.txt")

        st.header("Environment")
        key_present = bool(os.getenv("GEMINI_API_KEY"))
        st.write(f"Gemini API Key: {'set' if key_present else 'not set'}")

    query = st.text_input("Enter your financial query", placeholder="e.g., Show PAT margin for Infosys 2020-2024")
    run = st.button("Run")

    if run and query.strip():
        compiled = build_graph()
        try:
            final_state = compiled.invoke({"query": query.strip()})
        except Exception as e:
            st.error(f"Failed to run workflow: {e}")
            return

        # Metadata / parsing
        render_metadata(final_state)

        # Retrieved data
        retrieved = final_state.get("retrieved_data", {})
        if isinstance(retrieved, dict):
            if retrieved.get("error"):
                st.error(retrieved["error"])
            else:
                render_dataframe_section("Retrieved Data", retrieved.get("data"))
        else:
            st.info("No retrieved data available.")

        # Aggregated data (if any)
        aggregated = final_state.get("aggregated_data", {})
        if isinstance(aggregated, dict) and not aggregated.get("error"):
            render_dataframe_section("Aggregated Data", aggregated.get("data"))

        # Graph (if any)
        graph_path = final_state.get("graph_path")
        if graph_path and os.path.exists(graph_path):
            st.subheader("Graph")
            st.image(graph_path, caption=os.path.basename(graph_path), use_column_width=True)
        else:
            vis_msg = final_state.get("visualization")
            if vis_msg:
                st.info(vis_msg)


if __name__ == "__main__":
    main()

