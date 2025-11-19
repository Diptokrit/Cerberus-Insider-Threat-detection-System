import streamlit as st
import pandas as pd
import os

# --- Configuration ---
BASELINE_RESULTS_PATH = 'model_results.csv'
ADVANCED_RESULTS_PATH = 'advanced_model_results.csv'

# --- Page Setup ---
st.set_page_config(layout="wide", page_title="Project Cerberus Final Report")
st.title("Project Cerberus: Final Results Comparison")
st.header("Context-Aware Insider Threat Detection")

# --- Load Data ---
try:
    baseline_df = pd.read_csv(BASELINE_RESULTS_PATH)
    advanced_df = pd.read_csv(ADVANCED_RESULTS_PATH)

    # Count insiders
    insiders_baseline = baseline_df['is_insider'].sum()
    insiders_advanced = advanced_df['is_insider'].sum()

    # --- 1. The Graph (Bar Chart) ---
    st.subheader("Model Performance: Insiders Found in Top 10")
    
    # Create data for the chart
    chart_data = pd.DataFrame({
        'Model': ['1. Baseline (Isolation Forest)', '2. Advanced (Temporal GNN)'],
        'Insiders Found': [insiders_baseline, insiders_advanced],
        'color': ['#FF7276', '#A6FFA6'] # Red vs Green
    })
    
    # Draw the chart using the color column
    st.bar_chart(chart_data, x='Model', y='Insiders Found', color='color')

    st.success(f"**Conclusion:** The Advanced Model (TGN) found **{insiders_advanced} insiders**, outperforming the Baseline Model's **{insiders_baseline} insiders**.")
    st.markdown("---")

    # --- 2. The Detailed Proof Tables ---
    st.header("Detailed Anomaly Tables (The Proof)")
    col1, col2 = st.columns(2)

    # Baseline Table
    with col1:
        st.subheader("Baseline Model Results")
        st.write("Summary-based detection.")
        st.dataframe(baseline_df.style.apply(
            lambda row: ['background-color: #FF7276' if row.get('is_insider') else '' for _ in row], axis=1
        ))

    # Advanced Table
    with col2:
        st.subheader("Advanced Model Results")
        st.write("Sequence-based detection.")
        st.dataframe(advanced_df.style.apply(
            lambda row: ['background-color: #A6FFA6' if row.get('is_insider') else '' for _ in row], axis=1
        ))

except FileNotFoundError as e:
    st.error(f"Error: Could not find a results file. {e}")