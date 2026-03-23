"""
DriftLens — ML Model Monitoring & Data Drift Detection Dashboard
================================================================
Launch:  streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np

from src.drift_detector import DriftDetector
from src.model_monitor import ModelMonitor
from src.sample_data import generate_reference_data, generate_drifted_data
from src.visualizer import (
    drift_heatmap,
    distribution_comparison,
    drift_timeline,
    psi_bar_chart,
)

# Page Config 
st.set_page_config(
    page_title="DriftLens | Model Monitoring",
    page_icon="D",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS 
st.markdown(
    """
    <style>
    /*  Global  */
    .stApp {
        background-color: #0a0a0a;
        color: #e0e0e0;
    }

    /*  Sidebar  */
    section[data-testid="stSidebar"] {
        background-color: #0f0f1a;
        border-right: 1px solid rgba(0,255,136,0.15);
    }

    /*  Metric cards  */
    div[data-testid="stMetric"] {
        background: linear-gradient(135deg, #0f0f1a 0%, #141428 100%);
        border: 1px solid rgba(0,255,136,0.2);
        border-radius: 12px;
        padding: 16px 20px;
    }
    div[data-testid="stMetric"] label {
        color: #00d4ff !important;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.8rem !important;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    div[data-testid="stMetric"] div[data-testid="stMetricValue"] {
        color: #00ff88 !important;
        font-family: 'JetBrains Mono', monospace;
        font-size: 1.8rem !important;
        font-weight: 700;
    }

    /*  Tabs  */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: #0f0f1a;
        border-radius: 10px;
        padding: 4px;
    }
    .stTabs [data-baseweb="tab"] {
        color: #888;
        border-radius: 8px;
        font-family: 'JetBrains Mono', monospace;
    }
    .stTabs [aria-selected="true"] {
        background-color: rgba(0,255,136,0.12) !important;
        color: #00ff88 !important;
    }

    /*  Table  */
    .stDataFrame {
        border: 1px solid rgba(0,212,255,0.15);
        border-radius: 10px;
    }

    /*  Header bar  */
    .drift-header {
        background: linear-gradient(90deg, #0f0f1a 0%, #0d1117 50%, #0f0f1a 100%);
        border: 1px solid rgba(0,255,136,0.2);
        border-radius: 14px;
        padding: 24px 32px;
        margin-bottom: 24px;
        text-align: center;
    }
    .drift-header h1 {
        background: linear-gradient(135deg, #00ff88, #00d4ff);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-family: 'JetBrains Mono', monospace;
        font-size: 2rem;
        margin: 0;
    }
    .drift-header p {
        color: #888;
        margin: 6px 0 0;
        font-size: 0.95rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Header 
st.markdown(
    """
    <div class="drift-header">
        <h1> DriftLens</h1>
        <p>Real-time ML Model Monitoring &amp; Data Drift Detection</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# Sidebar 
with st.sidebar:
    st.markdown("### Configuration")

    data_source = st.radio(
        "Data Source",
        ["Demo Data", "Upload CSV"],
        help="Use built-in demo data or upload your own reference / production CSVs.",
    )

    drift_level = st.select_slider(
        "Drift Level (demo)",
        options=["low", "medium", "high"],
        value="medium",
        help="Controls how much the production distribution deviates from the reference.",
    )

    st.markdown("---")
    st.markdown("### Test Configuration")
    n_samples = st.slider("Sample size", 500, 5000, 2000, step=250)
    n_bins = st.slider("Histogram bins (PSI / JS)", 10, 50, 20)

    st.markdown("---")
    st.markdown(
        "<p style='color:#555;font-size:0.75rem;text-align:center'>"
        "Built by Yogesh Kuchimanchi<br>MIT License</p>",
        unsafe_allow_html=True,
    )

# Load Data 
if data_source == "Demo Data":
    ref_df = generate_reference_data(n=n_samples)
    prod_df = generate_drifted_data(n=n_samples, drift_level=drift_level)
else:
    col_up1, col_up2 = st.columns(2)
    with col_up1:
        ref_file = st.file_uploader("Reference CSV", type="csv")
    with col_up2:
        prod_file = st.file_uploader("Production CSV", type="csv")

    if ref_file and prod_file:
        ref_df = pd.read_csv(ref_file)
        prod_df = pd.read_csv(prod_file)
    else:
        st.info(" Upload both a reference and production CSV to begin.")
        st.stop()

# Run Drift Detection 
detector = DriftDetector(n_bins=n_bins)
drift_results = detector.detect_all(ref_df, prod_df)

# Run Model Monitoring 
monitor = ModelMonitor()
monitor.fit(ref_df)
model_report = monitor.generate_report(prod_df)

# Metrics Row 
features = list(drift_results.keys())
drifted = [f for f in features if drift_results[f]["overall_drift"]]
psi_values = [drift_results[f]["psi"]["psi_value"] for f in features]
overall_score = np.mean(psi_values)

m1, m2, m3, m4 = st.columns(4)
m1.metric("Features Analyzed", len(features))
m2.metric("Features Drifted", len(drifted), delta=f"{len(drifted)}/{len(features)}")
m3.metric("Max PSI", f"{max(psi_values):.4f}")
m4.metric("Overall Drift Score", f"{overall_score:.4f}")

st.markdown("")

# Tabs 
tab_heatmap, tab_dist, tab_psi, tab_timeline, tab_model, tab_raw = st.tabs(
    [
        " Drift Heatmap",
        " Distribution Comparison",
        " PSI Analysis",
        "⏱ Drift Timeline",
        " Model Report",
        " Raw Results",
    ]
)

# Tab 1: Heatmap 
with tab_heatmap:
    st.plotly_chart(drift_heatmap(drift_results), use_container_width=True)

# Tab 2: Distribution Comparison 
with tab_dist:
    sel_feature = st.selectbox("Select Feature", features, key="dist_feat")
    st.plotly_chart(
        distribution_comparison(
            ref_df[sel_feature],
            prod_df[sel_feature],
            sel_feature,
        ),
        use_container_width=True,
    )

# Tab 3: PSI Bar Chart 
with tab_psi:
    st.plotly_chart(psi_bar_chart(drift_results), use_container_width=True)

# Tab 4: Timeline 
with tab_timeline:
    st.plotly_chart(drift_timeline(drift_results), use_container_width=True)

# Tab 5: Model Report 
with tab_model:
    st.markdown("#### Reference vs. Production Model Metrics")

    mc1, mc2 = st.columns(2)
    with mc1:
        st.markdown("**Reference Metrics**")
        for k, v in model_report["reference_metrics"].items():
            st.metric(k.upper(), f"{v:.4f}")
    with mc2:
        st.markdown("**Production Metrics**")
        for k, v in model_report["production_metrics"].items():
            ref_v = model_report["reference_metrics"][k]
            delta = v - ref_v
            st.metric(k.upper(), f"{v:.4f}", delta=f"{delta:+.4f}")

    if model_report["concept_drift_detected"]:
        st.error(" Concept drift detected — model performance has degraded significantly.")
    else:
        st.success(" No significant concept drift — model is performing within acceptable bounds.")

    st.markdown("#### Feature Importances")
    imp_df = pd.DataFrame(
        list(model_report["feature_importances"].items()),
        columns=["Feature", "Importance"],
    ).sort_values("Importance", ascending=False)
    st.bar_chart(imp_df.set_index("Feature"), color="#00ff88")

# Tab 6: Raw Results 
with tab_raw:
    rows = []
    for feat in features:
        r = drift_results[feat]
        rows.append({
            "Feature": feat,
            "KS Statistic": r["ks"]["statistic"],
            "KS p-value": r["ks"]["p_value"],
            "KS Drift": "Yes" if r["ks"]["drift"] else "No",
            "PSI": r["psi"]["psi_value"],
            "PSI Drift": "Yes" if r["psi"]["drift"] else "No",
            "JS Divergence": r["js"]["js_value"],
            "JS Drift": "Yes" if r["js"]["drift"] else "No",
            "Overall Drift": "Yes" if r["overall_drift"] else "No",
        })

    results_df = pd.DataFrame(rows)

    # Highlight drifted rows
    def _highlight(row):
        if row["Overall Drift"] == "Yes":
            return ["background-color: rgba(255,68,68,0.15)"] * len(row)
        return ["background-color: rgba(0,255,136,0.08)"] * len(row)

    st.dataframe(
        results_df.style.apply(_highlight, axis=1),
        use_container_width=True,
        height=300,
    )
