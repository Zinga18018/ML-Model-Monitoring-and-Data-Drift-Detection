"""
Visualizer — Plotly Charts for Drift Analysis
==============================================
All charts follow a consistent dark theme:
  - Background  : #0a0a0a
  - Accent green : #00ff88
  - Accent cyan  : #00d4ff
  - Danger red   : #ff4444
  - Font         : JetBrains Mono
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


# Shared Layout Defaults 
_DARK_LAYOUT = dict(
    paper_bgcolor="#0a0a0a",
    plot_bgcolor="#0a0a0a",
    font=dict(family="JetBrains Mono, monospace", color="white", size=12),
    margin=dict(l=60, r=30, t=50, b=50),
    hoverlabel=dict(
        bgcolor="#1a1a2e",
        font_size=12,
        font_family="JetBrains Mono, monospace",
    ),
)

GREEN = "#00ff88"
CYAN = "#00d4ff"
RED = "#ff4444"
YELLOW = "#ffaa00"


def _apply_dark(fig: go.Figure) -> go.Figure:
    """Apply the shared dark theme to any figure."""
    fig.update_layout(**_DARK_LAYOUT)
    fig.update_xaxes(
        gridcolor="rgba(255,255,255,0.06)",
        zerolinecolor="rgba(255,255,255,0.08)",
    )
    fig.update_yaxes(
        gridcolor="rgba(255,255,255,0.06)",
        zerolinecolor="rgba(255,255,255,0.08)",
    )
    return fig


# 1. Drift Heatmap 
def drift_heatmap(drift_results: dict) -> go.Figure:
    """
    Feature x Metric heatmap.  Green = safe, red = drifted.

    Parameters
    ----------
    drift_results : dict returned by DriftDetector.detect_all()
    """
    features = list(drift_results.keys())
    metrics = ["KS Statistic", "PSI", "JS Divergence"]

    z = []
    text = []
    for feat in features:
        row_z = [
            drift_results[feat]["ks"]["statistic"],
            drift_results[feat]["psi"]["psi_value"],
            drift_results[feat]["js"]["js_value"],
        ]
        row_text = [
            f"KS={row_z[0]:.4f}<br>p={drift_results[feat]['ks']['p_value']:.4f}",
            f"PSI={row_z[1]:.4f}",
            f"JS={row_z[2]:.4f}",
        ]
        z.append(row_z)
        text.append(row_text)

    # Normalise each column to [0, 1] for colour mapping
    z_arr = np.array(z)
    z_norm = np.zeros_like(z_arr)
    for col in range(z_arr.shape[1]):
        col_max = z_arr[:, col].max()
        z_norm[:, col] = z_arr[:, col] / col_max if col_max > 0 else 0

    colorscale = [
        [0.0, GREEN],
        [0.5, YELLOW],
        [1.0, RED],
    ]

    fig = go.Figure(
        go.Heatmap(
            z=z_norm,
            x=metrics,
            y=features,
            text=text,
            texttemplate="%{text}",
            textfont=dict(size=11),
            colorscale=colorscale,
            showscale=True,
            colorbar=dict(
                title="Drift Severity",
                tickvals=[0, 0.5, 1],
                ticktext=["None", "Moderate", "High"],
            ),
            hovertemplate="Feature: %{y}<br>Metric: %{x}<br>%{text}<extra></extra>",
        )
    )
    fig.update_layout(title="Drift Heatmap — Feature x Metric", height=max(350, len(features) * 60 + 100))
    return _apply_dark(fig)


# 2. Distribution Comparison 
def distribution_comparison(
    reference: pd.Series,
    production: pd.Series,
    feature_name: str,
) -> go.Figure:
    """Overlaid histograms of reference vs. production for one feature."""
    fig = go.Figure()

    fig.add_trace(
        go.Histogram(
            x=reference,
            name="Reference",
            marker_color=CYAN,
            opacity=0.55,
            nbinsx=40,
            histnorm="probability density",
        )
    )
    fig.add_trace(
        go.Histogram(
            x=production,
            name="Production",
            marker_color=RED,
            opacity=0.55,
            nbinsx=40,
            histnorm="probability density",
        )
    )

    fig.update_layout(
        title=f"Distribution Comparison — {feature_name}",
        xaxis_title=feature_name,
        yaxis_title="Density",
        barmode="overlay",
        legend=dict(x=0.78, y=0.97, bgcolor="rgba(0,0,0,0)"),
        height=420,
    )
    return _apply_dark(fig)


# 3. Drift Timeline 
def drift_timeline(drift_results: dict, n_windows: int = 10) -> go.Figure:
    """
    Simulated drift-score evolution over *n_windows* time windows.
    Each window adds Gaussian noise to the latest drift scores to model
    a realistic monitoring timeline.
    """
    features = list(drift_results.keys())
    windows = [f"W{i+1}" for i in range(n_windows)]

    fig = go.Figure()
    palette = [GREEN, CYAN, RED, YELLOW, "#a855f7"]

    np.random.seed(0)
    for idx, feat in enumerate(features):
        base = drift_results[feat]["psi"]["psi_value"]
        values = np.clip(
            base + np.cumsum(np.random.normal(0, base * 0.15, n_windows)),
            0,
            None,
        )
        color = palette[idx % len(palette)]
        fig.add_trace(
            go.Scatter(
                x=windows,
                y=values,
                mode="lines+markers",
                name=feat,
                line=dict(color=color, width=2),
                marker=dict(size=6),
            )
        )

    # Threshold line
    fig.add_hline(
        y=0.2,
        line_dash="dash",
        line_color=RED,
        annotation_text="PSI Drift Threshold (0.2)",
        annotation_position="top left",
        annotation_font_color=RED,
    )

    fig.update_layout(
        title="Drift Score Timeline (Simulated Windows)",
        xaxis_title="Time Window",
        yaxis_title="PSI Score",
        height=420,
        legend=dict(bgcolor="rgba(0,0,0,0)"),
    )
    return _apply_dark(fig)


# 4. PSI Bar Chart 
def psi_bar_chart(drift_results: dict) -> go.Figure:
    """Horizontal bars per feature, colored by drift severity."""
    features = list(drift_results.keys())
    psi_values = [drift_results[f]["psi"]["psi_value"] for f in features]

    colors = []
    for v in psi_values:
        if v > 0.25:
            colors.append(RED)
        elif v > 0.1:
            colors.append(YELLOW)
        else:
            colors.append(GREEN)

    fig = go.Figure(
        go.Bar(
            x=psi_values,
            y=features,
            orientation="h",
            marker_color=colors,
            text=[f"{v:.4f}" for v in psi_values],
            textposition="outside",
            textfont=dict(color="white"),
            hovertemplate="Feature: %{y}<br>PSI: %{x:.4f}<extra></extra>",
        )
    )

    # Threshold line
    fig.add_vline(
        x=0.2,
        line_dash="dash",
        line_color=RED,
        annotation_text="Significant Drift (0.2)",
        annotation_position="top right",
        annotation_font_color=RED,
    )
    fig.add_vline(
        x=0.1,
        line_dash="dot",
        line_color=YELLOW,
        annotation_text="Moderate (0.1)",
        annotation_position="bottom right",
        annotation_font_color=YELLOW,
    )

    fig.update_layout(
        title="Population Stability Index by Feature",
        xaxis_title="PSI Value",
        yaxis_title="Feature",
        height=max(350, len(features) * 55 + 100),
    )
    return _apply_dark(fig)
