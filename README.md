# DriftLens -- ML Model Monitoring and Data Drift Detection

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.31%2B-FF4B4B?style=flat-square&logo=streamlit&logoColor=white)](https://streamlit.io)
[![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.4%2B-F7931E?style=flat-square&logo=scikit-learn&logoColor=white)](https://scikit-learn.org)
[![SciPy](https://img.shields.io/badge/SciPy-1.12%2B-8CAAE6?style=flat-square&logo=scipy&logoColor=white)](https://scipy.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-00ff88?style=flat-square)](LICENSE)

A real-time dashboard for monitoring ML model performance and detecting data distribution drift between reference (training) and production (live) datasets using statistical tests.

---

## Features

- **Statistical Drift Detection** -- Three complementary tests (KS, PSI, JS) for robust drift identification
- **Model Performance Monitoring** -- Tracks accuracy, precision, recall, F1, and AUC-ROC degradation
- **Interactive Visualizations** -- Dark-themed Plotly charts with drill-down capabilities
- **Demo Data Generator** -- Synthetic datasets with configurable drift levels (low / medium / high)
- **CSV Upload Support** -- Bring your own reference and production data
- **Concept Drift Detection** -- Compares model behavior on reference vs. production data

---

## Quick Start

```bash
git clone https://github.com/Zinga18018/ML-Model-Monitoring-and-Data-Drift-Detection.git
cd ML-Model-Monitoring-and-Data-Drift-Detection
python -m venv venv
venv\Scripts\activate           # Windows
source venv/bin/activate        # macOS / Linux
pip install -r requirements.txt
streamlit run app.py
```

The dashboard opens at **http://localhost:8501**.

---

## Supported Statistical Tests

| Test | What It Measures | Drift Threshold | Interpretation |
|------|-----------------|-----------------|----------------|
| **Kolmogorov-Smirnov** | Maximum CDF distance between two distributions | p < 0.05 | Sensitive to any distribution change |
| **Population Stability Index (PSI)** | Shift in population distributions over time | PSI > 0.2 | Industry-standard for credit scoring |
| **Jensen-Shannon Divergence** | Symmetric information-theoretic divergence | JS > 0.1 | Bounded [0, 1], robust to outliers |

---

## Dashboard Tabs

| Tab | Description |
|-----|-------------|
| Drift Heatmap | Feature x Metric matrix showing drift severity with color coding |
| Distribution Comparison | Overlaid histograms of reference vs. production for any selected feature |
| PSI Analysis | Horizontal bar chart with threshold lines for quick triage |
| Drift Timeline | Simulated time-series view of drift score evolution |
| Model Report | Side-by-side reference vs. production model metrics with degradation alerts |
| Raw Results | Tabular view of all test statistics with conditional formatting |

---

## Project Structure

```
ML-Model-Monitoring-and-Data-Drift-Detection/
|-- app.py                  # Streamlit dashboard entry point
|-- requirements.txt
|-- .gitignore
|-- README.md
|-- src/
|   |-- __init__.py
|   |-- drift_detector.py   # KS, PSI, JS drift tests
|   |-- model_monitor.py    # RandomForest model tracking
|   |-- sample_data.py      # Synthetic data generator
|   +-- visualizer.py       # Plotly chart builders
```

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Frontend | Streamlit, Plotly |
| Statistical Tests | SciPy, NumPy |
| ML Engine | Scikit-learn (RandomForest) |
| Data | Pandas, NumPy |

---

## License

MIT License -- Yogesh Kuchimanchi
