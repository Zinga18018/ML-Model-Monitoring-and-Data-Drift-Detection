# 📡 DriftLens — ML Model Monitoring & Data Drift Detection Platform

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.31%2B-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io)
[![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.4%2B-F7931E?logo=scikit-learn&logoColor=white)](https://scikit-learn.org)
[![SciPy](https://img.shields.io/badge/SciPy-1.12%2B-8CAAE6?logo=scipy&logoColor=white)](https://scipy.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-00ff88.svg)](LICENSE)

A real-time, interactive dashboard for monitoring ML model performance and detecting data distribution drift between reference (training) and production (live) datasets using rigorous statistical tests.

---

## Architecture

```
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────────┐
│                 │     │                  │     │                     │
│  Reference Data │────▶│  Drift Detector  │────▶│  Streamlit Dashboard│
│  (Training)     │     │                  │     │                     │
└─────────────────┘     │  ┌────────────┐  │     │  ┌───────────────┐  │
                        │  │  KS Test   │  │     │  │  Heatmap      │  │
┌─────────────────┐     │  ├────────────┤  │     │  ├───────────────┤  │
│                 │     │  │  PSI       │  │     │  │  Distributions│  │
│ Production Data │────▶│  ├────────────┤  │     │  ├───────────────┤  │
│  (Live)         │     │  │  JS Div    │  │     │  │  PSI Chart    │  │
└─────────────────┘     │  └────────────┘  │     │  ├───────────────┤  │
                        └──────────────────┘     │  │  Timeline     │  │
                                │                │  ├───────────────┤  │
                        ┌───────▼──────────┐     │  │  Model Report │  │
                        │  Model Monitor   │────▶│  ├───────────────┤  │
                        │  (RandomForest)  │     │  │  Raw Results  │  │
                        └──────────────────┘     │  └───────────────┘  │
                                                 └─────────────────────┘
```

---

## Features

- **Statistical Drift Detection** — Three complementary tests (KS, PSI, JS) for robust drift identification
- **Model Performance Monitoring** — Tracks accuracy, precision, recall, F1, and AUC-ROC degradation
- **Interactive Visualizations** — Dark-themed Plotly charts with drill-down capabilities
- **Demo Data Generator** — Synthetic datasets with configurable drift levels (low / medium / high)
- **CSV Upload Support** — Bring your own reference and production data
- **Concept Drift Detection** — Compares model behavior on reference vs. production data

---

## Quick Start

### 1. Clone & Install

```bash
git clone https://github.com/your-username/ML-Model-Monitoring-and-Data-Drift-Detection.git
cd ML-Model-Monitoring-and-Data-Drift-Detection

python -m venv venv
# Windows
venv\Scripts\activate
# macOS / Linux
source venv/bin/activate

pip install -r requirements.txt
```

### 2. Launch

```bash
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
| **Drift Heatmap** | Feature × Metric matrix showing drift severity with color coding |
| **Distribution Comparison** | Overlaid histograms of reference vs. production for any selected feature |
| **PSI Analysis** | Horizontal bar chart with threshold lines for quick triage |
| **Drift Timeline** | Simulated time-series view of drift score evolution |
| **Model Report** | Side-by-side reference vs. production model metrics with degradation alerts |
| **Raw Results** | Tabular view of all test statistics with conditional formatting |

---

## Project Structure

```
ML-Model-Monitoring-and-Data-Drift-Detection/
├── app.py                  # Streamlit dashboard entry point
├── requirements.txt        # Python dependencies
├── .gitignore
├── README.md
├── src/
│   ├── __init__.py
│   ├── drift_detector.py   # KS, PSI, JS drift tests
│   ├── model_monitor.py    # RandomForest model tracking
│   ├── sample_data.py      # Synthetic data generator
│   └── visualizer.py       # Plotly chart builders
├── data/                   # (optional) saved datasets
└── assets/                 # (optional) images, logos
```

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| **Frontend** | Streamlit, Plotly |
| **Statistical Tests** | SciPy, NumPy |
| **ML Engine** | Scikit-learn (RandomForest) |
| **Data** | Pandas, NumPy |
| **Styling** | Custom CSS, JetBrains Mono |

---

## License

MIT License — Yogesh Kuchimanchi

```
MIT License

Copyright (c) 2026 Yogesh Kuchimanchi

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```
