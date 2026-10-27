# DriftLens: tabular drift demo

A local Streamlit app that compares numeric columns in two datasets using
Kolmogorov–Smirnov tests, population stability index (PSI), and Jensen–Shannon
divergence. It accepts two CSV files or generates a seeded synthetic example.
It does not connect to a deployed model or a live production stream.

## Run locally

Requires Python 3.10 or newer. From a clone of this repository:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
streamlit run app.py
```

On Windows, activate with `.venv\Scripts\activate` instead. The app normally opens
at `http://localhost:8501`. No API key is required. Dependencies have minimum
versions rather than a fully pinned environment.

## What the views show

- **Distribution comparison and PSI:** differences between the selected reference
  and comparison datasets. The built-in example generates five numeric features;
  reference seed is 42 and comparison seed is 123.
- **Drift results:** feature-level test statistics and flags using fixed defaults
  (`p < 0.05`, `PSI > 0.2`, `JS > 0.1`). These are demonstration thresholds, not
  thresholds calibrated for a particular application; multiple feature tests are
  not corrected for multiplicity.
- **Model report:** a new random forest trained on automatically generated binary
  pseudo-labels. Labels are generated separately for each dataset from its own
  standardized features and noise. The reported metric changes are simulation
  diagnostics, not measured performance of a supplied model or evidence that
  real-world concept drift has occurred.
- **Timeline:** a synthetic sequence made by adding random perturbations to the
  current PSI values. It is not a history of observed time windows.

For CSV input, use matching numeric columns in both files, with finite values and
no missing values for the model-report path. The upload path does not provide
comprehensive schema or edge-case validation. Do not upload sensitive records to
an instance you do not control.

## Source

- [`src/drift_detector.py`](src/drift_detector.py): statistical comparisons.
- [`src/sample_data.py`](src/sample_data.py): synthetic data and shift settings.
- [`src/model_monitor.py`](src/model_monitor.py): pseudo-label construction and
  random-forest metrics.
- [`src/visualizer.py`](src/visualizer.py): Plotly views, including the simulated timeline.
- [`app.py`](app.py): Streamlit interface.

There is no committed test suite or saved evaluation run. This documentation
update was checked against source; it is not an end-to-end runtime validation.

## Attribution and license

Yogesh Kuchimanchi. The existing project documentation states MIT licensing;
a standalone `LICENSE` file is not currently included in this repository.
