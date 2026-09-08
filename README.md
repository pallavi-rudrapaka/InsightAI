# 📊 InsightAI – AI-Assisted Data Analysis & Decision Support System

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28%2B-FF4B4B.svg)](https://streamlit.io/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.3%2B-F7931E.svg)](https://scikit-learn.org/)
[![Plotly](https://img.shields.io/badge/Plotly-5.15%2B-3F4F75.svg)](https://plotly.com/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

**InsightAI** is a portfolio-quality, production-ready AI and Data Science web application built with Python, Streamlit, Scikit-learn, and Plotly. It empowers data analysts, scientists, and decision-makers to upload any tabular dataset (or run an immediate one-click demonstration with a built-in realistic customer analytics dataset) to perform automated end-to-end exploratory analysis, data quality auditing, Isolation Forest anomaly detection, supervised machine learning with baseline vs. strong tree model benchmarking, model explainability, LLM-based natural language insight generation (with a 100% deterministic mathematical fallback engine), data-driven recommendations, and analyst review governance.

---

## 🎯 Problem Statement

Modern organizations frequently struggle with tabular data ingestion pipelines:
1. **Manual Exploratory Data Analysis (EDA)** is time-consuming and prone to human oversight.
2. **Data hygiene defects** (missingness, subtle duplicates, extreme distribution outliers) often pass unnoticed into downstream models.
3. **Black-box machine learning** often lacks baseline comparisons, misleading stakeholders on true model utility.
4. **LLM hallucinations** frequently invent figures or misinterpret statistical correlations when analyzing tabular reports.
5. **Lack of Governance**: Insights are often deployed without human-in-the-loop validation or explicit causality disclaimers.

**InsightAI bridges this gap** by combining rigorous statistical computing and machine learning with strict LLM factual guardrails and responsible AI decision governance.

---

## 🚀 Key Features

### 1. 📋 Automated Dataset Profiling
- Ingests CSV files via drag-and-drop or loads the built-in 1,000-row sample dataset with one click.
- Computes dataset shape, memory footprint, missing rate, duplicate frequency, and full column catalog.
- Automatically infers data types: numerical, categorical, datetime, and identifier/key columns.
- Generates descriptive statistics for numerical variables (mean, std, percentiles, skewness) and categorical variables (frequencies, uniqueness).

### 2. 🛡️ Data Quality Audit & Scoring
- **Composite Data Quality Score (0–100)**: Evaluates Completeness (40%), Uniqueness (30%), and Distribution Health (30%) with letter grades (A–D).
- **Missing Value Audit**: Column-by-column breakdown of missing counts and percentages.
- **Duplicate Detection**: Identifies exact duplicate rows with dedicated inspection tables.
- **Outlier Detection**: Interquartile Range (IQR) rule ($Q_1 - 1.5 \times IQR, Q_3 + 1.5 \times IQR$) and Z-score ($|z| > 3$) diagnostics.

### 3. 📈 Interactive Exploratory Data Analysis (EDA)
- **Univariate Analysis**: Histograms with KDE curves, box plot marginals, and mean/median reference lines; categorical frequency charts.
- **Bivariate Analysis**: Scatter plots with OLS regression trendlines; grouped box and violin plots.
- **Correlation Heatmap**: Annotated Pearson and Spearman correlation matrices across continuous variables.

### 4. 🔍 Multi-Dimensional Anomaly Detection (Isolation Forest)
- Unsupervised anomaly isolation using Scikit-learn's `IsolationForest` algorithm.
- Interactive contamination slider (1%–15%) and automatic numerical feature extraction.
- Computes anomaly decision function scores (lower score = more isolated).
- Interactive 2D scatter visualization and score distribution histograms.
- **Analyst Advisory**: Explicitly clarifies that statistical anomalies represent multi-dimensional deviations, *not automatic fraud or defect*.

### 5. 🤖 Supervised Machine Learning with Leakage-Free Pipelines
- **Automated Task Detection**: Infers Binary Classification, Multiclass Classification, or Continuous Regression based on target cardinality and data types.
- **Zero Information Leakage**: Data preprocessing (`ColumnTransformer` with `SimpleImputer`, `StandardScaler`, and `OneHotEncoder`) is fit exclusively on the training split.
- **Train/Test Holdout Split**: Stratified splits for classification, configurable ratio (10%–40%).

### 6. ⚖️ Baseline vs. Strong Tree Model Comparison
- Benchmarks naive heuristic baselines against powerful tree ensembles:
  - **Classification**: `DummyClassifier` (Majority Class) vs. `RandomForestClassifier`.
  - **Regression**: `DummyRegressor` (Mean Value) vs. `RandomForestRegressor`.
- Side-by-side metric comparison table evaluating Accuracy, Precision, Recall, F1-Score, and ROC-AUC (classification), or MAE, RMSE, and $R^2$ (regression).
- Interactive confusion matrix heatmaps and regression residual diagnostics.

### 7. 💡 Model Explainability
- Extracts tree feature importances mapped back to human-readable one-hot feature names.
- Ranked horizontal bar chart highlighting top predictive signals.
- Computes percentage and cumulative importance contributions.
- **Causation Disclaimer**: Prominently warns that feature importance reflects predictive utility, *not causal business mechanisms*.

### 8. 🧠 AI Insights & Natural Language Synthesis
- Generates executive-ready analytical reports covering dataset summaries, data quality, anomaly findings, model benchmarks, key drivers, and limitations.
- **Strict Factual Guardrails**: The LLM prompt enforces strict grounding in computed JSON payload metrics—**zero hallucinated numbers**.
- **Dual Engine Architecture**:
  - Connects to **Google Gemini** (`gemini-2.0-flash`, `gemini-1.5-pro`) or **OpenAI** (`gpt-4o-mini`, `gpt-4o`).
  - **Deterministic Fallback Engine**: If no API key is provided, an offline mathematical synthesizer produces structured analytical findings with zero dependencies.

### 9. 🎯 Data-Driven Recommendations
- Actionable suggestions categorized into Ingestion Cleanliness, Outlier Auditing, and Model Operationalization.
- Clearly labeled as *"Data-driven suggestions (Not guaranteed conclusions)"*.

### 10. 📋 Analyst Review & Responsible AI Governance
- Comprehensive verification checklist covering Correlation vs. Causation, Anomaly Context, and Data Drift.
- Operational risk matrix with severity ratings and mitigation protocols.

---

## 🏗️ Architecture & Tech Stack

```
                                 ┌─────────────────────────┐
                                 │     User CSV Upload     │
                                 │  or Built-in Sample CSV │
                                 └────────────┬────────────┘
                                              │
                                              ▼
                                 ┌─────────────────────────┐
                                 │   src/data_loader.py    │
                                 │  Schema & Profiling     │
                                 └──────┬───────────┬──────┘
                                        │           │
                     ┌──────────────────┴──┐     ┌──┴──────────────────┐
                     ▼                     ▼     ▼                     ▼
           ┌──────────────────┐  ┌──────────────────┐  ┌──────────────────┐
           │src/data_quality  │  │   src/eda.py     │  │src/anomaly_detect│
           │Quality Scoring   │  │ Plotly Charts    │  │Isolation Forest  │
           └─────────┬────────┘  └────────┬─────────┘  └────────┬─────────┘
                     │                    │                     │
                     └────────────────────┼─────────────────────┘
                                          │
                                          ▼
                                 ┌─────────────────────────┐
                                 │    src/ml_models.py     │
                                 │  Leakage-Free Pipeline  │
                                 │ Baseline vs Tree Model  │
                                 └────────────┬────────────┘
                                              │
                                              ▼
                                 ┌─────────────────────────┐
                                 │  src/explainability.py  │
                                 │ Tree Feature Importance │
                                 └────────────┬────────────┘
                                              │
                                              ▼
                                 ┌─────────────────────────┐
                                 │    src/insights.py      │
                                 │  Payload Serialization  │
                                 │ Gemini / OpenAI / Rule  │
                                 └────────────┬────────────┘
                                              │
                                              ▼
                                 ┌─────────────────────────┐
                                 │         app.py          │
                                 │  Streamlit Multi-Tab UI │
                                 └─────────────────────────┘
```

- **Frontend / Framework**: [Streamlit](https://streamlit.io/)
- **Data Manipulation**: [Pandas](https://pandas.pydata.org/), [NumPy](https://numpy.org/)
- **Machine Learning**: [Scikit-learn](https://scikit-learn.org/) (`IsolationForest`, `RandomForestClassifier`, `RandomForestRegressor`, `ColumnTransformer`, `Pipeline`)
- **Interactive Visualizations**: [Plotly Express & Graph Objects](https://plotly.com/python/)
- **LLM Connectivity**: HTTP REST integration with Google Gemini and OpenAI APIs
- **Environment Management**: Python 3.10+ / Anaconda

---

## 📁 Project Structure

```
InsightAI/
├── app.py                      # Main Streamlit web application with modern multi-tab interface
├── requirements.txt            # Python dependencies
├── README.md                   # Comprehensive project documentation
├── .gitignore                  # Git ignore rules for Python, cache, and Streamlit
├── data/
│   └── sample_data.csv         # Realistic 1,000-row customer analytics & churn dataset
├── src/
│   ├── __init__.py             # Package marker
│   ├── data_loader.py          # Data ingestion, schema inference, profiling & summary stats
│   ├── data_quality.py         # Missing values, duplicates, outliers (IQR/Z-score), quality scoring
│   ├── eda.py                  # Interactive Plotly visualizations (univariate, bivariate, correlation)
│   ├── anomaly_detection.py    # Isolation Forest pipeline, scoring, thresholding, and diagnostics
│   ├── ml_models.py            # Automated task detection, leakage-free pipelines, baseline vs tree comparison
│   ├── explainability.py       # Tree feature importance & explainability with caution labels
│   └── insights.py             # LLM insight generation (Gemini/OpenAI) + deterministic offline fallback
├── reports/                    # Directory for generated reports and exports
└── tests/
    └── test_modules.py         # Automated test suite covering all modules and edge cases
```

---

## 💻 Installation & Setup

### 1. Clone the Repository
```bash
git clone https://github.com/your-username/InsightAI.git
cd InsightAI
```

### 2. Create and Activate Virtual Environment
```bash
# Using standard venv:
python -m venv venv
# Windows:
.\venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

---

## 🏃 Running the Application

Launch the Streamlit web application:
```bash
streamlit run app.py
```

Open your browser and navigate to:
```
http://localhost:8501
```

### Using Anaconda Environment
```bash
& "C:\ProgramData\anaconda3\python.exe" -m streamlit run app.py
```

---

## 🔬 Example Workflow

1. **Ingest Data**:
   - Click **"Load Customer Analytics Dataset"** in the sidebar for an instant live demonstration, or upload your own CSV file.
2. **Review Profiling & Schema**:
   - Inspect high-level KPI cards, column type classifications, and summary statistics on the **Overview** tab.
3. **Audit Data Quality**:
   - Check the **Composite Quality Score (0–100)**, view missing value tables, duplicate rows, and IQR outlier boundaries on the **Data Quality** tab.
4. **Explore Trends & Correlations**:
   - Navigate to the **EDA** tab to generate interactive distributions, bivariate scatter plots with trendlines, and the correlation heatmap.
5. **Isolate Anomalies**:
   - On the **Anomaly Detection** tab, adjust the contamination slider, run Isolation Forest, and review flagged rows.
6. **Train & Benchmark ML Models**:
   - On the **ML Prediction** tab, select a target variable (e.g. `Churn` for classification or `MonthlyCharges` for regression).
   - Click **"Train ML Models"** to view metrics, confusion matrices, or residual plots.
   - Switch to **Model Comparison** to inspect the performance lift over the naive baseline.
7. **Inspect Model Explainability**:
   - On the **Explainability** tab, explore top feature importance drivers and review the causality warning.
8. **Synthesize AI Insights & Recommendations**:
   - On the **AI Insights** tab, select your preferred engine (Deterministic or Gemini/OpenAI), generate an executive markdown report, and download it.
9. **Review Analyst Governance**:
   - Audit findings against the **Analyst Verification Checklist** and Operational Risk Matrix on the **Analyst Review** tab.

---

## 🧪 Automated Testing

InsightAI includes a complete unit test suite validating all core components and edge cases:
```bash
python tests/test_modules.py
```

The test suite validates:
- Ingestion, schema inference, and summary stats calculation
- Missing value, duplicate, and outlier detection logic
- Plotly chart generation
- Isolation Forest anomaly detection
- Supervised classification (`Churn`) and regression (`MonthlyCharges`)
- Baseline vs. tree model benchmarking
- Feature importance extraction
- LLM payload serialization and deterministic fallback generation
- Edge cases: empty CSVs, single-column datasets, purely categorical datasets

---

## ⚠️ Limitations

- **Tabular Datasets Only**: Optimized for tabular structured CSV data; unstructured formats (images, audio, free text) are not supported.
- **Dataset Scale**: Designed for local in-memory analysis (up to ~500,000 rows / 100 MB). Large-scale big data distributed computing (e.g., PySpark) is outside current scope.
- **Correlation $\neq$ Causation**: High feature importance indicates predictive association within the trained model, not verified causal mechanisms.
- **Model Scope**: Focuses on baseline heuristic models and Random Forest ensembles; deep neural networks and automated hyperparameter Bayesian searches are deferred to future releases.

---

## 🔮 Future Improvements

- [ ] SHAP (SHapley Additive exPlanations) values integration for localized instance-level explanations.
- [ ] Automated hyperparameter tuning with Bayesian Optimization (Optuna).
- [ ] Exportable interactive HTML and PDF executive summary reports.
- [ ] Time-series forecasting tab for temporal sequential datasets.
- [ ] Cloud data warehouse connectors (Snowflake, BigQuery, PostgreSQL).

---

## 📄 License

This project is licensed under the MIT License.
