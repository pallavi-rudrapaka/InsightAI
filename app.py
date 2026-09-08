"""
app.py
InsightAI - AI-Assisted Data Analysis & Decision Support System
A portfolio-grade web application for automated profiling, quality auditing,
interactive EDA, Isolation Forest anomaly detection, supervised ML,
feature explainability, LLM-based natural language insights, and analyst review governance.
"""

import os
import sys
import io
import pandas as pd
import numpy as np
import streamlit as st

# Configure page settings
st.set_page_config(
    page_title="InsightAI | AI-Assisted Data Analysis",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E293B;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #64748B;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 8px;
        padding: 16px;
        text-align: center;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    .metric-value {
        font-size: 1.8rem;
        font-weight: 700;
        color: #2563EB;
    }
    .metric-label {
        font-size: 0.85rem;
        font-weight: 500;
        color: #64748B;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    .badge-excellent {
        background-color: #DEF7EC;
        color: #03543F;
        padding: 4px 10px;
        border-radius: 9999px;
        font-weight: 600;
    }
    .badge-good {
        background-color: #E1EFFE;
        color: #1E429F;
        padding: 4px 10px;
        border-radius: 9999px;
        font-weight: 600;
    }
    .badge-warning {
        background-color: #FDF6B2;
        color: #723B13;
        padding: 4px 10px;
        border-radius: 9999px;
        font-weight: 600;
    }
    .advisory-box {
        background-color: #EFF6FF;
        border-left: 4px solid #3B82F6;
        padding: 14px 18px;
        border-radius: 4px;
        margin-top: 10px;
        margin-bottom: 15px;
    }
    .warning-box {
        background-color: #FEF2F2;
        border-left: 4px solid #EF4444;
        padding: 14px 18px;
        border-radius: 4px;
        margin-top: 10px;
        margin-bottom: 15px;
    }
</style>
""", unsafe_allow_html=True)

# Import Core Modules
from src.data_loader import (
    load_data,
    get_dataset_overview,
    identify_column_types,
    get_summary_statistics
)
from src.data_quality import (
    analyze_missing_values,
    analyze_duplicates,
    analyze_outliers,
    compute_data_quality_score
)
from src.eda import (
    plot_numerical_distribution,
    plot_categorical_distribution,
    plot_correlation_heatmap,
    plot_bivariate_scatter,
    plot_bivariate_box
)
from src.anomaly_detection import (
    detect_anomalies_isolation_forest,
    plot_anomaly_distribution,
    plot_anomaly_score_histogram
)
from src.ml_models import (
    detect_problem_type,
    train_and_evaluate_models,
    plot_confusion_matrix_heatmap,
    plot_regression_actual_vs_pred,
    plot_regression_residuals
)
from src.explainability import (
    extract_feature_importance,
    plot_feature_importance_bar
)
from src.insights import (
    build_analytics_payload,
    generate_deterministic_insights,
    generate_llm_insights
)

# Initialize Session State
if "df" not in st.session_state:
    st.session_state.df = None
if "dataset_name" not in st.session_state:
    st.session_state.dataset_name = None
if "anomaly_results" not in st.session_state:
    st.session_state.anomaly_results = None
if "ml_results" not in st.session_state:
    st.session_state.ml_results = None
if "explainability_results" not in st.session_state:
    st.session_state.explainability_results = None
if "insights_result" not in st.session_state:
    st.session_state.insights_result = None

# ==========================================
# SIDEBAR CONTROLS
# ==========================================
with st.sidebar:
    st.title("InsightAI")
    st.caption("AI-Assisted Data Analysis & Decision Support")
    st.markdown("---")

    st.subheader("Data Source")
    data_source = st.radio(
        "Select Data Ingestion Method:",
        ["Load Sample Dataset", "Upload Custom CSV"],
        index=0
    )

    sample_path = os.path.join(os.path.dirname(__file__), "data", "sample_data.csv")

    if data_source == "Load Sample Dataset":
        if st.button("Load Customer Analytics Dataset", use_container_width=True, type="primary"):
            if os.path.exists(sample_path):
                st.session_state.df = load_data(sample_path)
                st.session_state.dataset_name = "sample_data.csv (Customer Analytics)"
                # Reset downstream state
                st.session_state.anomaly_results = None
                st.session_state.ml_results = None
                st.session_state.explainability_results = None
                st.session_state.insights_result = None
                st.success("Sample dataset loaded!")
            else:
                st.error("Sample dataset file not found.")

    else:
        uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])
        if uploaded_file is not None:
            try:
                st.session_state.df = load_data(uploaded_file)
                st.session_state.dataset_name = uploaded_file.name
                # Reset downstream state
                st.session_state.anomaly_results = None
                st.session_state.ml_results = None
                st.session_state.explainability_results = None
                st.session_state.insights_result = None
                st.success(f"Uploaded: {uploaded_file.name}")
            except Exception as e:
                st.error(f"Error loading CSV: {str(e)}")

    # Auto-load sample dataset if state is empty on startup
    if st.session_state.df is None and os.path.exists(sample_path):
        st.session_state.df = load_data(sample_path)
        st.session_state.dataset_name = "sample_data.csv (Customer Analytics)"

    st.markdown("---")
    st.subheader("AI Insights Settings")
    llm_provider = st.selectbox(
        "Insight Engine:",
        ["Deterministic Offline Engine", "Google Gemini", "OpenAI"],
        index=0
    )

    api_key_input = ""
    llm_model = ""
    if llm_provider == "Google Gemini":
        api_key_input = st.text_input("Gemini API Key", type="password", value=os.environ.get("GEMINI_API_KEY", ""))
        llm_model = st.selectbox("Gemini Model", ["gemini-2.0-flash", "gemini-1.5-flash", "gemini-1.5-pro"])
        if not api_key_input:
            st.info("No API key entered; system will use deterministic fallback.")
    elif llm_provider == "OpenAI":
        api_key_input = st.text_input("OpenAI API Key", type="password", value=os.environ.get("OPENAI_API_KEY", ""))
        llm_model = st.selectbox("OpenAI Model", ["gpt-4o-mini", "gpt-4o", "gpt-3.5-turbo"])
        if not api_key_input:
            st.info("No API key entered; system will use deterministic fallback.")
    else:
        st.caption("100% deterministic, rules-based mathematical synthesis with zero API dependencies.")

    st.markdown("---")
    if st.session_state.df is not None:
        st.markdown(f"**Active File:** `{st.session_state.dataset_name}`")
        st.markdown(f"**Dimensions:** `{len(st.session_state.df):,} rows × {len(st.session_state.df.columns)} cols`")
        if st.button("Clear Dataset & State", use_container_width=True):
            st.session_state.df = None
            st.session_state.dataset_name = None
            st.session_state.anomaly_results = None
            st.session_state.ml_results = None
            st.session_state.explainability_results = None
            st.session_state.insights_result = None
            st.rerun()

# ==========================================
# MAIN INTERFACE
# ==========================================
st.markdown('<div class="main-header">InsightAI</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-header">Automated Tabular Data Profiling, Quality Auditing, Anomaly Detection, Machine Learning & Decision Support</div>',
    unsafe_allow_html=True
)

if st.session_state.df is None:
    st.info("👈 Please load the sample dataset or upload a CSV file from the sidebar to begin analysis.")
    st.stop()

df = st.session_state.df

# Compute High-level Profile and Quality Metrics once for the active dataset
try:
    overview = get_dataset_overview(df)
    col_types = identify_column_types(df)
    quality_score = compute_data_quality_score(df)
except Exception as e:
    st.error(f"Error computing dataset profile: {str(e)}")
    st.stop()

# Define Navigation Tabs
tab_overview, tab_quality, tab_eda, tab_anomaly, tab_ml, tab_comp, tab_explain, tab_insights, tab_recs, tab_review = st.tabs([
    "📋 Overview",
    "🛡️ Data Quality",
    "📈 EDA",
    "🔍 Anomaly Detection",
    "🤖 ML Prediction",
    "⚖️ Model Comparison",
    "💡 Explainability",
    "🧠 AI Insights",
    "🎯 Recommendations",
    "📋 Analyst Review"
])

# ==========================================
# TAB 1: OVERVIEW & PROFILING
# ==========================================
with tab_overview:
    st.subheader("Dataset Profiling & Schema Overview")

    kpi1, kpi2, kpi3, kpi4, kpi5 = st.columns(5)
    with kpi1:
        st.markdown(
            f'<div class="metric-card"><div class="metric-value">{overview["num_rows"]:,}</div><div class="metric-label">Total Rows</div></div>',
            unsafe_allow_html=True
        )
    with kpi2:
        st.markdown(
            f'<div class="metric-card"><div class="metric-value">{overview["num_cols"]}</div><div class="metric-label">Columns</div></div>',
            unsafe_allow_html=True
        )
    with kpi3:
        st.markdown(
            f'<div class="metric-card"><div class="metric-value">{overview["total_missing"]:,}</div><div class="metric-label">Missing Cells ({overview["missing_pct"]}%)</div></div>',
            unsafe_allow_html=True
        )
    with kpi4:
        st.markdown(
            f'<div class="metric-card"><div class="metric-value">{overview["duplicate_rows"]:,}</div><div class="metric-label">Duplicates ({overview["duplicate_pct"]}%)</div></div>',
            unsafe_allow_html=True
        )
    with kpi5:
        st.markdown(
            f'<div class="metric-card"><div class="metric-value">{overview["memory_usage"]}</div><div class="metric-label">Memory Footprint</div></div>',
            unsafe_allow_html=True
        )

    st.markdown("---")
    st.markdown("#### Inferred Attribute Types")
    t1, t2, t3 = st.columns(3)
    with t1:
        st.markdown(f"**Numerical Features ({len(col_types['numerical'])}):**")
        st.write(", ".join(col_types["numerical"]) if col_types["numerical"] else "_None_")
    with t2:
        st.markdown(f"**Categorical Features ({len(col_types['categorical'])}):**")
        st.write(", ".join(col_types["categorical"]) if col_types["categorical"] else "_None_")
    with t3:
        st.markdown(f"**ID / Key Columns ({len(col_types['id_cols'])}):**")
        st.write(", ".join(col_types["id_cols"]) if col_types["id_cols"] else "_None detected_")

    st.markdown("---")
    st.markdown("#### Raw Dataset Explorer")
    st.dataframe(df.head(50), use_container_width=True)

    st.markdown("---")
    st.markdown("#### Statistical Summaries")
    num_summary, cat_summary = get_summary_statistics(df)
    
    st.markdown("##### Numerical Features Distribution Statistics")
    if not num_summary.empty:
        st.dataframe(num_summary, use_container_width=True)
    else:
        st.info("No numerical features detected.")

    st.markdown("##### Categorical Features Cardinality & Frequencies")
    if not cat_summary.empty:
        st.dataframe(cat_summary, use_container_width=True)
    else:
        st.info("No categorical features detected.")


# ==========================================
# TAB 2: DATA QUALITY AUDIT
# ==========================================
with tab_quality:
    st.subheader("Automated Data Quality Audit")

    # Score Card Banner
    score_col, status_col = st.columns([1, 2])
    with score_col:
        st.markdown(
            f"""
            <div class="metric-card" style="border-left: 6px solid #2563EB;">
                <div class="metric-value" style="font-size: 2.4rem;">{quality_score['score']}/100</div>
                <div class="metric-label">Composite Quality Index (Grade: {quality_score['grade']})</div>
                <div style="margin-top: 8px;"><span class="badge-good">{quality_score['health_status']} Health</span></div>
            </div>
            """,
            unsafe_allow_html=True
        )
    with status_col:
        c1, c2, c3 = st.columns(3)
        with c1:
            st.metric("Completeness Score", f"{quality_score['completeness_score']}%")
        with c2:
            st.metric("Uniqueness Score", f"{quality_score['uniqueness_score']}%")
        with c3:
            st.metric("Distribution Health", f"{quality_score['distribution_score']}%")
        
        if quality_score["critical_flags"]:
            st.warning("**Quality Alerts:**\n" + "\n".join([f"- {f}" for f in quality_score["critical_flags"]]))
        else:
            st.success("No severe structural defects detected in the active dataset.")

    st.markdown("---")
    st.markdown("#### 1. Missing Values Analysis")
    missing_data = analyze_missing_values(df)
    if missing_data["num_columns_with_missing"] > 0:
        st.write(f"Detected **{missing_data['total_missing_cells']} missing values** across **{missing_data['num_columns_with_missing']} columns**.")
        st.dataframe(missing_data["columns_with_missing"], use_container_width=True)
    else:
        st.success("Complete Dataset: Zero missing or null values found.")

    st.markdown("---")
    st.markdown("#### 2. Duplicate Observations Audit")
    dup_data = analyze_duplicates(df)
    if dup_data["has_duplicates"]:
        st.warning(f"Identified **{dup_data['duplicate_count']} exact duplicate rows** ({dup_data['duplicate_pct']}% of data).")
        st.dataframe(dup_data["sample_duplicates"], use_container_width=True)
    else:
        st.success("Zero duplicate rows detected. All observations are unique.")

    st.markdown("---")
    st.markdown("#### 3. Statistical Outliers Audit (IQR & Z-Score)")
    outlier_data = analyze_outliers(df)
    if not outlier_data["summary_table"].empty:
        st.write(f"Total rows with at least one IQR outlier: **{outlier_data['total_outlier_rows']}** ({outlier_data['total_outlier_pct']}%).")
        st.dataframe(outlier_data["summary_table"], use_container_width=True)
    else:
        st.info("No numerical features available for outlier analysis.")


# ==========================================
# TAB 3: EXPLORATORY DATA ANALYSIS (EDA)
# ==========================================
with tab_eda:
    st.subheader("Exploratory Data Analysis (EDA)")

    eda_mode = st.radio(
        "Select Visualization Mode:",
        ["Univariate Analysis", "Bivariate Analysis", "Correlation Matrix"],
        horizontal=True
    )

    if eda_mode == "Univariate Analysis":
        col_type_choice = st.selectbox("Feature Type:", ["Numerical", "Categorical"])
        if col_type_choice == "Numerical" and col_types["numerical"]:
            selected_num = st.selectbox("Select Numerical Feature:", col_types["numerical"])
            fig = plot_numerical_distribution(df, selected_num)
            st.plotly_chart(fig, use_container_width=True)
        elif col_type_choice == "Categorical" and col_types["categorical"]:
            selected_cat = st.selectbox("Select Categorical Feature:", col_types["categorical"])
            fig = plot_categorical_distribution(df, selected_cat)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("No features of the selected type are present in this dataset.")

    elif eda_mode == "Bivariate Analysis":
        biv_type = st.selectbox("Bivariate Mode:", ["Numerical vs. Numerical (Scatter)", "Numerical vs. Categorical (Box)"])
        if biv_type == "Numerical vs. Numerical (Scatter)":
            if len(col_types["numerical"]) >= 2:
                c1, c2, c3 = st.columns(3)
                with c1:
                    x_axis = st.selectbox("X-Axis Feature:", col_types["numerical"], index=0)
                with c2:
                    y_axis = st.selectbox("Y-Axis Feature:", col_types["numerical"], index=1)
                with c3:
                    color_by = st.selectbox("Color Grouping (Optional):", [None] + col_types["categorical"])
                fig = plot_bivariate_scatter(df, x_axis, y_axis, color_col=color_by)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("At least 2 numerical features required for scatter analysis.")
        else:
            if col_types["categorical"] and col_types["numerical"]:
                c1, c2 = st.columns(2)
                with c1:
                    cat_sel = st.selectbox("Categorical Feature:", col_types["categorical"])
                with c2:
                    num_sel = st.selectbox("Numerical Feature:", col_types["numerical"])
                fig = plot_bivariate_box(df, cat_sel, num_sel)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("Both numerical and categorical features are required for box plot analysis.")

    elif eda_mode == "Correlation Matrix":
        if len(col_types["numerical"]) >= 2:
            method = st.selectbox("Correlation Method:", ["pearson", "spearman"])
            fig = plot_correlation_heatmap(df, method=method)
            if fig is not None:
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Insufficient numerical features for correlation.")
        else:
            st.warning("At least 2 numerical features are required for correlation analysis.")


# ==========================================
# TAB 4: ANOMALY DETECTION
# ==========================================
with tab_anomaly:
    st.subheader("Isolation Forest Anomaly Detection")

    st.markdown("""
    <div class="advisory-box">
        <b>Analyst Review Advisory:</b> Isolation Forest identifies data points that are statistically isolated in multi-dimensional space.
        An anomaly indicates a mathematical deviation—it is <b>NOT automatically fraud, an error, or malicious activity</b>.
        Domain-specific validation is essential.
    </div>
    """, unsafe_allow_html=True)

    if not col_types["numerical"]:
        st.warning("Anomaly detection requires numerical features.")
    else:
        ctrl1, ctrl2, ctrl3 = st.columns([2, 2, 1])
        with ctrl1:
            anom_features = st.multiselect(
                "Select Numerical Dimensions for Anomaly Isolation:",
                options=col_types["numerical"],
                default=col_types["numerical"][:min(5, len(col_types["numerical"]))]
            )
        with ctrl2:
            contamination = st.slider(
                "Expected Contamination Rate (Anomaly %):",
                min_value=0.01,
                max_value=0.15,
                value=0.05,
                step=0.01,
                help="Estimated proportion of outliers in the data population."
            )
        with ctrl3:
            st.write("")
            st.write("")
            run_anom_btn = st.button("Run Anomaly Detection", type="primary", use_container_width=True)

        # Run or use cached results
        if run_anom_btn or st.session_state.anomaly_results is None:
            if anom_features:
                try:
                    with st.spinner("Executing Isolation Forest..."):
                        anom_res = detect_anomalies_isolation_forest(
                            df,
                            numerical_cols=anom_features,
                            contamination=contamination
                        )
                        st.session_state.anomaly_results = anom_res
                except Exception as e:
                    st.error(f"Anomaly detection failed: {str(e)}")
            else:
                st.warning("Please select at least one numerical feature.")

        if st.session_state.anomaly_results is not None:
            res = st.session_state.anomaly_results
            k1, k2, k3 = st.columns(3)
            with k1:
                st.metric("Total Observations", f"{res['total_records']:,}")
            with k2:
                st.metric("Detected Anomalies", f"{res['anomaly_count']:,}")
            with k3:
                st.metric("Anomaly Rate", f"{res['anomaly_percentage']}%")

            # Visualizations
            st.markdown("---")
            if len(anom_features) >= 2:
                v1, v2 = st.columns(2)
                with v1:
                    x_anom = st.selectbox("X-Axis Feature for Visualization:", anom_features, index=0)
                with v2:
                    y_anom = st.selectbox("Y-Axis Feature for Visualization:", anom_features, index=1)
                
                fig_anom = plot_anomaly_distribution(res["annotated_df"], x_anom, y_anom)
                st.plotly_chart(fig_anom, use_container_width=True)

            fig_hist = plot_anomaly_score_histogram(res["annotated_df"])
            st.plotly_chart(fig_hist, use_container_width=True)

            st.markdown("---")
            st.markdown("#### Inspect Anomalous Observations")
            st.caption("Sorted by lowest decision score (most mathematically isolated first)")
            if not res["anomalous_rows"].empty:
                st.dataframe(res["anomalous_rows"].head(50), use_container_width=True)
            else:
                st.info("Zero anomalies detected under current contamination threshold.")


# ==========================================
# TAB 5: ML PREDICTION
# ==========================================
with tab_ml:
    st.subheader("Supervised Machine Learning Modeling")

    all_cols = list(df.columns)
    default_target_idx = 0
    if "Churn" in all_cols:
        default_target_idx = all_cols.index("Churn")
    elif len(all_cols) > 0:
        default_target_idx = len(all_cols) - 1

    m_col1, m_col2, m_col3 = st.columns([2, 1, 1])
    with m_col1:
        target_col = st.selectbox("Select Target Variable to Predict:", all_cols, index=default_target_idx)
    with m_col2:
        test_size = st.slider("Holdout Test Split (%):", min_value=10, max_value=40, value=20, step=5) / 100.0
    with m_col3:
        st.write("")
        st.write("")
        train_btn = st.button("Train ML Models", type="primary", use_container_width=True)

    # Task detection preview
    task_info = detect_problem_type(df, target_col)
    st.info(
        f"**Inferred Task:** `{task_info['problem_type']}` | "
        f"**Target Cardinality:** `{task_info['unique_values_count']}` unique values | "
        f"**Sample Values:** `{task_info['sample_values']}`"
    )

    if train_btn or st.session_state.ml_results is None:
        try:
            with st.spinner("Building leakage-free pipeline, fitting baseline and strong models..."):
                ml_res = train_and_evaluate_models(
                    df,
                    target_col=target_col,
                    test_size=test_size,
                    random_state=42
                )
                st.session_state.ml_results = ml_res
                # Also auto-extract explainability
                if "strong_model" in ml_res and "feature_names" in ml_res:
                    st.session_state.explainability_results = extract_feature_importance(
                        ml_res["strong_model"],
                        ml_res["feature_names"],
                        top_n=15
                    )
        except Exception as e:
            st.error(f"Model training failed: {str(e)}")

    if st.session_state.ml_results is not None:
        res = st.session_state.ml_results
        st.success(f"Modeling complete for **{res['target_col']}** ({res['task_type']}) on {res['train_samples']} train & {res['test_samples']} test records.")

        if res["task_type"] == "Classification":
            strong = res["strong_metrics"]
            b1, b2, b3, b4 = st.columns(4)
            with b1:
                st.metric("Test Accuracy", f"{strong['accuracy']*100:.1f}%")
            with b2:
                st.metric("Weighted F1-Score", f"{strong['f1']:.3f}")
            with b3:
                st.metric("Weighted Precision", f"{strong['precision']:.3f}")
            with b4:
                st.metric("Weighted Recall", f"{strong['recall']:.3f}")

            st.markdown("---")
            st.markdown("#### Classification Confusion Matrix")
            cm = res["confusion_matrix"]
            labels = [res["label_mapping"][i] for i in range(len(res["label_mapping"]))] if res.get("label_mapping") else None
            fig_cm = plot_confusion_matrix_heatmap(cm, labels=labels)
            st.plotly_chart(fig_cm, use_container_width=True)

        else:
            strong = res["strong_metrics"]
            b1, b2, b3 = st.columns(3)
            with b1:
                st.metric("R² Variance Explained", f"{strong['r2']:.3f}")
            with b2:
                st.metric("Root Mean Squared Error (RMSE)", f"{strong['rmse']:.2f}")
            with b3:
                st.metric("Mean Absolute Error (MAE)", f"{strong['mae']:.2f}")

            st.markdown("---")
            st.markdown("#### Regression Diagnostic Plots")
            r1, r2 = st.columns(2)
            with r1:
                fig_reg = plot_regression_actual_vs_pred(res["y_test"], res["y_pred"])
                st.plotly_chart(fig_reg, use_container_width=True)
            with r2:
                fig_res = plot_regression_residuals(res["y_test"], res["y_pred"])
                st.plotly_chart(fig_res, use_container_width=True)


# ==========================================
# TAB 6: MODEL COMPARISON
# ==========================================
with tab_comp:
    st.subheader("Baseline vs. Strong Model Benchmark Comparison")

    if st.session_state.ml_results is None:
        st.info("Train machine learning models in the **ML Prediction** tab first to see the performance comparison.")
    else:
        res = st.session_state.ml_results
        st.markdown(f"Comparing naive baseline model against Random Forest on target **`{res['target_col']}`**:")
        st.dataframe(res["comparison_df"], use_container_width=True)

        st.markdown("""
        <div class="advisory-box">
            <b>Why Baseline Comparison Matters:</b>
            A naive baseline represents the performance of an uninformed heuristic (e.g. always predicting the majority class or the dataset mean).
            Any production-ready model must demonstrate statistically significant lift over this baseline to justify operational deployment.
        </div>
        """, unsafe_allow_html=True)


# ==========================================
# TAB 7: EXPLAINABILITY
# ==========================================
with tab_explain:
    st.subheader("Model Feature Importance & Explainability")

    st.markdown("""
    <div class="warning-box">
        <b>Causality Caution:</b> Feature importance indicates how much a feature helped the tree model partition data and reduce prediction loss.
        <b>Feature importance DOES NOT equal causality.</b> High importance does not guarantee that altering this variable in the real world will produce an outcome change.
    </div>
    """, unsafe_allow_html=True)

    if st.session_state.explainability_results is None:
        st.info("Train machine learning models in the **ML Prediction** tab to generate feature importance.")
    else:
        exp = st.session_state.explainability_results
        e1, e2 = st.columns([1, 3])
        with e1:
            st.metric("Top Predictive Driver", exp["top_feature"])
            st.metric("Top Feature Influence Share", f"{exp['top_feature_share']:.1f}%")
        with e2:
            fig_bar = plot_feature_importance_bar(exp["top_table"])
            st.plotly_chart(fig_bar, use_container_width=True)

        st.markdown("---")
        st.markdown("#### Feature Importance Ranking Table")
        st.dataframe(exp["top_table"], use_container_width=True)


# ==========================================
# TAB 8: AI INSIGHTS
# ==========================================
with tab_insights:
    st.subheader("Natural Language AI Insights Generation")

    st.caption("Synthesizes structured metrics across profiling, data quality, anomaly isolation, model performance, and feature drivers.")

    gen_insights_btn = st.button("Generate Executive Insights", type="primary", use_container_width=True)

    if gen_insights_btn or st.session_state.insights_result is None:
        with st.spinner("Synthesizing analytics report..."):
            payload = build_analytics_payload(
                overview=overview,
                quality=quality_score,
                anomalies=st.session_state.anomaly_results,
                ml_results=st.session_state.ml_results,
                feature_importance=st.session_state.explainability_results
            )

            res_insights = generate_llm_insights(
                payload=payload,
                api_key=api_key_input,
                provider="gemini" if llm_provider == "Google Gemini" else ("openai" if llm_provider == "OpenAI" else "deterministic"),
                model_name=llm_model
            )
            st.session_state.insights_result = res_insights

    if st.session_state.insights_result is not None:
        ins = st.session_state.insights_result
        st.info(f"**Report Source:** `{ins['source']}` | **Factual Guardrails:** Strict payload grounding (zero fabricated metrics)")
        st.markdown(ins["insights_markdown"])

        st.markdown("---")
        st.download_button(
            label="Download Insights Report (Markdown)",
            data=ins["insights_markdown"],
            file_name="InsightAI_Report.md",
            mime="text/markdown"
        )


# ==========================================
# TAB 9: RECOMMENDATIONS
# ==========================================
with tab_recs:
    st.subheader("Data-Driven Recommendations")

    st.markdown("""
    <div class="advisory-box">
        <b>Data-Driven Suggestions:</b> The following analytical recommendations are derived solely from computed dataset metrics and model diagnostics.
        These are suggestions for investigation, <b>not guaranteed conclusions</b>.
    </div>
    """, unsafe_allow_html=True)

    # 1. Data Quality Recommendations
    st.markdown("#### 1. Data Quality & Ingestion Actions")
    if quality_score["completeness_score"] < 100:
        st.markdown(f"- **Address Missingness:** Total of **{overview['total_missing']}** missing cells detected. Ensure automated imputation pipelines (such as median/mode imputation used here) are integrated into upstream ETL pipelines.")
    else:
        st.markdown("- **Ingestion Cleanliness:** Ingested dataset has 100% completeness. Maintain current validation constraints.")

    if overview["duplicate_rows"] > 0:
        st.markdown(f"- **Deduplication:** Remove **{overview['duplicate_rows']}** exact duplicate records prior to downstream operational consumption.")
    else:
        st.markdown("- **Record Integrity:** No duplicate records detected.")

    # 2. Anomaly Recommendations
    st.markdown("#### 2. Anomaly & Risk Actions")
    if st.session_state.anomaly_results is not None:
        anom = st.session_state.anomaly_results
        st.markdown(f"- **Audit Outliers:** Focus analyst review on the **{anom['anomaly_count']} flagged anomalous rows** ({anom['anomaly_percentage']}% of total). Verify whether these reflect high-value outliers, telemetry errors, or emerging user behaviors.")
    else:
        st.markdown("- Run Anomaly Detection in the respective tab to generate specific anomaly action items.")

    # 3. Model Deployment Recommendations
    st.markdown("#### 3. Machine Learning Operationalization")
    if st.session_state.ml_results is not None:
        ml = st.session_state.ml_results
        if ml["task_type"] == "Classification":
            strong_acc = ml["strong_metrics"]["accuracy"]
            base_acc = ml["baseline_metrics"]["accuracy"]
            delta = (strong_acc - base_acc) * 100
            if delta > 10:
                st.markdown(f"- **Candidate Readiness:** Strong model exhibits substantial accuracy lift (+{delta:.1f}%) over naive baseline. Conduct shadow-mode testing in production.")
            else:
                st.markdown("- **Iterate Modeling:** Strong model shows modest lift over baseline. Gather additional domain features to improve discriminative power.")
        else:
            r2 = ml["strong_metrics"]["r2"]
            if r2 > 0.5:
                st.markdown(f"- **Decision Support Feasibility:** Strong model explains **{r2*100:.1f}%** of target variance. Suitable for assistive decision support.")
            else:
                st.markdown("- **Model Refinement:** Target variance explained is below 50%. Explore nonlinear interactions or external data sources.")
    else:
        st.markdown("- Train models in the ML Prediction tab to view model operationalization suggestions.")


# ==========================================
# TAB 10: ANALYST REVIEW & GOVERNANCE
# ==========================================
with tab_review:
    st.subheader("Analyst Review & Responsible AI Governance")

    st.markdown("""
    All machine learning models and automated insights require critical human oversight prior to business execution.
    Use the following governance checklist to audit findings.
    """)

    st.markdown("#### Analyst Verification Checklist")
    chk1 = st.checkbox("1. Correlation vs. Causation: Validated that high feature importance is not mistakenly interpreted as causal business levers.", value=True)
    chk2 = st.checkbox("2. Anomaly Context: Confirmed that statistical anomalies are investigated as deviations, not automatically flagged as fraud or defect.", value=True)
    chk3 = st.checkbox("3. Data Hygiene & Drift: Verified that current data reflects target operational distributions and lacks temporal leakage.", value=True)
    chk4 = st.checkbox("4. AI Insight Review: Audited generated natural language insights against verified tabular metrics to ensure zero hallucination.", value=True)

    st.markdown("---")
    st.markdown("#### Operational Risk Matrix")
    risk_df = pd.DataFrame({
        "Risk Dimension": ["Data Quality Risk", "Model Overfitting Risk", "Interpretability Risk", "Operational Decision Risk"],
        "Severity": ["Low", "Medium", "Medium", "High"],
        "Mitigation Protocol": [
            "Continuous schema validation and null-threshold alerts.",
            "Strict holdout validation, cross-validation, and baseline comparison.",
            "Feature importance accompanied by causality disclaimer banners.",
            "Mandatory human-in-the-loop review before any automated action."
        ]
    })
    st.table(risk_df)

    st.markdown("---")
    st.caption("InsightAI System v1.0.0 | Built with Python, Streamlit, Scikit-Learn & Plotly")
