"""
anomaly_detection.py
Isolation Forest unsupervised anomaly detection with multi-dimensional scoring and diagnostics.
"""

from typing import Dict, Any, List, Optional
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler
import plotly.express as px
import plotly.graph_objects as go


def detect_anomalies_isolation_forest(
    df: pd.DataFrame,
    numerical_cols: Optional[List[str]] = None,
    contamination: float = 0.05,
    random_state: int = 42
) -> Dict[str, Any]:
    """
    Fits an Isolation Forest model on numerical features to identify anomalous observations.
    
    Parameters:
    - df: Input dataframe.
    - numerical_cols: Specific numerical features to use. If None, auto-selects valid numeric columns.
    - contamination: Proportion of outliers in the data set (0.01 to 0.20).
    - random_state: Seed for reproducibility.
    
    Returns:
    - Dictionary with model results, summary metrics, annotated dataframe, and anomalous rows.
    """
    if df.empty:
        raise ValueError("Cannot perform anomaly detection on an empty dataset.")

    if numerical_cols is None:
        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        # Filter out constant columns
        numerical_cols = [c for c in num_cols if df[c].nunique(dropna=True) > 1]

    if len(numerical_cols) < 1:
        raise ValueError("At least one non-constant numerical column is required for Isolation Forest.")

    # Extract numerical matrix
    X_raw = df[numerical_cols].copy()

    # Median imputation for missing values
    imputer = SimpleImputer(strategy="median")
    X_imputed = imputer.fit_transform(X_raw)

    # Robust scaling (resilient to existing outliers)
    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(X_imputed)

    # Fit Isolation Forest
    iso_forest = IsolationForest(
        contamination=contamination,
        random_state=random_state,
        n_estimators=100,
        n_jobs=-1
    )
    predictions = iso_forest.fit_predict(X_scaled)
    # Decision function: average anomaly score of base estimators. Lower = more abnormal
    scores = iso_forest.decision_function(X_scaled)

    # Annotate original dataframe
    annotated_df = df.copy()
    # 1: Normal, -1: Anomaly
    annotated_df["anomaly_label"] = np.where(predictions == -1, "Anomaly", "Normal")
    annotated_df["anomaly_score"] = np.round(scores, 4)

    total_records = len(df)
    anomaly_count = int((predictions == -1).sum())
    anomaly_pct = round((anomaly_count / total_records * 100), 2) if total_records > 0 else 0.0

    # Extract anomalous records sorted by lowest score (most extreme first)
    anomalous_rows = annotated_df[annotated_df["anomaly_label"] == "Anomaly"].sort_values(
        by="anomaly_score", ascending=True
    )

    return {
        "features_used": numerical_cols,
        "contamination": contamination,
        "total_records": total_records,
        "anomaly_count": anomaly_count,
        "normal_count": total_records - anomaly_count,
        "anomaly_percentage": anomaly_pct,
        "annotated_df": annotated_df,
        "anomalous_rows": anomalous_rows,
        "model": iso_forest
    }


def plot_anomaly_distribution(
    annotated_df: pd.DataFrame,
    x_col: str,
    y_col: str
) -> go.Figure:
    """
    Renders an interactive 2D scatter plot highlighting anomalies versus normal records.
    """
    color_discrete_map = {
        "Normal": "#3B82F6",
        "Anomaly": "#EF4444"
    }

    fig = px.scatter(
        annotated_df,
        x=x_col,
        y=y_col,
        color="anomaly_label",
        color_discrete_map=color_discrete_map,
        title=f"Anomaly Detection Scatter: <b>{x_col}</b> vs <b>{y_col}</b>",
        hover_data=["anomaly_score"],
        opacity=0.8,
        template="plotly_white"
    )

    fig.update_traces(
        marker=dict(size=8, line=dict(width=0.6, color="#1E293B"))
    )
    fig.update_layout(
        margin=dict(l=40, r=40, t=60, b=40),
        legend_title_text="Status",
        hovermode="closest"
    )
    return fig


def plot_anomaly_score_histogram(annotated_df: pd.DataFrame) -> go.Figure:
    """
    Renders the distribution of anomaly decision function scores.
    """
    fig = px.histogram(
        annotated_df,
        x="anomaly_score",
        color="anomaly_label",
        nbins=40,
        color_discrete_map={"Normal": "#3B82F6", "Anomaly": "#EF4444"},
        title="Anomaly Decision Function Score Distribution",
        template="plotly_white",
        barmode="overlay",
        opacity=0.75
    )

    fig.update_layout(
        xaxis_title="Anomaly Score (Lower = More Anomalous)",
        yaxis_title="Record Count",
        margin=dict(l=40, r=40, t=60, b=40)
    )
    return fig
