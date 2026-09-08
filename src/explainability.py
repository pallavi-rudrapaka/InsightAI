"""
explainability.py
Model explainability: Tree-based feature importance extraction, visual diagnostics, and causality warnings.
"""

from typing import Dict, Any, List, Optional
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go


def extract_feature_importance(
    model: Any,
    feature_names: List[str],
    top_n: int = 15
) -> Dict[str, Any]:
    """
    Extracts Gini / variance-reduction feature importance from tree-based estimators.
    Normalizes importances and ranks top predictive drivers.
    """
    if not hasattr(model, "feature_importances_"):
        raise ValueError("Model does not provide feature_importances_ attribute.")

    importances = model.feature_importances_

    # Match length safely
    min_len = min(len(feature_names), len(importances))
    feat_subset = feature_names[:min_len]
    imp_subset = importances[:min_len]

    # Clean feature names (remove encoding prefixes if present)
    clean_names = [
        f.replace("remainder__", "").replace("onehot__", "").replace("standardscaler__", "")
        for f in feat_subset
    ]

    importance_df = pd.DataFrame({
        "Feature": clean_names,
        "Importance": imp_subset
    })

    # Sort descending
    importance_df = importance_df.sort_values(by="Importance", ascending=False).reset_index(drop=True)

    # Compute percentage and cumulative percentage
    total_imp = importance_df["Importance"].sum()
    if total_imp > 0:
        importance_df["Importance %"] = (importance_df["Importance"] / total_imp * 100).round(2)
        importance_df["Cumulative %"] = importance_df["Importance %"].cumsum().round(2)
    else:
        importance_df["Importance %"] = 0.0
        importance_df["Cumulative %"] = 0.0

    top_df = importance_df.head(top_n).copy()

    return {
        "full_table": importance_df,
        "top_table": top_df,
        "top_feature": top_df.iloc[0]["Feature"] if not top_df.empty else "None",
        "top_feature_share": top_df.iloc[0]["Importance %"] if not top_df.empty else 0.0
    }


def plot_feature_importance_bar(top_importance_df: pd.DataFrame) -> go.Figure:
    """
    Renders an interactive horizontal bar chart of top feature importances.
    """
    # Sort ascending for horizontal bar so highest is at top
    plot_df = top_importance_df.sort_values(by="Importance", ascending=True)

    fig = px.bar(
        plot_df,
        x="Importance",
        y="Feature",
        orientation="h",
        text=plot_df["Importance %"].apply(lambda v: f"{v:.1f}%"),
        title="<b>Top Predictive Feature Importance</b> (Tree-Based)",
        color="Importance",
        color_continuous_scale="Blues"
    )

    fig.update_traces(textposition="outside")
    fig.update_layout(
        template="plotly_white",
        margin=dict(l=60, r=40, t=60, b=40),
        xaxis_title="Relative Importance Score",
        yaxis_title="Feature Name",
        coloraxis_showscale=False
    )
    return fig
