"""
eda.py
Interactive Plotly visualizations for Exploratory Data Analysis.
"""

from typing import Optional, List
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go


def plot_numerical_distribution(df: pd.DataFrame, column: str) -> go.Figure:
    """
    Renders an interactive distribution plot (histogram + box marginal)
    with mean and median markers.
    """
    series = df[column].dropna()
    mean_val = series.mean()
    median_val = series.median()

    fig = px.histogram(
        df,
        x=column,
        marginal="box",
        nbins=35,
        title=f"Distribution of <b>{column}</b>",
        color_discrete_sequence=["#2563EB"],
        opacity=0.85
    )

    fig.add_vline(
        x=mean_val,
        line_dash="dash",
        line_color="#DC2626",
        annotation_text=f"Mean: {mean_val:.2f}",
        annotation_position="top right"
    )
    fig.add_vline(
        x=median_val,
        line_dash="dot",
        line_color="#059669",
        annotation_text=f"Median: {median_val:.2f}",
        annotation_position="top left"
    )

    fig.update_layout(
        template="plotly_white",
        bargap=0.08,
        hovermode="x unified",
        margin=dict(l=40, r=40, t=60, b=40),
        xaxis_title=column,
        yaxis_title="Frequency"
    )
    return fig


def plot_categorical_distribution(df: pd.DataFrame, column: str, max_categories: int = 15) -> go.Figure:
    """
    Renders an interactive bar chart of categorical frequencies and percentages.
    """
    counts = df[column].value_counts(dropna=False).reset_index()
    counts.columns = [column, "Count"]

    total = counts["Count"].sum()
    counts["Percentage"] = (counts["Count"] / total * 100).round(1)
    counts[column] = counts[column].fillna("Missing / Null").astype(str)

    if len(counts) > max_categories:
        top = counts.iloc[:max_categories]
        other_count = counts.iloc[max_categories:]["Count"].sum()
        other_pct = round(other_count / total * 100, 1)
        other_row = pd.DataFrame([{column: "Other (grouped)", "Count": other_count, "Percentage": other_pct}])
        counts = pd.concat([top, other_row], ignore_index=True)

    fig = px.bar(
        counts,
        x=column,
        y="Count",
        text=counts.apply(lambda r: f"{r['Count']} ({r['Percentage']}%)", axis=1),
        title=f"Frequency Breakdown of <b>{column}</b>",
        color="Count",
        color_continuous_scale="Viridis"
    )

    fig.update_traces(textposition="outside")
    fig.update_layout(
        template="plotly_white",
        margin=dict(l=40, r=40, t=60, b=50),
        xaxis_title=column,
        yaxis_title="Count",
        coloraxis_showscale=False
    )
    return fig


def plot_correlation_heatmap(df: pd.DataFrame, method: str = "pearson") -> Optional[go.Figure]:
    """
    Renders an annotated correlation heatmap for numerical features.
    Returns None if fewer than 2 numerical features exist.
    """
    num_df = df.select_dtypes(include=[np.number])
    if num_df.shape[1] < 2:
        return None

    corr = num_df.corr(method=method).round(2)

    fig = px.imshow(
        corr,
        text_auto=True,
        aspect="auto",
        color_continuous_scale="RdBu_r",
        zmin=-1,
        zmax=1,
        title=f"Feature Correlation Heatmap ({method.capitalize()})"
    )

    fig.update_layout(
        template="plotly_white",
        margin=dict(l=50, r=50, t=60, b=50),
        xaxis_tickangle=-45
    )
    return fig


def plot_bivariate_scatter(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    color_col: Optional[str] = None
) -> go.Figure:
    """
    Renders an interactive scatter plot with trendline between two continuous features.
    """
    fig = px.scatter(
        df,
        x=x_col,
        y=y_col,
        color=color_col,
        trendline="ols" if color_col is None else None,
        title=f"Bivariate Analysis: <b>{x_col}</b> vs <b>{y_col}</b>",
        opacity=0.75,
        template="plotly_white"
    )

    fig.update_traces(marker=dict(size=7, line=dict(width=0.5, color="DarkSlateGrey")))
    fig.update_layout(
        margin=dict(l=40, r=40, t=60, b=40),
        hovermode="closest"
    )
    return fig


def plot_bivariate_box(df: pd.DataFrame, cat_col: str, num_col: str) -> go.Figure:
    """
    Renders a box plot showing a numerical distribution conditioned on a categorical feature.
    """
    fig = px.box(
        df,
        x=cat_col,
        y=num_col,
        color=cat_col,
        title=f"Distribution of <b>{num_col}</b> grouped by <b>{cat_col}</b>",
        template="plotly_white",
        points="outliers"
    )

    fig.update_layout(
        margin=dict(l=40, r=40, t=60, b=40),
        showlegend=False
    )
    return fig
