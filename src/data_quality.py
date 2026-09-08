"""
data_quality.py
Automated data quality auditing: missing values, duplicates, outliers, and composite scoring.
"""

from typing import Dict, Any, List
import pandas as pd
import numpy as np


def analyze_missing_values(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Computes missing value metrics per column and for the entire dataset.
    """
    total_rows = len(df)
    missing_counts = df.isna().sum()
    missing_pcts = (missing_counts / total_rows * 100).round(2)

    missing_df = pd.DataFrame({
        "Column": df.columns,
        "Missing Count": missing_counts.values,
        "Missing %": missing_pcts.values,
        "Data Type": [str(df[c].dtype) for c in df.columns]
    }).sort_values(by="Missing Count", ascending=False)

    cols_with_missing = missing_df[missing_df["Missing Count"] > 0]
    complete_rows = int(df.dropna().shape[0])
    complete_rows_pct = round(complete_rows / total_rows * 100, 2) if total_rows > 0 else 0.0

    return {
        "summary_table": missing_df,
        "columns_with_missing": cols_with_missing,
        "num_columns_with_missing": len(cols_with_missing),
        "total_missing_cells": int(missing_counts.sum()),
        "complete_rows": complete_rows,
        "complete_rows_pct": complete_rows_pct
    }


def analyze_duplicates(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Identifies exact duplicate rows in the dataset.
    """
    total_rows = len(df)
    dup_mask = df.duplicated(keep=False)
    dup_count = int(df.duplicated(keep="first").sum())
    dup_pct = round((dup_count / total_rows * 100), 2) if total_rows > 0 else 0.0

    duplicated_rows_sample = df[dup_mask].head(20) if dup_count > 0 else pd.DataFrame()

    return {
        "duplicate_count": dup_count,
        "duplicate_pct": dup_pct,
        "has_duplicates": dup_count > 0,
        "sample_duplicates": duplicated_rows_sample
    }


def analyze_outliers(df: pd.DataFrame, method: str = "iqr") -> Dict[str, Any]:
    """
    Identifies numerical outliers using the Interquartile Range (IQR) rule:
    Lower bound = Q1 - 1.5 * IQR
    Upper bound = Q3 + 1.5 * IQR
    Also provides Z-score based counts (|z| > 3).
    """
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    results = []
    outlier_indices_all = set()

    for col in num_cols:
        series = df[col].dropna()
        n = len(series)
        if n < 5:
            continue

        q1 = float(series.quantile(0.25))
        q3 = float(series.quantile(0.75))
        iqr = q3 - q1

        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr

        outliers_iqr = series[(series < lower_bound) | (series > upper_bound)]
        outlier_count_iqr = len(outliers_iqr)
        outlier_pct_iqr = round((outlier_count_iqr / n * 100), 2) if n > 0 else 0.0

        # Z-score method
        std = series.std()
        if std > 0:
            z_scores = np.abs((series - series.mean()) / std)
            outliers_z = series[z_scores > 3.0]
            outlier_count_z = len(outliers_z)
        else:
            outlier_count_z = 0

        outlier_indices_all.update(outliers_iqr.index.tolist())

        results.append({
            "Column": col,
            "Q1 (25%)": round(q1, 3),
            "Q3 (75%)": round(q3, 3),
            "IQR": round(iqr, 3),
            "Lower Bound": round(lower_bound, 3),
            "Upper Bound": round(upper_bound, 3),
            "Outliers (IQR)": outlier_count_iqr,
            "Outliers % (IQR)": outlier_pct_iqr,
            "Outliers (Z > 3)": outlier_count_z
        })

    outlier_summary_df = pd.DataFrame(results)
    total_outlier_rows = len(outlier_indices_all)
    total_rows = len(df)
    total_outlier_pct = round(total_outlier_rows / total_rows * 100, 2) if total_rows > 0 else 0.0

    return {
        "summary_table": outlier_summary_df,
        "total_outlier_rows": total_outlier_rows,
        "total_outlier_pct": total_outlier_pct,
        "outlier_indices": list(outlier_indices_all)
    }


def compute_data_quality_score(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Computes a multi-dimensional Data Quality Index (0 to 100)
    evaluating Completeness (40%), Uniqueness (30%), and Distribution Health (30%).
    """
    total_rows, total_cols = df.shape
    total_cells = total_rows * total_cols

    if total_cells == 0:
        return {
            "score": 0.0,
            "grade": "F",
            "completeness_score": 0.0,
            "uniqueness_score": 0.0,
            "distribution_score": 0.0,
            "health_status": "Critical",
            "critical_flags": ["Dataset is empty"]
        }

    # 1. Completeness Score (0-100)
    missing_cells = df.isna().sum().sum()
    missing_rate = missing_cells / total_cells
    completeness_score = max(0.0, 100.0 - (missing_rate * 250.0))

    # 2. Uniqueness Score (0-100)
    dup_rows = df.duplicated().sum()
    dup_rate = dup_rows / total_rows
    uniqueness_score = max(0.0, 100.0 - (dup_rate * 200.0))

    # 3. Distribution & Outlier Health (0-100)
    num_df = df.select_dtypes(include=[np.number])
    if not num_df.empty:
        outlier_data = analyze_outliers(df)
        outlier_rate = outlier_data["total_outlier_rows"] / total_rows
        # Moderate outliers are expected in real data, penalize excessive (>10%)
        distribution_score = max(0.0, 100.0 - (max(0.0, outlier_rate - 0.03) * 200.0))
    else:
        distribution_score = 100.0

    # Weighted Composite Score
    composite_score = round(
        (completeness_score * 0.40) + (uniqueness_score * 0.30) + (distribution_score * 0.30),
        1
    )

    # Health status & Grade
    if composite_score >= 90:
        grade = "A"
        health = "Excellent"
        badge_color = "green"
    elif composite_score >= 75:
        grade = "B"
        health = "Good"
        badge_color = "blue"
    elif composite_score >= 60:
        grade = "C"
        health = "Fair"
        badge_color = "orange"
    else:
        grade = "D"
        health = "Needs Attention"
        badge_color = "red"

    critical_flags = []
    if missing_rate > 0.10:
        critical_flags.append(f"High missingness: {missing_rate*100:.1f}% of cells are missing.")
    if dup_rate > 0.05:
        critical_flags.append(f"High duplicate rate: {dup_rate*100:.1f}% rows are exact duplicates.")
    if not num_df.empty and outlier_rate > 0.15:
        critical_flags.append(f"Elevated outlier presence: {outlier_rate*100:.1f}% rows exhibit extreme values.")

    return {
        "score": composite_score,
        "grade": grade,
        "health_status": health,
        "badge_color": badge_color,
        "completeness_score": round(completeness_score, 1),
        "uniqueness_score": round(uniqueness_score, 1),
        "distribution_score": round(distribution_score, 1),
        "critical_flags": critical_flags
    }
