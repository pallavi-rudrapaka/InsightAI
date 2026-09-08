"""
data_loader.py
Dataset ingestion, schema inference, profiling, and summary statistics.
"""

from typing import Dict, Any, List, Tuple, Union
import io
import pandas as pd
import numpy as np


def load_data(file_input: Union[str, io.BytesIO, io.StringIO, Any]) -> pd.DataFrame:
    """
    Load a CSV dataset from a file path or file-like buffer.
    Supports UTF-8, Latin-1 fallback, and auto-detects common delimiters.
    """
    if file_input is None:
        raise ValueError("No file provided.")

    encodings = ["utf-8", "latin-1", "cp1252"]
    df = None
    last_err = None

    for enc in encodings:
        try:
            if hasattr(file_input, "seek"):
                file_input.seek(0)
            df = pd.read_csv(file_input, encoding=enc, sep=None, engine="python")
            break
        except Exception as e:
            last_err = e
            continue

    if df is None:
        raise ValueError(f"Failed to load dataset: {str(last_err)}")

    if df.empty:
        raise ValueError("The uploaded CSV file is empty.")

    # Strip column names whitespace
    df.columns = [str(c).strip() for c in df.columns]
    return df


def get_dataset_overview(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Computes high-level profile metrics of the dataset.
    """
    num_rows, num_cols = df.shape
    total_cells = num_rows * num_cols
    total_missing = int(df.isna().sum().sum())
    missing_pct = (total_missing / total_cells * 100.0) if total_cells > 0 else 0.0
    duplicate_rows = int(df.duplicated().sum())
    duplicate_pct = (duplicate_rows / num_rows * 100.0) if num_rows > 0 else 0.0
    memory_bytes = int(df.memory_usage(deep=True).sum())

    if memory_bytes > 1024 * 1024:
        memory_str = f"{memory_bytes / (1024 * 1024):.2f} MB"
    else:
        memory_str = f"{memory_bytes / 1024:.2f} KB"

    return {
        "num_rows": num_rows,
        "num_cols": num_cols,
        "total_cells": total_cells,
        "total_missing": total_missing,
        "missing_pct": round(missing_pct, 2),
        "duplicate_rows": duplicate_rows,
        "duplicate_pct": round(duplicate_pct, 2),
        "memory_usage": memory_str,
        "columns": list(df.columns)
    }


def identify_column_types(df: pd.DataFrame) -> Dict[str, List[str]]:
    """
    Categorizes columns into:
    - numerical: numeric types with meaningful variance
    - categorical: string, categorical, boolean, or low-cardinality discrete
    - datetime: parsable datetime objects
    - id_cols: identifier columns (cardinality == rows or name pattern)
    - constant_cols: single unique value
    """
    numerical = []
    categorical = []
    datetime_cols = []
    id_cols = []
    constant_cols = []

    for col in df.columns:
        series = df[col]
        nunique = series.nunique(dropna=True)
        col_lower = col.lower()

        # Check for constant columns
        if nunique <= 1:
            constant_cols.append(col)
            continue

        # Check for ID-like columns
        if (
            (nunique == len(df) and ("id" in col_lower or "code" in col_lower or "key" in col_lower))
            or (col_lower.endswith("_id") or col_lower == "id" or col_lower.endswith("id"))
        ) and (series.dtype == "object" or nunique == len(df)):
            id_cols.append(col)
            continue

        # Check datetime
        if pd.api.types.is_datetime64_any_dtype(series):
            datetime_cols.append(col)
            continue
        elif series.dtype == "object":
            # Try sniffing datetime if sampled entries look like dates
            try:
                import warnings
                sample = series.dropna().head(10)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    if len(sample) > 0 and pd.to_datetime(sample, errors="coerce", format="mixed").notna().all():
                        datetime_cols.append(col)
                        continue
            except Exception:
                pass

        # Check numerical vs categorical
        if pd.api.types.is_numeric_dtype(series):
            # If boolean (0 and 1) or very low cardinality with object-like semantics
            if nunique == 2 and set(series.dropna().unique()).issubset({0, 1}):
                # Can be treated as numerical or binary categorical
                numerical.append(col)
            else:
                numerical.append(col)
        else:
            categorical.append(col)

    return {
        "numerical": numerical,
        "categorical": categorical,
        "datetime": datetime_cols,
        "id_cols": id_cols,
        "constant_cols": constant_cols
    }


def get_summary_statistics(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Computes statistical profiles for numerical and categorical variables.
    Returns (numerical_summary_df, categorical_summary_df).
    """
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()

    # Numerical summary
    num_summary = pd.DataFrame()
    if num_cols:
        stats = []
        for col in num_cols:
            s = df[col].dropna()
            if len(s) == 0:
                continue
            stats.append({
                "Column": col,
                "Count": int(s.count()),
                "Missing": int(df[col].isna().sum()),
                "Missing %": round(df[col].isna().mean() * 100, 2),
                "Mean": round(float(s.mean()), 4),
                "Std": round(float(s.std()), 4) if len(s) > 1 else 0.0,
                "Min": round(float(s.min()), 4),
                "25%": round(float(s.quantile(0.25)), 4),
                "Median (50%)": round(float(s.median()), 4),
                "75%": round(float(s.quantile(0.75)), 4),
                "Max": round(float(s.max()), 4),
                "Skewness": round(float(s.skew()), 4) if len(s) > 2 else 0.0
            })
        num_summary = pd.DataFrame(stats)

    # Categorical summary
    cat_summary = pd.DataFrame()
    if cat_cols:
        cat_stats = []
        for col in cat_cols:
            s = df[col].dropna().astype(str)
            if len(s) == 0:
                continue
            top_val = s.mode().iloc[0] if not s.empty else "N/A"
            top_freq = int((s == top_val).sum()) if not s.empty else 0
            cat_stats.append({
                "Column": col,
                "Count": int(s.count()),
                "Missing": int(df[col].isna().sum()),
                "Missing %": round(df[col].isna().mean() * 100, 2),
                "Unique": int(s.nunique()),
                "Top Value": top_val,
                "Top Frequency": top_freq,
                "Top %": round(top_freq / len(s) * 100, 2) if len(s) > 0 else 0.0
            })
        cat_summary = pd.DataFrame(cat_stats)

    return num_summary, cat_summary
