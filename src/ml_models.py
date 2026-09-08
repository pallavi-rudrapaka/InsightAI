"""
ml_models.py
Automated task detection, leakage-free pipelines, baseline vs strong model training & comparison.
"""

from typing import Dict, Any, List, Tuple, Optional
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    mean_absolute_error,
    mean_squared_error,
    r2_score
)
import plotly.express as px
import plotly.graph_objects as go


def detect_problem_type(df: pd.DataFrame, target_col: str) -> Dict[str, Any]:
    """
    Infers whether the target variable represents a Classification or Regression problem.
    """
    series = df[target_col].dropna()
    nunique = series.nunique()
    is_numeric = pd.api.types.is_numeric_dtype(series)

    if not is_numeric or nunique == 2 or (is_numeric and nunique <= 10 and series.dtype in ["int64", "int32"]):
        if nunique == 2:
            problem_type = "Binary Classification"
        elif nunique > 2 and nunique <= 20:
            problem_type = "Multiclass Classification"
        else:
            problem_type = "Classification"
        is_classification = True
    else:
        problem_type = "Regression"
        is_classification = False

    return {
        "problem_type": problem_type,
        "is_classification": is_classification,
        "unique_values_count": nunique,
        "sample_values": series.unique()[:5].tolist(),
        "is_numeric": is_numeric
    }


def build_preprocessor(
    X_train: pd.DataFrame,
    numerical_cols: List[str],
    categorical_cols: List[str]
) -> ColumnTransformer:
    """
    Creates a scikit-learn ColumnTransformer for numerical scaling and one-hot encoding.
    """
    transformers = []

    if numerical_cols:
        num_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ])
        transformers.append(("num", num_pipeline, numerical_cols))

    if categorical_cols:
        cat_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
        ])
        transformers.append(("cat", cat_pipeline, categorical_cols))

    return ColumnTransformer(transformers=transformers, remainder="drop")


def train_and_evaluate_models(
    df: pd.DataFrame,
    target_col: str,
    test_size: float = 0.2,
    random_state: int = 42
) -> Dict[str, Any]:
    """
    Executes an end-to-end ML pipeline with baseline vs strong tree model comparison.
    Guarantees no data leakage by fitting preprocessing strictly on training data.
    """
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in dataset.")

    # Drop rows where target is missing
    clean_df = df.dropna(subset=[target_col]).copy()
    if len(clean_df) < 10:
        raise ValueError("Dataset has too few records for machine learning training.")

    task_info = detect_problem_type(clean_df, target_col)
    is_classif = task_info["is_classification"]

    # Target variable preparation
    y_raw = clean_df[target_col]
    label_mapping = None

    if is_classif:
        le = LabelEncoder()
        y = le.fit_transform(y_raw)
        label_mapping = {int(i): str(cls) for i, cls in enumerate(le.classes_)}
    else:
        y = y_raw.values.astype(float)

    # Feature segregation: drop target and ID-like columns
    id_like = [c for c in clean_df.columns if c.lower() in ["id", "customerid", "customer_id", "id_code"]]
    drop_cols = [target_col] + id_like
    X = clean_df.drop(columns=[c for c in drop_cols if c in clean_df.columns])

    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = X.select_dtypes(exclude=[np.number]).columns.tolist()

    if not num_cols and not cat_cols:
        raise ValueError("No valid predictive features remaining in dataset.")

    # Train/Test Split
    stratify = y if (is_classif and pd.Series(y).value_counts().min() >= 2) else None
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=stratify
    )

    # Preprocessing
    preprocessor = build_preprocessor(X_train, num_cols, cat_cols)
    X_train_trans = preprocessor.fit_transform(X_train)
    X_test_trans = preprocessor.transform(X_test)

    # Extract feature names after transformation
    feature_names = []
    if hasattr(preprocessor, "get_feature_names_out"):
        try:
            raw_names = preprocessor.get_feature_names_out()
            feature_names = [n.replace("num__", "").replace("cat__", "") for n in raw_names]
        except Exception:
            feature_names = [f"feat_{i}" for i in range(X_train_trans.shape[1])]
    else:
        feature_names = [f"feat_{i}" for i in range(X_train_trans.shape[1])]

    results = {}

    if is_classif:
        # 1. Baseline Model (Dummy Classifier - Majority Class)
        baseline = DummyClassifier(strategy="most_frequent")
        baseline.fit(X_train_trans, y_train)
        y_pred_base = baseline.predict(X_test_trans)

        base_acc = float(accuracy_score(y_test, y_pred_base))
        base_prec = float(precision_score(y_test, y_pred_base, average="weighted", zero_division=0))
        base_rec = float(recall_score(y_test, y_pred_base, average="weighted", zero_division=0))
        base_f1 = float(f1_score(y_test, y_pred_base, average="weighted", zero_division=0))

        # 2. Strong Tree Model (Random Forest Classifier)
        strong_model = RandomForestClassifier(
            n_estimators=100,
            max_depth=8,
            random_state=random_state,
            n_jobs=-1
        )
        strong_model.fit(X_train_trans, y_train)
        y_pred_strong = strong_model.predict(X_test_trans)
        y_prob_strong = strong_model.predict_proba(X_test_trans)

        strong_acc = float(accuracy_score(y_test, y_pred_strong))
        strong_prec = float(precision_score(y_test, y_pred_strong, average="weighted", zero_division=0))
        strong_rec = float(recall_score(y_test, y_pred_strong, average="weighted", zero_division=0))
        strong_f1 = float(f1_score(y_test, y_pred_strong, average="weighted", zero_division=0))

        # ROC-AUC calculation
        roc_auc = None
        if len(np.unique(y_test)) == 2 and y_prob_strong.shape[1] == 2:
            try:
                roc_auc = float(roc_auc_score(y_test, y_prob_strong[:, 1]))
            except Exception:
                roc_auc = None
        elif len(np.unique(y_test)) > 2:
            try:
                roc_auc = float(roc_auc_score(y_test, y_prob_strong, multi_class="ovr", average="weighted"))
            except Exception:
                roc_auc = None

        cm = confusion_matrix(y_test, y_pred_strong)

        comparison_df = pd.DataFrame({
            "Metric": ["Accuracy", "Precision (Weighted)", "Recall (Weighted)", "F1 Score (Weighted)"],
            "Baseline Model (Dummy)": [round(base_acc, 4), round(base_prec, 4), round(base_rec, 4), round(base_f1, 4)],
            "Strong Model (Random Forest)": [round(strong_acc, 4), round(strong_prec, 4), round(strong_rec, 4), round(strong_f1, 4)],
            "Improvement (Delta)": [
                round(strong_acc - base_acc, 4),
                round(strong_prec - base_prec, 4),
                round(strong_rec - base_rec, 4),
                round(strong_f1 - base_f1, 4)
            ]
        })

        results = {
            "task_type": "Classification",
            "problem_type": task_info["problem_type"],
            "target_col": target_col,
            "train_samples": len(X_train),
            "test_samples": len(X_test),
            "comparison_df": comparison_df,
            "baseline_metrics": {"accuracy": base_acc, "precision": base_prec, "recall": base_rec, "f1": base_f1},
            "strong_metrics": {"accuracy": strong_acc, "precision": strong_prec, "recall": strong_rec, "f1": strong_f1, "roc_auc": roc_auc},
            "confusion_matrix": cm,
            "label_mapping": label_mapping,
            "strong_model": strong_model,
            "feature_names": feature_names,
            "X_test_trans": X_test_trans,
            "y_test": y_test,
            "y_pred": y_pred_strong
        }

    else:
        # Continuous Regression Task
        # 1. Baseline Model (Mean Predictor)
        baseline = DummyRegressor(strategy="mean")
        baseline.fit(X_train_trans, y_train)
        y_pred_base = baseline.predict(X_test_trans)

        base_mae = float(mean_absolute_error(y_test, y_pred_base))
        base_mse = float(mean_squared_error(y_test, y_pred_base))
        base_rmse = float(np.sqrt(base_mse))
        base_r2 = float(r2_score(y_test, y_pred_base))

        # 2. Strong Tree Model (Random Forest Regressor)
        strong_model = RandomForestRegressor(
            n_estimators=100,
            max_depth=8,
            random_state=random_state,
            n_jobs=-1
        )
        strong_model.fit(X_train_trans, y_train)
        y_pred_strong = strong_model.predict(X_test_trans)

        strong_mae = float(mean_absolute_error(y_test, y_pred_strong))
        strong_mse = float(mean_squared_error(y_test, y_pred_strong))
        strong_rmse = float(np.sqrt(strong_mse))
        strong_r2 = float(r2_score(y_test, y_pred_strong))

        comparison_df = pd.DataFrame({
            "Metric": ["Mean Absolute Error (MAE)", "Root Mean Squared Error (RMSE)", "R² Score (Variance Explained)"],
            "Baseline Model (Mean)": [round(base_mae, 4), round(base_rmse, 4), round(base_r2, 4)],
            "Strong Model (Random Forest)": [round(strong_mae, 4), round(strong_rmse, 4), round(strong_r2, 4)],
            "Improvement (Delta)": [
                round(base_mae - strong_mae, 4),  # Lower is better for error
                round(base_rmse - strong_rmse, 4),
                round(strong_r2 - base_r2, 4)      # Higher is better for R2
            ]
        })

        results = {
            "task_type": "Regression",
            "problem_type": task_info["problem_type"],
            "target_col": target_col,
            "train_samples": len(X_train),
            "test_samples": len(X_test),
            "comparison_df": comparison_df,
            "baseline_metrics": {"mae": base_mae, "rmse": base_rmse, "r2": base_r2},
            "strong_metrics": {"mae": strong_mae, "rmse": strong_rmse, "r2": strong_r2},
            "strong_model": strong_model,
            "feature_names": feature_names,
            "X_test_trans": X_test_trans,
            "y_test": y_test,
            "y_pred": y_pred_strong
        }

    return results


def plot_confusion_matrix_heatmap(cm: np.ndarray, labels: Optional[List[str]] = None) -> go.Figure:
    """
    Renders an interactive heatmap of the classification confusion matrix.
    """
    n_classes = cm.shape[0]
    if labels is None or len(labels) != n_classes:
        labels = [f"Class {i}" for i in range(n_classes)]

    fig = px.imshow(
        cm,
        text_auto=True,
        labels=dict(x="Predicted Label", y="True Label", color="Count"),
        x=labels,
        y=labels,
        color_continuous_scale="Blues",
        title="<b>Confusion Matrix</b> (Strong Model)"
    )

    fig.update_layout(
        template="plotly_white",
        margin=dict(l=50, r=50, t=60, b=50)
    )
    return fig


def plot_regression_actual_vs_pred(y_test: np.ndarray, y_pred: np.ndarray) -> go.Figure:
    """
    Renders Actual vs. Predicted scatter plot with perfect prediction identity line.
    """
    df_plot = pd.DataFrame({"Actual": y_test, "Predicted": y_pred})
    min_val = min(y_test.min(), y_pred.min())
    max_val = max(y_test.max(), y_pred.max())

    fig = px.scatter(
        df_plot,
        x="Actual",
        y="Predicted",
        title="<b>Actual vs. Predicted Values</b> (Strong Model)",
        opacity=0.7,
        template="plotly_white"
    )

    fig.add_trace(
        go.Scatter(
            x=[min_val, max_val],
            y=[min_val, max_val],
            mode="lines",
            name="Ideal Fit (y = x)",
            line=dict(color="#EF4444", dash="dash", width=2)
        )
    )

    fig.update_layout(
        margin=dict(l=40, r=40, t=60, b=40),
        xaxis_title="True Actual Value",
        yaxis_title="Model Predicted Value"
    )
    return fig


def plot_regression_residuals(y_test: np.ndarray, y_pred: np.ndarray) -> go.Figure:
    """
    Renders residual errors vs predicted values.
    """
    residuals = y_test - y_pred
    df_plot = pd.DataFrame({"Predicted": y_pred, "Residual": residuals})

    fig = px.scatter(
        df_plot,
        x="Predicted",
        y="Residual",
        title="<b>Residual Diagnostics Plot</b> (True - Predicted)",
        opacity=0.7,
        template="plotly_white"
    )

    fig.add_hline(y=0, line_dash="dash", line_color="#EF4444")
    fig.update_layout(
        margin=dict(l=40, r=40, t=60, b=40),
        xaxis_title="Predicted Value",
        yaxis_title="Residual Error"
    )
    return fig
