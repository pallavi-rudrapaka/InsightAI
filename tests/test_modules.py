"""
test_modules.py
Comprehensive automated test suite for all InsightAI core modules and edge cases.
"""

import os
import sys
import unittest
import numpy as np
import pandas as pd

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data_loader import load_data, get_dataset_overview, identify_column_types, get_summary_statistics
from src.data_quality import analyze_missing_values, analyze_duplicates, analyze_outliers, compute_data_quality_score
from src.eda import (
    plot_numerical_distribution,
    plot_categorical_distribution,
    plot_correlation_heatmap,
    plot_bivariate_scatter,
    plot_bivariate_box
)
from src.anomaly_detection import detect_anomalies_isolation_forest, plot_anomaly_distribution
from src.ml_models import detect_problem_type, train_and_evaluate_models
from src.explainability import extract_feature_importance, plot_feature_importance_bar
from src.insights import build_analytics_payload, generate_deterministic_insights, generate_llm_insights


class TestInsightAICore(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.sample_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "sample_data.csv"))
        cls.df = load_data(cls.sample_path)

    def test_01_data_loader(self):
        self.assertFalse(self.df.empty)
        self.assertEqual(len(self.df), 1000)

        overview = get_dataset_overview(self.df)
        self.assertEqual(overview["num_rows"], 1000)
        self.assertGreater(overview["num_cols"], 10)

        col_types = identify_column_types(self.df)
        self.assertIn("Age", col_types["numerical"])
        self.assertIn("ContractType", col_types["categorical"])
        self.assertIn("CustomerID", col_types["id_cols"])

        num_stats, cat_stats = get_summary_statistics(self.df)
        self.assertFalse(num_stats.empty)
        self.assertFalse(cat_stats.empty)

    def test_02_data_quality(self):
        missing = analyze_missing_values(self.df)
        self.assertIn("TotalCharges", missing["columns_with_missing"]["Column"].values)

        dups = analyze_duplicates(self.df)
        self.assertFalse(dups["has_duplicates"])

        outliers = analyze_outliers(self.df)
        self.assertGreater(outliers["total_outlier_rows"], 0)

        q_score = compute_data_quality_score(self.df)
        self.assertGreaterEqual(q_score["score"], 0)
        self.assertLessEqual(q_score["score"], 100)
        self.assertIn(q_score["grade"], ["A", "B", "C", "D"])

    def test_03_eda_visualizations(self):
        fig_num = plot_numerical_distribution(self.df, "Age")
        self.assertIsNotNone(fig_num)

        fig_cat = plot_categorical_distribution(self.df, "ContractType")
        self.assertIsNotNone(fig_cat)

        fig_corr = plot_correlation_heatmap(self.df)
        self.assertIsNotNone(fig_corr)

        fig_scatter = plot_bivariate_scatter(self.df, "TenureMonths", "TotalCharges")
        self.assertIsNotNone(fig_scatter)

        fig_box = plot_bivariate_box(self.df, "ContractType", "MonthlyCharges")
        self.assertIsNotNone(fig_box)

    def test_04_anomaly_detection(self):
        result = detect_anomalies_isolation_forest(self.df, contamination=0.05)
        self.assertEqual(result["total_records"], 1000)
        self.assertEqual(result["anomaly_count"], 50)
        self.assertEqual(result["anomaly_percentage"], 5.0)
        self.assertIn("anomaly_label", result["annotated_df"].columns)
        self.assertIn("anomaly_score", result["annotated_df"].columns)

        fig_anom = plot_anomaly_distribution(result["annotated_df"], "TenureMonths", "MonthlyCharges")
        self.assertIsNotNone(fig_anom)

    def test_05_machine_learning_classification(self):
        task_info = detect_problem_type(self.df, "Churn")
        self.assertTrue(task_info["is_classification"])

        ml_res = train_and_evaluate_models(self.df, target_col="Churn", test_size=0.2, random_state=42)
        self.assertEqual(ml_res["task_type"], "Classification")
        self.assertIn("comparison_df", ml_res)
        self.assertGreater(ml_res["strong_metrics"]["accuracy"], 0.60)
        self.assertIsNotNone(ml_res["strong_model"])

        # Test explainability
        exp = extract_feature_importance(ml_res["strong_model"], ml_res["feature_names"], top_n=10)
        self.assertFalse(exp["top_table"].empty)
        fig_imp = plot_feature_importance_bar(exp["top_table"])
        self.assertIsNotNone(fig_imp)

    def test_06_machine_learning_regression(self):
        task_info = detect_problem_type(self.df, "MonthlyCharges")
        self.assertFalse(task_info["is_classification"])

        ml_res = train_and_evaluate_models(self.df, target_col="MonthlyCharges", test_size=0.2, random_state=42)
        self.assertEqual(ml_res["task_type"], "Regression")
        self.assertIn("comparison_df", ml_res)
        self.assertGreater(ml_res["strong_metrics"]["r2"], 0.50)

    def test_07_insights_generation(self):
        overview = get_dataset_overview(self.df)
        quality = compute_data_quality_score(self.df)
        anomalies = detect_anomalies_isolation_forest(self.df, contamination=0.05)
        ml_res = train_and_evaluate_models(self.df, target_col="Churn")
        exp = extract_feature_importance(ml_res["strong_model"], ml_res["feature_names"])

        payload = build_analytics_payload(overview, quality, anomalies, ml_res, exp)
        self.assertIn("dataset_profile", payload)
        self.assertIn("machine_learning", payload)

        deterministic_text = generate_deterministic_insights(payload)
        self.assertIn("Executive Dataset Summary", deterministic_text)
        self.assertIn("Data Quality & Hygiene Assessment", deterministic_text)
        self.assertIn("Random Forest Classifier", deterministic_text)

        llm_res = generate_llm_insights(payload, api_key=None)
        self.assertTrue(llm_res["fallback"])
        self.assertIn("Executive Dataset Summary", llm_res["insights_markdown"])

    def test_08_edge_cases(self):
        # Empty CSV
        with self.assertRaises(ValueError):
            load_data(pd.io.common.StringIO(""))

        # Single numeric column
        single_num_df = pd.DataFrame({"Metric": [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]})
        overview = get_dataset_overview(single_num_df)
        self.assertEqual(overview["num_cols"], 1)
        heatmap = plot_correlation_heatmap(single_num_df)
        self.assertIsNone(heatmap)

        # Pure categorical
        cat_df = pd.DataFrame({"Category": ["A", "B", "A", "C", "B", "A", "C", "A", "B", "C"]})
        col_types = identify_column_types(cat_df)
        self.assertEqual(len(col_types["numerical"]), 0)
        self.assertEqual(len(col_types["categorical"]), 1)


if __name__ == "__main__":
    unittest.main()
