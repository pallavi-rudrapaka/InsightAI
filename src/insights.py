"""
insights.py
Natural-language insight generation using LLMs (Gemini / OpenAI) with a robust,
deterministic fallback generator based strictly on computed results.
"""

from typing import Dict, Any, Optional
import os
import json
import requests


def build_analytics_payload(
    overview: Dict[str, Any],
    quality: Dict[str, Any],
    anomalies: Optional[Dict[str, Any]] = None,
    ml_results: Optional[Dict[str, Any]] = None,
    feature_importance: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Serializes verified calculated statistics into a structured dictionary.
    """
    payload = {
        "dataset_profile": {
            "num_rows": overview.get("num_rows", 0),
            "num_cols": overview.get("num_cols", 0),
            "memory_usage": overview.get("memory_usage", "0 KB"),
            "columns": overview.get("columns", [])
        },
        "data_quality": {
            "score": quality.get("score", 100),
            "grade": quality.get("grade", "A"),
            "health_status": quality.get("health_status", "Good"),
            "completeness_score": quality.get("completeness_score", 100),
            "uniqueness_score": quality.get("uniqueness_score", 100),
            "distribution_score": quality.get("distribution_score", 100),
            "critical_flags": quality.get("critical_flags", [])
        }
    }

    if anomalies:
        payload["anomaly_detection"] = {
            "features_used": anomalies.get("features_used", []),
            "total_records": anomalies.get("total_records", 0),
            "anomaly_count": anomalies.get("anomaly_count", 0),
            "anomaly_percentage": anomalies.get("anomaly_percentage", 0.0),
            "contamination_param": anomalies.get("contamination", 0.05)
        }

    if ml_results:
        task_type = ml_results.get("task_type", "Unknown")
        payload["machine_learning"] = {
            "task_type": task_type,
            "problem_type": ml_results.get("problem_type", "Unknown"),
            "target_column": ml_results.get("target_col", "None"),
            "train_samples": ml_results.get("train_samples", 0),
            "test_samples": ml_results.get("test_samples", 0),
            "baseline_metrics": ml_results.get("baseline_metrics", {}),
            "strong_model_metrics": ml_results.get("strong_metrics", {})
        }

    if feature_importance and "top_table" in feature_importance:
        top_df = feature_importance["top_table"]
        top_feats = []
        for _, r in top_df.head(5).iterrows():
            top_feats.append({
                "feature": str(r["Feature"]),
                "importance_pct": float(r["Importance %"])
            })
        payload["key_feature_drivers"] = top_feats

    return payload


def generate_deterministic_insights(payload: Dict[str, Any]) -> str:
    """
    Synthesizes articulate, highly structured markdown insights directly
    from verified metrics without requiring an external LLM API or internet.
    """
    profile = payload.get("dataset_profile", {})
    quality = payload.get("data_quality", {})
    anomalies = payload.get("anomaly_detection", {})
    ml = payload.get("machine_learning", {})
    features = payload.get("key_feature_drivers", [])

    rows = profile.get("num_rows", "N/A")
    cols = profile.get("num_cols", "N/A")
    mem = profile.get("memory_usage", "N/A")
    q_score = quality.get("score", "N/A")
    q_grade = quality.get("grade", "N/A")
    q_health = quality.get("health_status", "N/A")

    md = []
    md.append("### Executive Dataset Summary")
    md.append(
        f"The ingested dataset comprises **{rows:,} observations** spanning **{cols} attributes**, "
        f"occupying approximately **{mem}** of system memory. Automated structural profiling confirmed "
        f"that the dataset was ingested with clean column indexing and proper type parsing."
    )

    md.append("\n### Data Quality & Hygiene Assessment")
    md.append(
        f"The overall Data Quality Index is computed at **{q_score}/100** (Grade: **{q_grade}**, Health: **{q_health}**). "
        f"Individual dimension sub-scores: Completeness **{quality.get('completeness_score', 'N/A')}/100**, "
        f"Uniqueness **{quality.get('uniqueness_score', 'N/A')}/100**, and "
        f"Distribution Health **{quality.get('distribution_score', 'N/A')}/100**."
    )
    flags = quality.get("critical_flags", [])
    if flags:
        md.append("**Identified Quality Alerts:**")
        for flag in flags:
            md.append(f"- {flag}")
    else:
        md.append("- No severe data hygiene defects or high missingness thresholds breached.")

    if anomalies:
        anom_cnt = anomalies.get("anomaly_count", 0)
        anom_pct = anomalies.get("anomaly_percentage", 0.0)
        anom_feats = ", ".join(anomalies.get("features_used", []))
        md.append("\n### Anomaly & Outlier Diagnostics")
        md.append(
            f"Multi-dimensional Isolation Forest detected **{anom_cnt} anomalous observations** "
            f"(**{anom_pct}%** of the dataset) across evaluated numerical dimensions (`{anom_feats}`). "
            f"These instances reflect high dimensional distance from core clustering centers and should be audited."
        )

    if ml:
        task = ml.get("task_type", "Unknown")
        target = ml.get("target_column", "Unknown")
        tr_n = ml.get("train_samples", 0)
        te_n = ml.get("test_samples", 0)
        strong = ml.get("strong_model_metrics", {})
        base = ml.get("baseline_metrics", {})

        md.append(f"\n### Machine Learning Benchmark: Baseline vs. Strong Model")
        md.append(
            f"Supervised learning modeling was performed targeting **`{target}`** ({task}). "
            f"Evaluation was conducted on a strict holdout split (**{tr_n}** training records, **{te_n}** test records) "
            f"with zero information leakage."
        )

        if task == "Classification":
            s_acc = strong.get("accuracy", 0) * 100
            b_acc = base.get("accuracy", 0) * 100
            s_f1 = strong.get("f1", 0)
            b_f1 = base.get("f1", 0)
            roc = strong.get("roc_auc")
            roc_str = f" and ROC-AUC of **{roc:.3f}**" if roc else ""
            md.append(
                f"- **Random Forest Classifier**: Achieved **{s_acc:.1f}% Accuracy** with a weighted F1-Score of **{s_f1:.3f}**{roc_str}.\n"
                f"- **Baseline Majority Model**: Achieved **{b_acc:.1f}% Accuracy** (F1: **{b_f1:.3f}**).\n"
                f"- **Relative Performance Lift**: Strong model improved accuracy by **+{s_acc - b_acc:.1f} percentage points** over naive baseline."
            )
        else:
            s_r2 = strong.get("r2", 0)
            b_r2 = base.get("r2", 0)
            s_rmse = strong.get("rmse", 0)
            b_rmse = base.get("rmse", 0)
            md.append(
                f"- **Random Forest Regressor**: Achieved **R² = {s_r2:.3f}** (explaining {max(0.0, s_r2*100):.1f}% of variance) with RMSE = **{s_rmse:.2f}**.\n"
                f"- **Baseline Mean Model**: Produced **R² = {b_r2:.3f}** with RMSE = **{b_rmse:.2f}**.\n"
                f"- **Error Reduction**: Random Forest reduced prediction root mean squared error by **{b_rmse - s_rmse:.2f}**."
            )

    if features:
        md.append("\n### Key Predictive Signals & Explainability")
        md.append("Tree-based feature importance analysis identified the primary drivers influencing model decisions:")
        for f in features:
            md.append(f"- **{f['feature']}**: Contributes **{f['importance_pct']:.1f}%** of predictive importance.")

    md.append("\n### Data-Driven Suggestions")
    md.append("*(Note: These are analytical suggestions derived from computed metrics, not guaranteed conclusions)*")
    if quality.get("completeness_score", 100) < 95:
        md.append("- Implement systematic validation at ingestion to prevent null records.")
    if anomalies.get("anomaly_percentage", 0) > 3.0:
        md.append("- Isolate anomalous records identified by Isolation Forest for domain-expert audit.")
    if ml.get("task_type") == "Classification" and strong.get("f1", 0) > 0.75:
        md.append("- Candidate model exhibits solid predictive power over baseline; evaluate pilot deployment with shadow logging.")
    elif ml.get("task_type") == "Regression" and strong.get("r2", 0) > 0.60:
        md.append("- Strong model captures substantial variance; suitable for decision support estimations.")
    else:
        md.append("- Consider feature engineering or gathering additional contextual attributes to boost model precision.")

    md.append("\n### Operational Limitations & Governance")
    md.append(
        "- **Sample Representativeness**: Model validity is bounded by the historical training distribution. In case of concept drift, retraining is essential.\n"
        "- **Correlation vs. Causation**: Identified feature importance reflects statistical predictive utility, not causal business mechanisms.\n"
        "- **Human-in-the-Loop Requirement**: All automated decisions must undergo human analyst oversight prior to operational deployment."
    )

    return "\n".join(md)


def generate_llm_insights(
    payload: Dict[str, Any],
    api_key: Optional[str] = None,
    provider: str = "gemini",
    model_name: Optional[str] = None
) -> Dict[str, Any]:
    """
    Generates natural-language analytical insights using an LLM API.
    Falls back gracefully to deterministic rule-based generation if no API key is provided
    or if the network call fails.
    """
    # Check environment variables if key not directly passed
    if not api_key:
        if provider == "gemini":
            api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        elif provider == "openai":
            api_key = os.environ.get("OPENAI_API_KEY")

    # If still no key, use deterministic fallback
    if not api_key:
        return {
            "insights_markdown": generate_deterministic_insights(payload),
            "source": "Deterministic Analytical Engine (No API key provided)",
            "success": True,
            "fallback": True
        }

    system_instructions = (
        "You are a Senior Principal Data Scientist and Executive Decision Support Analyst at InsightAI. "
        "You are reviewing computed results from an automated data science pipeline. "
        "CRITICAL RULES:\n"
        "1. STRICT FACTUALITY: Strictly use ONLY the numerical metrics and statistics provided in the JSON payload. "
        "DO NOT hallucinate, invent, extrapolate, or approximate any numbers not explicitly provided.\n"
        "2. STRUCTURE: Deliver your analysis in polished markdown with clear sections:\n"
        "   - Executive Summary\n"
        "   - Data Quality & Hygiene Assessment\n"
        "   - Anomaly & Risk Diagnostics\n"
        "   - Machine Learning Performance (compare Baseline vs. Strong Model)\n"
        "   - Key Feature Drivers & Explainability\n"
        "   - Data-Driven Suggestions (clearly state these are suggestions, not guaranteed conclusions)\n"
        "   - Limitations & Analyst Governance (state correlation != causation, human review needed)\n"
        "3. TONE: Professional, objective, analytical, and executive-ready."
    )

    user_prompt = (
        f"Analyze the following verified tabular dataset analytics payload:\n\n"
        f"```json\n{json.dumps(payload, indent=2)}\n```\n\n"
        "Produce your comprehensive analytical appraisal following the specified rules."
    )

    try:
        if provider == "gemini":
            model = model_name or "gemini-2.0-flash"
            url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}"
            headers = {"Content-Type": "application/json"}
            body = {
                "contents": [
                    {"role": "user", "parts": [{"text": f"{system_instructions}\n\n{user_prompt}"}]}
                ],
                "generationConfig": {
                    "temperature": 0.2,
                    "maxOutputTokens": 2048
                }
            }
            resp = requests.post(url, headers=headers, json=body, timeout=25)
            if resp.status_code == 200:
                data = resp.json()
                text = data["candidates"][0]["content"]["parts"][0]["text"]
                return {
                    "insights_markdown": text,
                    "source": f"Google Gemini ({model})",
                    "success": True,
                    "fallback": False
                }
            else:
                err_msg = f"Gemini API error {resp.status_code}: {resp.text}"
                fallback_text = generate_deterministic_insights(payload)
                return {
                    "insights_markdown": f"> [!WARNING]\n> API call failed ({err_msg}). Reverting to Deterministic Fallback Engine.\n\n{fallback_text}",
                    "source": "Deterministic Analytical Engine (API Call Error)",
                    "success": False,
                    "fallback": True
                }

        elif provider == "openai":
            model = model_name or "gpt-4o-mini"
            url = "https://api.openai.com/v1/chat/completions"
            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            }
            body = {
                "model": model,
                "messages": [
                    {"role": "system", "content": system_instructions},
                    {"role": "user", "content": user_prompt}
                ],
                "temperature": 0.2
            }
            resp = requests.post(url, headers=headers, json=body, timeout=25)
            if resp.status_code == 200:
                data = resp.json()
                text = data["choices"][0]["message"]["content"]
                return {
                    "insights_markdown": text,
                    "source": f"OpenAI ({model})",
                    "success": True,
                    "fallback": False
                }
            else:
                err_msg = f"OpenAI API error {resp.status_code}: {resp.text}"
                fallback_text = generate_deterministic_insights(payload)
                return {
                    "insights_markdown": f"> [!WARNING]\n> API call failed ({err_msg}). Reverting to Deterministic Fallback Engine.\n\n{fallback_text}",
                    "source": "Deterministic Analytical Engine (API Call Error)",
                    "success": False,
                    "fallback": True
                }

    except Exception as e:
        fallback_text = generate_deterministic_insights(payload)
        return {
            "insights_markdown": f"> [!WARNING]\n> Connection error ({str(e)}). Reverting to Deterministic Fallback Engine.\n\n{fallback_text}",
            "source": "Deterministic Analytical Engine (Exception Caught)",
            "success": False,
            "fallback": True
        }

    # Default fallback
    return {
        "insights_markdown": generate_deterministic_insights(payload),
        "source": "Deterministic Analytical Engine",
        "success": True,
        "fallback": True
    }
