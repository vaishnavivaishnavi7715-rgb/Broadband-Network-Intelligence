from pathlib import Path
import json
import pandas as pd
import streamlit as st
PROJECT_ROOT = Path(__file__).resolve().parents[2]
METRICS_PATH = PROJECT_ROOT / "reports" / "candidate_holdout_metrics.json"
MATRIX_PATH = PROJECT_ROOT / "reports" / "candidate_holdout_confusion_matrix.csv"
st.divider()
st.header("Model Evaluation")
st.caption("Held-out evaluation of a Random Forest model on synthetic telemetry.")
if not METRICS_PATH.exists():
    st.warning("Evaluation metrics not found. Run src/evaluate_candidate_holdout.py first.")
else:
    try:
        with METRICS_PATH.open("r", encoding="utf-8") as file:
            metrics = json.load(file)
        if metrics.get("evaluation_type") != "Fresh model, stratified 80/20 holdout":
            st.warning("The evaluation type differs from the expected holdout evaluation.")
        metric_cols = st.columns(5)
        metric_cols[0].metric("Accuracy", f"{metrics['accuracy']:.2%}")
        metric_cols[1].metric("Precision", f"{metrics['precision']:.2%}")
        metric_cols[2].metric("Recall", f"{metrics['recall']:.2%}")
        metric_cols[3].metric("F1-score", f"{metrics['f1_score']:.2%}")
        metric_cols[4].metric("ROC-AUC", f"{metrics['roc_auc']:.2%}")
        detail_cols = st.columns(3)
        detail_cols[0].metric("Total records", f"{metrics['total_rows']:,}")
        detail_cols[1].metric("Training records", f"{metrics['training_rows']:,}")
        detail_cols[2].metric("Test records", f"{metrics['test_rows']:,}")
        st.subheader("Confusion Matrix")
        if MATRIX_PATH.exists():
            matrix_df = pd.read_csv(MATRIX_PATH, index_col=0)
            st.dataframe(matrix_df, use_container_width=True)
        else:
            st.warning("Confusion matrix CSV not found.")
        st.warning(metrics.get("warning", "Check dataset provenance before interpreting these results."))
    except (OSError, ValueError, KeyError, TypeError) as exc:
        st.error(f"Could not load evaluation results: {exc}")
