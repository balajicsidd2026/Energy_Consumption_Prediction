"""
STAGE 3 — DATA PREPARATION FOR AUTOGLUON
─────────────────────────────────────────
AutoGluon handles ALL feature engineering internally:
  - categorical encoding
  - missing value imputation
  - feature transformations
  - skewness handling
  - automatic feature engineering
  - model-specific preprocessing

This stage only does the minimal necessary work:
  1. Drop explicitly configured columns (IDs, leakage cols)
  2. Auto-detect and drop ID-like columns (unique per row, object type)
  3. Drop constant columns (zero variance)
  4. Save cleaned train/val/test CSVs — AutoGluon reads them directly
  5. Log data profile to MLflow

NO manual encoding, scaling, imputation, or transformation is done here.
AutoGluon owns that responsibility completely.
"""

import os
import sys
import json
import warnings
import pandas as pd
import numpy as np

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(__file__))
from utils import (load_config, base_dir, init_mlflow,
                   safe_log_artifact, safe_log_metrics, safe_log_params, get_logger)
import mlflow

logger = get_logger("preprocessing")


def detect_problem_type(series: pd.Series, threshold: int = 15) -> str:
    if pd.api.types.is_numeric_dtype(series) and series.nunique() > threshold:
        return "regression"
    return "classification"


def run_preprocessing():
    logger.info("=" * 60)
    logger.info("STAGE 3: DATA PREPARATION (AutoGluon-Ready)")
    logger.info("NOTE: All feature engineering delegated to AutoGluon")
    logger.info("=" * 60)

    cfg      = load_config()
    bd       = base_dir()
    proc_dir = os.path.join(bd, cfg["paths"]["processed_dir"])
    arts_dir = os.path.join(bd, cfg["paths"]["artifacts_dir"], "preprocessing")
    os.makedirs(arts_dir, exist_ok=True)

    train_path = os.path.join(proc_dir, "train.csv")
    val_path   = os.path.join(proc_dir, "val.csv")
    test_path  = os.path.join(proc_dir, "test.csv")
    if not os.path.exists(train_path):
        raise FileNotFoundError("train.csv not found — run Stage 1 (data_ingestion) first.")

    train_df = pd.read_csv(train_path, low_memory=False)
    val_df   = pd.read_csv(val_path,   low_memory=False) if os.path.exists(val_path) else pd.DataFrame()
    test_df  = pd.read_csv(test_path,  low_memory=False)

    target_col   = cfg.get("data_schema", {}).get("target_column") or train_df.columns[-1]
    problem_type = detect_problem_type(train_df[target_col])
    logger.info(f"Target: '{target_col}'  |  Mode: {problem_type.upper()}")

    # ── Step 1: Explicit drops from config ───────────────────────────────
    # These either leak the target (Actual_Operation_Duration_Minutes,
    # Delay_Minutes, Scheduled_SLA_Minutes, SLA_Performance) or are row
    # identifiers (AWB_Number).
    explicit_drops = cfg.get("data_schema", {}).get("drop_columns", []) or []

    # ── Step 2: Auto-detect ID columns (unique per row, string type) ─────
    id_cols = [
        c for c in train_df.columns
        if c != target_col
        and train_df[c].dtype == object
        and train_df[c].nunique() == len(train_df)
    ]

    # Also catch numeric index-like columns (monotonically increasing integers)
    numeric_id_cols = [
        c for c in train_df.select_dtypes(include=["int64"]).columns
        if c != target_col
        and train_df[c].nunique() == len(train_df)
        and (train_df[c].diff().dropna() == 1).all()  # monotonically +1
    ]

    # ── Step 3: Constant columns ─────────────────────────────────────────
    const_cols = [
        c for c in train_df.columns
        if c != target_col and train_df[c].nunique() <= 1
    ]

    all_drops = list(set(explicit_drops + id_cols + numeric_id_cols + const_cols))

    if all_drops:
        logger.info(f"  Dropping columns: {all_drops}")
        logger.info(f"    → explicit config : {explicit_drops}")
        logger.info(f"    → auto ID cols    : {id_cols + numeric_id_cols}")
        logger.info(f"    → constant cols   : {const_cols}")
        train_df = train_df.drop(columns=[c for c in all_drops if c in train_df.columns])
        if not val_df.empty:
            val_df = val_df.drop(columns=[c for c in all_drops if c in val_df.columns])
        test_df  = test_df.drop(columns=[c for c in all_drops if c in test_df.columns])
    else:
        logger.info("  No columns dropped — passing full dataset to AutoGluon")

    # ── Step 4: Data profile (for reporting only) ─────────────────────────
    numeric_cols = train_df.select_dtypes(include=["int64", "float64"]).columns.tolist()
    cat_cols     = train_df.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
    if target_col in cat_cols:
        cat_cols.remove(target_col)
    if target_col in numeric_cols:
        numeric_cols.remove(target_col)

    missing_total = int(train_df.isnull().sum().sum())
    skew_vals     = train_df[numeric_cols].skew().abs()
    high_skew     = skew_vals[skew_vals > 1.0].index.tolist()

    logger.info(f"  Train shape          : {train_df.shape}")
    if not val_df.empty:
        logger.info(f"  Val shape            : {val_df.shape}")
    logger.info(f"  Test shape           : {test_df.shape}")
    logger.info(f"  Numeric features     : {len(numeric_cols)}")
    logger.info(f"  Categorical features : {len(cat_cols)}")
    logger.info(f"  Missing values       : {missing_total} (AutoGluon will impute)")
    logger.info(f"  High-skew features   : {len(high_skew)} (AutoGluon will handle)")

    if problem_type == "classification":
        class_counts = train_df[target_col].value_counts().to_dict()
        logger.info(f"  Class distribution   : {class_counts}")

    # ── Step 5: Save AutoGluon-ready CSVs ────────────────────────────────
    # These are written with ALL original columns intact (minus drops).
    # AutoGluon reads them directly — no further transformation needed.
    ag_train_path = os.path.join(proc_dir, "train_autogluon.csv")
    ag_val_path   = os.path.join(proc_dir, "val_autogluon.csv")
    ag_test_path  = os.path.join(proc_dir, "test_autogluon.csv")
    train_df.to_csv(ag_train_path, index=False)
    if not val_df.empty:
        val_df.to_csv(ag_val_path, index=False)
    test_df.to_csv(ag_test_path,   index=False)
    logger.info(f"  Saved: train_autogluon.csv")
    if not val_df.empty:
        logger.info(f"  Saved: val_autogluon.csv")
    logger.info(f"  Saved: test_autogluon.csv")

    # ── Report artifact ───────────────────────────────────────────────────
    report = {
        "stage":           "data_preparation",
        "target_column":   target_col,
        "problem_type":    problem_type,
        "preprocessing_engine": "AutoGluon (internal)",
        "dropped_cols":    all_drops,
        "train_shape":     list(train_df.shape),
        "val_shape":       list(val_df.shape) if not val_df.empty else None,
        "test_shape":      list(test_df.shape),
        "n_numeric_features":     len(numeric_cols),
        "n_categorical_features": len(cat_cols),
        "total_missing":   missing_total,
        "high_skew_features": high_skew,
        "note": "No manual encoding/scaling/imputation performed. AutoGluon handles all preprocessing internally per model."
    }
    rpt_path = os.path.join(arts_dir, "data_preparation_report.json")
    with open(rpt_path, "w") as f:
        json.dump(report, f, indent=2)

    # ── MLflow ────────────────────────────────────────────────────────────
    init_mlflow(cfg)
    with mlflow.start_run(run_name="Stage_3_Data_Preparation"):
        safe_log_params({
            "target_column":         target_col,
            "problem_type":          problem_type,
            "preprocessing_engine":  "AutoGluon",
            "dropped_cols":          str(all_drops),
            "n_numeric_features":    len(numeric_cols),
            "n_categorical_features": len(cat_cols),
        })
        safe_log_metrics({
            "train_rows":    train_df.shape[0],
            "train_cols":    train_df.shape[1],
            "val_rows":      val_df.shape[0] if not val_df.empty else 0,
            "test_rows":     test_df.shape[0],
            "total_missing": missing_total,
            "high_skew_cols": len(high_skew),
        })
        safe_log_artifact(rpt_path, "preprocessing")

    logger.info("Stage 3 — Data Preparation COMPLETE ✓")
    return report


if __name__ == "__main__":
    run_preprocessing()
