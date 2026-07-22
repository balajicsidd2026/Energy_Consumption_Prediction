"""
STAGE 1 — DATA INGESTION
────────────────────────
• Loads raw CSV (path from config.yaml)
• Auto-detects delimiter, encoding
• Validates schema: missing target, duplicate rows, shape sanity checks
• Derives calendar features from the operation date (Year, Month, Quarter,
  Day_of_Week, Weekend) so downstream stages can use them
• Chronological train/validation/test split (sorted by date — NOT random)
• Logs all metrics + artifact to MLflow
"""

import os
import sys
import json
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from utils import load_config, base_dir, init_mlflow, safe_log_artifact, safe_log_metrics, safe_log_params, get_logger
import mlflow

logger = get_logger("data_ingestion")


def detect_problem_type(series: pd.Series, threshold: int = 15) -> str:
    if pd.api.types.is_numeric_dtype(series.dtype) and series.nunique() > threshold:
        return "regression"
    return "classification"


def load_raw_data(path: str) -> pd.DataFrame:
    """Try common encodings / delimiters automatically."""
    for enc in ["utf-8", "latin-1", "cp1252"]:
        for sep in [",", ";", "\t", "|"]:
            try:
                df = pd.read_csv(path, encoding=enc, sep=sep, low_memory=False)
                if df.shape[1] > 1:
                    logger.info(f"Loaded '{path}' | enc={enc} sep='{sep}' | shape={df.shape}")
                    return df
            except Exception:
                pass
    raise ValueError(f"Cannot parse {path} — check encoding/delimiter.")


def add_calendar_features(df: pd.DataFrame, date_col: str, derived_cols=None) -> pd.DataFrame:
    """Derive Year, Month, Quarter, Day_of_Week, Weekend from the date column,
    but only the ones listed in config['feature_engineering']['derived_columns']
    and only if they are not already present in the raw data."""
    derived_cols = derived_cols or []
    if date_col not in df.columns:
        logger.warning(f"  Date column '{date_col}' not found — skipping calendar feature engineering.")
        return df

    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    n_bad = int(df[date_col].isna().sum())
    if n_bad:
        logger.warning(f"  Dropping {n_bad} rows with unparseable '{date_col}' values.")
        df = df.dropna(subset=[date_col]).reset_index(drop=True)

    to_add = [c for c in derived_cols if c not in df.columns]
    if not to_add:
        logger.info(f"  Calendar features already present in raw data — no derivation needed.")
        return df

    if "Year" in to_add:
        df["Year"] = df[date_col].dt.year
    if "Month" in to_add:
        df["Month"] = df[date_col].dt.month
    if "Quarter" in to_add:
        df["Quarter"] = df[date_col].dt.quarter
    if "Day_of_Week" in to_add:
        df["Day_of_Week"] = df[date_col].dt.dayofweek
    if "Weekend" in to_add:
        df["Weekend"] = (df[date_col].dt.dayofweek >= 5).astype(int)

    logger.info(f"  Derived calendar features: {to_add}")
    return df


def initiate_data_ingestion():
    logger.info("=" * 60)
    logger.info("STAGE 1: DATA INGESTION")
    logger.info("=" * 60)

    cfg = load_config()
    bd  = base_dir()

    raw_file    = os.path.join(bd, cfg["paths"]["raw_data_file"])
    raw_dir     = os.path.join(bd, cfg["paths"]["raw_dir"])
    proc_dir    = os.path.join(bd, cfg["paths"]["processed_dir"])
    arts_dir    = os.path.join(bd, cfg["paths"]["artifacts_dir"], "ingestion")
    os.makedirs(raw_dir, exist_ok=True)
    os.makedirs(proc_dir, exist_ok=True)
    os.makedirs(arts_dir, exist_ok=True)

    # ── Load ──────────────────────────────────────────────────────────────
    if not os.path.exists(raw_file):
        raise FileNotFoundError(f"Raw data not found: {raw_file}")

    df = load_raw_data(raw_file)

    # ── Target column ─────────────────────────────────────────────────────
    target_col = cfg.get("data_schema", {}).get("target_column") or df.columns[-1]
    if target_col not in df.columns:
        raise KeyError(f"Target column '{target_col}' not found. Columns: {df.columns.tolist()}")

    # Drop rows with missing target values
    if df[target_col].isna().any():
        logger.warning(f"Dropping {df[target_col].isna().sum()} rows with missing target values.")
        df = df.dropna(subset=[target_col]).reset_index(drop=True)

    problem_type = detect_problem_type(df[target_col])
    logger.info(f"Target: '{target_col}'  |  Problem type: {problem_type.upper()}")

    # ── Data quality checks ───────────────────────────────────────────────
    total_rows       = len(df)
    duplicate_rows   = int(df.duplicated().sum())
    missing_total    = int(df.isnull().sum().sum())
    missing_pct      = round(100 * missing_total / df.size, 2)
    constant_cols    = [c for c in df.columns if df[c].nunique() <= 1]

    if duplicate_rows:
        logger.warning(f"Dropping {duplicate_rows} duplicate rows.")
        df = df.drop_duplicates()

    # Drop fully empty columns
    empty_cols = df.columns[df.isnull().all()].tolist()
    if empty_cols:
        logger.warning(f"Dropping fully-null columns: {empty_cols}")
        df = df.drop(columns=empty_cols)

    # ── Calendar feature engineering ──────────────────────────────────────
    date_col = cfg.get("feature_engineering", {}).get("date_column", "Operation_Date")
    derived_cols = cfg.get("feature_engineering", {}).get("derived_columns", []) or []
    df = add_calendar_features(df, date_col, derived_cols)

    df.to_csv(os.path.join(raw_dir, "raw_data.csv"), index=False)

    # ── Chronological train / validation / test split ────────────────────
    split_cfg  = cfg["data_split"]
    train_size = split_cfg.get("train_size", 0.70)
    val_size   = split_cfg.get("val_size", 0.15)
    test_size  = split_cfg.get("test_size", 0.15)

    if date_col in df.columns:
        df = df.sort_values(by=date_col).reset_index(drop=True)
        logger.info(f"  Sorted rows chronologically by '{date_col}' (no random split).")
    else:
        logger.warning(f"  '{date_col}' not present — falling back to existing row order for the split.")

    n = len(df)
    train_end = int(n * train_size)
    val_end   = int(n * (train_size + val_size))

    train_df = df.iloc[:train_end].reset_index(drop=True)
    val_df   = df.iloc[train_end:val_end].reset_index(drop=True)
    test_df  = df.iloc[val_end:].reset_index(drop=True)

    train_df.to_csv(os.path.join(proc_dir, "train.csv"), index=False)
    val_df.to_csv(os.path.join(proc_dir, "val.csv"),   index=False)
    test_df.to_csv(os.path.join(proc_dir, "test.csv"),  index=False)

    # ── Console report ────────────────────────────────────────────────────
    logger.info(f"  Total rows       : {total_rows}")
    logger.info(f"  Duplicate rows   : {duplicate_rows}")
    logger.info(f"  Missing values   : {missing_total} ({missing_pct}%)")
    logger.info(f"  Constant cols    : {constant_cols}")
    logger.info(f"  Train rows       : {len(train_df)}  ({train_size*100:.0f}%)")
    logger.info(f"  Val rows         : {len(val_df)}  ({val_size*100:.0f}%)")
    logger.info(f"  Test rows        : {len(test_df)}  ({test_size*100:.0f}%)")
    logger.info(f"  Columns          : {df.shape[1]}")

    # ── Write artifact ────────────────────────────────────────────────────
    report = {
        "stage": "data_ingestion",
        "raw_file": raw_file,
        "target_column": target_col,
        "problem_type": problem_type,
        "split_strategy": "chronological",
        "date_column": date_col,
        "total_rows_raw": total_rows,
        "duplicate_rows_dropped": duplicate_rows,
        "total_missing_values": missing_total,
        "missing_pct": missing_pct,
        "constant_columns": constant_cols,
        "empty_columns_dropped": empty_cols,
        "train_rows": len(train_df),
        "val_rows": len(val_df),
        "test_rows": len(test_df),
        "n_columns": df.shape[1],
        "column_list": df.columns.tolist(),
    }

    artifact_path = os.path.join(arts_dir, "ingestion_report.json")
    with open(artifact_path, "w") as f:
        json.dump(report, f, indent=2, default=str)

    # ── MLflow ────────────────────────────────────────────────────────────
    init_mlflow(cfg)
    with mlflow.start_run(run_name="Stage_1_Data_Ingestion"):
        safe_log_params({
            "target_column":  target_col,
            "problem_type":   problem_type,
            "split_strategy": "chronological",
            "train_size":     train_size,
            "val_size":       val_size,
            "test_size":      test_size,
            "n_columns":      df.shape[1],
        })
        safe_log_metrics({
            "total_rows":           total_rows,
            "train_rows":           len(train_df),
            "val_rows":             len(val_df),
            "test_rows":            len(test_df),
            "duplicate_rows":       duplicate_rows,
            "total_missing_values": missing_total,
            "missing_pct":          missing_pct,
        })
        safe_log_artifact(artifact_path, "ingestion")

    logger.info("Stage 1 — Data Ingestion COMPLETE ✓")
    return report


if __name__ == "__main__":
    initiate_data_ingestion()
