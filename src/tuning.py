"""
STAGE 4 — AUTOGLUON MODEL TRAINING & SELECTION
───────────────────────────────────────────────
AutoGluon TabularPredictor handles:
  - Model selection (RF, GBT, XGB, LGB, CatBoost, NN, Linear, Ensembles)
  - Hyperparameter optimization
  - Stacking & ensembling
  - Best model selection
  - Internal cross-validation

Pipeline responsibilities:
  - Detect problem type (regression / binary / multi-class classification)
  - Set correct AutoGluon eval_metric aligned with requirements:
      Classification → f1_weighted (primary), accuracy (secondary)
      Regression     → rmse (primary), r2 (secondary)
  - Run AutoGluon with time_limit from config
  - Save leaderboard to artifacts
  - Save AutoGluon predictor directory (loadable via TabularPredictor.load())
  - Log all metrics to MLflow
"""

import os
import sys
import json
import time
import warnings
import shutil
import pandas as pd
import numpy as np

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(__file__))
from utils import (load_config, base_dir, init_mlflow,
                   safe_log_artifact, safe_log_metrics, safe_log_params, get_logger)
import mlflow

logger = get_logger("tuning")


def detect_problem_type(series: pd.Series, threshold: int = 15) -> str:
    if pd.api.types.is_numeric_dtype(series) and series.nunique() > threshold:
        return "regression"
    n_unique = series.nunique()
    if n_unique == 2:
        return "binary"
    return "multiclass"


def get_ag_eval_metric(problem_type: str) -> str:
    """
    Select AutoGluon eval metric per requirements:
      Binary classification     → f1
      Multiclass classification → f1_weighted (AutoGluon has no plain "f1" for multiclass)
      Regression                 → r2 (model selection is driven by Validation R² Score)
    """
    if problem_type == "binary":
        return "f1"
    if problem_type == "multiclass":
        return "f1_weighted"
    return "r2"


def run_automl_tournament():
    logger.info("=" * 60)
    logger.info("STAGE 4: AUTOGLUON MODEL TRAINING & SELECTION")
    logger.info("=" * 60)

    cfg        = load_config()
    bd         = base_dir()
    proc_dir   = os.path.join(bd, cfg["paths"]["processed_dir"])
    arts_dir   = os.path.join(bd, cfg["paths"]["artifacts_dir"], "tuning")
    models_dir = os.path.join(bd, cfg["paths"].get("models_dir", "artifacts/models"))
    os.makedirs(arts_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)

    # ── Load AutoGluon-ready data ─────────────────────────────────────────
    train_path = os.path.join(proc_dir, "train_autogluon.csv")
    if not os.path.exists(train_path):
        # Fallback to raw train split if prep stage not run separately
        train_path = os.path.join(proc_dir, "train.csv")
    if not os.path.exists(train_path):
        raise FileNotFoundError("train data not found — run Stage 3 first.")

    val_path = os.path.join(proc_dir, "val_autogluon.csv")
    if not os.path.exists(val_path):
        val_path = os.path.join(proc_dir, "val.csv")

    train_data = pd.read_csv(train_path, low_memory=False)
    val_data   = pd.read_csv(val_path, low_memory=False) if os.path.exists(val_path) else None
    target_col = cfg.get("data_schema", {}).get("target_column") or train_data.columns[-1]

    problem_type = detect_problem_type(train_data[target_col])
    eval_metric  = get_ag_eval_metric(problem_type)

    logger.info(f"Target        : '{target_col}'")
    logger.info(f"Problem type  : {problem_type.upper()}")
    logger.info(f"Eval metric   : {eval_metric}")
    logger.info(f"Train shape   : {train_data.shape}")
    if val_data is not None:
        logger.info(f"Val shape     : {val_data.shape}  (used as AutoGluon tuning_data)")

    # ── AutoGluon config ──────────────────────────────────────────────────
    ag_cfg       = cfg.get("autogluon", {})
    time_limit   = int(ag_cfg.get("time_limit", 300))
    presets      = ag_cfg.get("presets", "best_quality")
    verbosity    = int(ag_cfg.get("verbosity", 2))
    num_bag_folds   = ag_cfg.get("num_bag_folds")   # None → AutoGluon decides
    num_stack_levels= ag_cfg.get("num_stack_levels") # None → AutoGluon decides

    logger.info(f"Time limit    : {time_limit}s")
    logger.info(f"Presets       : {presets}")

    # ── Predictor save path ───────────────────────────────────────────────
    # AutoGluon saves a directory, not a single file
    predictor_path = os.path.join(models_dir, "autogluon_predictor")

    # Remove stale predictor if exists (fresh run)
    if os.path.exists(predictor_path):
        shutil.rmtree(predictor_path)
        logger.info(f"  Removed stale predictor at {predictor_path}")

    # ── Import AutoGluon ──────────────────────────────────────────────────
    try:
        from autogluon.tabular import TabularPredictor
    except ImportError:
        raise ImportError(
            "AutoGluon not installed. Run:\n"
            "  pip install autogluon.tabular"
        )

    # ── Build fit kwargs ──────────────────────────────────────────────────
    fit_kwargs = dict(
        time_limit=time_limit,
        presets=presets,
        verbosity=verbosity,
        excluded_model_types=["NN_TORCH", "FASTAI"],
    )
    if num_bag_folds is not None:
        fit_kwargs["num_bag_folds"] = int(num_bag_folds)
    if num_stack_levels is not None:
        fit_kwargs["num_stack_levels"] = int(num_stack_levels)
    # Use the explicit chronological validation split (rather than AutoGluon's
    # internal bagged CV) so model selection respects the 70/15/15 time-based split.
    if val_data is not None:
        fit_kwargs["tuning_data"] = val_data
        fit_kwargs["use_bag_holdout"] = True

    # ── Train ─────────────────────────────────────────────────────────────
    logger.info("")
    logger.info("  Starting AutoGluon training ...")
    logger.info(f"  Budget: {time_limit} seconds")
    logger.info("─" * 60)

    t0 = time.time()
    predictor = TabularPredictor(
        label=target_col,
        eval_metric=eval_metric,
        path=predictor_path,
        problem_type=problem_type,
        verbosity=verbosity,
    ).fit(
        train_data=train_data,
        **fit_kwargs,
    )
    elapsed = time.time() - t0

    logger.info("─" * 60)
    logger.info(f"  AutoGluon training complete in {elapsed:.1f}s")

    # ── Leaderboard ───────────────────────────────────────────────────────
    leaderboard = predictor.leaderboard(silent=True)

    # Best model: get_model_best() removed in AutoGluon 1.5
    # Use model_best property (AG 1.3-1.4) or leaderboard top row (AG 1.5+)
    if hasattr(predictor, "model_best"):
        best_model_name = predictor.model_best
    else:
        best_model_name = leaderboard.iloc[0]["model"]

    # Primary metric score of the best model (from leaderboard)
    best_row = leaderboard[leaderboard["model"] == best_model_name].iloc[0]
    best_score_val = float(best_row["score_val"])

    logger.info("")
    logger.info("  ╔══════════════════════════════════════════════╗")
    logger.info(f"  ║  BEST MODEL : {best_model_name:<30}║")
    logger.info(f"  ║  METRIC     : {eval_metric:<30}║")
    logger.info(f"  ║  VAL SCORE  : {best_score_val:<30.6f}║")
    logger.info(f"  ║  TRAINING   : {elapsed:.1f}s{'':<27}║")
    logger.info("  ╚══════════════════════════════════════════════╝")

    # Print top-5 leaderboard
    logger.info("")
    logger.info("  Top models:")
    for _, row in leaderboard.head(8).iterrows():
        logger.info(f"    {row['model']:<40} score_val={row['score_val']:.4f}")

    # ── Save leaderboard CSV ──────────────────────────────────────────────
    lb_path = os.path.join(arts_dir, "autogluon_leaderboard.csv")
    leaderboard.to_csv(lb_path, index=False)

    # ── Save training metadata JSON ───────────────────────────────────────
    metadata = {
        "stage":             "autogluon_training",
        "target_column":     target_col,
        "problem_type":      problem_type,
        "eval_metric":       eval_metric,
        "time_limit_s":      time_limit,
        "actual_time_s":     round(elapsed, 2),
        "presets":           presets,
        "best_model":        best_model_name,
        "best_val_score":    best_score_val,
        "n_models_trained":  len(leaderboard),
        "predictor_path":    predictor_path,
        "train_shape":       list(train_data.shape),
        "val_shape":         list(val_data.shape) if val_data is not None else None,
    }
    meta_path = os.path.join(arts_dir, "training_metadata.json")
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)

    # ── MLflow logging ────────────────────────────────────────────────────
    init_mlflow(cfg)
    with mlflow.start_run(run_name="Stage_4_AutoGluon_Training") as parent_run:
        parent_run_id = parent_run.info.run_id

        normalized_problem_type = (
            "classification" if problem_type in ("binary", "multiclass") else problem_type
        )
        safe_log_params({
            "target_column":    target_col,
            "problem_type":     normalized_problem_type,
            "eval_metric":      eval_metric,
            "time_limit_s":     time_limit,
            "presets":          presets,
            "selected_model":   best_model_name,
            "n_models_trained": len(leaderboard),
        })
        safe_log_metrics({
            "best_val_score":  best_score_val,
            "training_time_s": elapsed,
        })
        safe_log_artifact(lb_path,   "tuning")
        safe_log_artifact(meta_path, "tuning")

        # Log each model in the leaderboard as a nested run
        for _, row in leaderboard.iterrows():
            with mlflow.start_run(run_name=f"AG_{row['model']}", nested=True):
                safe_log_params({"model": row["model"]})
                log_metrics = {"score_val": float(row["score_val"])}
                if "fit_time" in row and not pd.isna(row.get("fit_time", None)):
                    log_metrics["fit_time_s"] = float(row["fit_time"])
                if "pred_time_val" in row and not pd.isna(row.get("pred_time_val", None)):
                    log_metrics["pred_time_val_s"] = float(row["pred_time_val"])
                safe_log_metrics(log_metrics)

    logger.info("Stage 4 — AutoGluon Training COMPLETE ✓")
    return metadata


if __name__ == "__main__":
    run_automl_tournament()
