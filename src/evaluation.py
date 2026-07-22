"""
STAGE 5 — MODEL EVALUATION (AutoGluon)
───────────────────────────────────────
Loads saved AutoGluon predictor and evaluates on held-out test set.

Classification metrics  : F1 (primary), Accuracy, Precision, Recall, ROC-AUC
Regression metrics      : R² (primary), RMSE, MAE

Charts generated:
  Classification:
    - Confusion matrix
    - ROC curve / multi-class OvR
    - Precision-Recall curve
    - AutoGluon feature importance (bar chart)
  Regression:
    - Actual vs Predicted scatter + residual plot
    - Residual distribution histogram
    - AutoGluon feature importance

All artifacts + metrics logged to MLflow.
AutoGluon predictor directory saved as loadable artifact.
"""

import os
import sys
import json
import warnings
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
    root_mean_squared_error, mean_absolute_error, r2_score,
    mean_squared_error, explained_variance_score,
)

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(__file__))
from utils import (load_config, base_dir, init_mlflow,
                   safe_log_artifact, safe_log_metrics, safe_log_params, get_logger)
from mlflow_autogluon_flavor import log_autogluon_model
import mlflow
from mlflow.models import infer_signature

logger = get_logger("evaluation")
sns.set_theme(style="darkgrid")


def savefig(fig, path: str):
    fig.tight_layout()
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"  Saved: {os.path.basename(path)}")


# ─── Classification charts ────────────────────────────────────────────────────

def plot_confusion_matrix(y_true, y_pred, labels, out_dir):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(max(6, len(labels) * 0.9), max(5, len(labels) * 0.8)))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=labels, yticklabels=labels, ax=ax)
    ax.set_title("Confusion Matrix — Test Set", fontsize=13)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    path = os.path.join(out_dir, "01_confusion_matrix.png")
    savefig(fig, path)
    return path


def plot_roc_curves(y_true, y_score, labels, out_dir):
    from sklearn.preprocessing import label_binarize
    from sklearn.metrics import roc_curve, auc
    paths = []
    if len(labels) == 2:
        scores = y_score[:, 1] if y_score.ndim == 2 else y_score
        fpr, tpr, _ = roc_curve(y_true, scores)
        roc_auc = auc(fpr, tpr)
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.plot(fpr, tpr, lw=2, label=f"AUC = {roc_auc:.4f}")
        ax.plot([0, 1], [0, 1], "k--", lw=1)
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title("ROC Curve (Binary)")
        ax.legend(loc="lower right")
        p = os.path.join(out_dir, "02_roc_curve.png")
        savefig(fig, p)
        paths.append(p)
    else:
        y_bin = label_binarize(y_true, classes=list(range(len(labels))))
        fig, ax = plt.subplots(figsize=(8, 6))
        for i, lbl in enumerate(labels[:12]):
            try:
                fpr, tpr, _ = roc_curve(y_bin[:, i], y_score[:, i])
                ax.plot(fpr, tpr, lw=1.2, label=f"{lbl} (AUC={auc(fpr,tpr):.2f})")
            except Exception:
                pass
        ax.plot([0, 1], [0, 1], "k--", lw=1)
        ax.set_title("ROC Curves (Multi-class OvR)")
        ax.set_xlabel("FPR")
        ax.set_ylabel("TPR")
        ax.legend(loc="lower right", fontsize=7)
        p = os.path.join(out_dir, "02_roc_multiclass.png")
        savefig(fig, p)
        paths.append(p)
    return paths


def plot_pr_curve(y_true, y_score, labels, out_dir):
    from sklearn.metrics import precision_recall_curve, average_precision_score
    from sklearn.preprocessing import label_binarize
    fig, ax = plt.subplots(figsize=(6, 5))
    if len(labels) == 2:
        scores = y_score[:, 1] if y_score.ndim == 2 else y_score
        prec, rec, _ = precision_recall_curve(y_true, scores)
        ap = average_precision_score(y_true, scores)
        ax.plot(rec, prec, lw=2, label=f"AP = {ap:.4f}")
    else:
        y_bin = label_binarize(y_true, classes=list(range(len(labels))))
        for i, lbl in enumerate(labels[:12]):
            try:
                prec, rec, _ = precision_recall_curve(y_bin[:, i], y_score[:, i])
                ax.plot(rec, prec, lw=1, label=str(lbl))
            except Exception:
                pass
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curve")
    ax.legend(fontsize=7)
    path = os.path.join(out_dir, "03_pr_curve.png")
    savefig(fig, path)
    return path


# ─── Regression charts ────────────────────────────────────────────────────────

def plot_actual_vs_predicted(y_true, y_pred, out_dir):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].scatter(y_true, y_pred, alpha=0.3, s=8, color="#4C72B0")
    mn = min(float(y_true.min()), float(y_pred.min()))
    mx = max(float(y_true.max()), float(y_pred.max()))
    axes[0].plot([mn, mx], [mn, mx], "r--", lw=1.5, label="Perfect fit")
    axes[0].set_xlabel("Actual")
    axes[0].set_ylabel("Predicted")
    axes[0].set_title("Actual vs Predicted")
    axes[0].legend()
    residuals = np.array(y_true) - np.array(y_pred)
    axes[1].scatter(y_pred, residuals, alpha=0.3, s=8, color="#DD8452")
    axes[1].axhline(0, color="r", lw=1.5, linestyle="--")
    axes[1].set_xlabel("Predicted")
    axes[1].set_ylabel("Residual")
    axes[1].set_title("Residual Plot")
    path = os.path.join(out_dir, "01_actual_vs_predicted.png")
    savefig(fig, path)
    return path


def plot_residual_distribution(y_true, y_pred, out_dir):
    residuals = np.array(y_true) - np.array(y_pred)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(residuals, bins=60, color="#55A868", edgecolor="white", alpha=0.85)
    ax.axvline(0, color="r", lw=1.5)
    ax.set_title("Residual Distribution")
    ax.set_xlabel("Residual")
    path = os.path.join(out_dir, "02_residual_distribution.png")
    savefig(fig, path)
    return path


# ─── Feature importance via AutoGluon ────────────────────────────────────────

def plot_ag_feature_importance(predictor, test_data, target_col, out_dir, subsample=2000):
    """Use AutoGluon's built-in permutation feature importance."""
    try:
        sample = test_data if len(test_data) <= subsample else test_data.sample(subsample, random_state=42)
        fi = predictor.feature_importance(data=sample, silent=True)
        fi = fi.sort_values("importance", ascending=False).head(30)
        fig, ax = plt.subplots(figsize=(9, max(5, len(fi) * 0.32)))
        colors = ["#4C72B0" if v >= 0 else "#C44E52" for v in fi["importance"]]
        ax.barh(fi.index[::-1], fi["importance"][::-1], color=colors[::-1])
        ax.set_title("AutoGluon Feature Importance (Permutation)", fontsize=13)
        ax.set_xlabel("Importance Score")
        ax.axvline(0, color="k", lw=0.8, linestyle="--")
        path = os.path.join(out_dir, "04_feature_importance.png")
        savefig(fig, path)

        # Save as CSV too
        fi_csv = os.path.join(out_dir, "feature_importance.csv")
        fi.to_csv(fi_csv)
        logger.info(f"  Saved: feature_importance.csv")
        return path, fi_csv
    except Exception as e:
        logger.warning(f"  Feature importance failed: {e}")
        return None, None


# ─── Main ─────────────────────────────────────────────────────────────────────

def run_dynamic_evaluation():
    logger.info("=" * 60)
    logger.info("STAGE 5: MODEL EVALUATION (AutoGluon)")
    logger.info("=" * 60)

    cfg        = load_config()
    bd         = base_dir()
    proc_dir   = os.path.join(bd, cfg["paths"]["processed_dir"])
    models_dir = os.path.join(bd, cfg["paths"].get("models_dir", "artifacts/models"))
    arts_dir   = os.path.join(bd, cfg["paths"]["artifacts_dir"], "evaluation")
    os.makedirs(arts_dir, exist_ok=True)

    # ── Load AutoGluon predictor ──────────────────────────────────────────
    predictor_path = os.path.join(models_dir, "autogluon_predictor")
    if not os.path.exists(predictor_path):
        raise FileNotFoundError(
            f"AutoGluon predictor not found at {predictor_path} — run Stage 4 first."
        )

    try:
        from autogluon.tabular import TabularPredictor
    except ImportError:
        raise ImportError("AutoGluon not installed. Run: pip install autogluon.tabular")

    logger.info(f"  Loading predictor from: {predictor_path}")
    predictor   = TabularPredictor.load(predictor_path)
    target_col  = predictor.label
    problem_type= predictor.problem_type   # "binary" | "multiclass" | "regression"
    # get_model_best() removed in AutoGluon 1.5 — use leaderboard top row
    _lb = predictor.leaderboard(silent=True)
    if hasattr(predictor, "model_best"):
        best_model = predictor.model_best
    else:
        best_model = _lb.iloc[0]["model"]
    eval_metric = predictor.eval_metric

    logger.info(f"  Target        : '{target_col}'")
    logger.info(f"  Problem type  : {problem_type.upper()}")
    logger.info(f"  Best model    : {best_model}")
    logger.info(f"  Eval metric   : {eval_metric}")

    # ── Load test data ────────────────────────────────────────────────────
    test_path = os.path.join(proc_dir, "test_autogluon.csv")
    if not os.path.exists(test_path):
        test_path = os.path.join(proc_dir, "test.csv")
    test_data = pd.read_csv(test_path, low_memory=False)
    logger.info(f"  Test data     : {test_data.shape}")

    X_test = test_data.drop(columns=[target_col], errors="ignore")
    y_test = test_data[target_col] if target_col in test_data.columns else None

    # ── Predictions ───────────────────────────────────────────────────────
    y_pred = predictor.predict(X_test)
    artifact_paths = []
    metrics = {}

    if problem_type in ("binary", "multiclass"):
        # ── Classification metrics ────────────────────────────────────────
        # Get class labels from predictor
        class_labels = predictor.class_labels
        if class_labels is None:
            class_labels = sorted(y_test.unique().tolist())
        label_to_idx = {lbl: i for i, lbl in enumerate(class_labels)}
        display_labels = [str(l) for l in class_labels]

        # Encode y_test & y_pred to int indices for sklearn metrics
        y_test_enc = y_test.map(label_to_idx)
        y_pred_enc = pd.Series(y_pred).map(label_to_idx)

        # Standardized metric names (required: accuracy, precision, recall, f1_score)
        metrics["accuracy"]  = float(accuracy_score(y_test_enc, y_pred_enc))
        metrics["precision"] = float(precision_score(y_test_enc, y_pred_enc, average="weighted", zero_division=0))
        metrics["recall"]    = float(recall_score(y_test_enc, y_pred_enc, average="weighted",    zero_division=0))
        metrics["f1_score"]  = float(f1_score(y_test_enc, y_pred_enc, average="weighted",        zero_division=0))

        # ROC-AUC (requires probabilities)
        try:
            y_proba = predictor.predict_proba(X_test)
            # Reorder columns to match class_labels order
            y_proba_arr = y_proba[[str(l) for l in class_labels]].values if \
                          all(str(l) in y_proba.columns for l in class_labels) else y_proba.values

            if problem_type == "binary":
                metrics["roc_auc"] = float(roc_auc_score(y_test_enc, y_proba_arr[:, 1]))
            else:
                metrics["roc_auc_ovr_weighted"] = float(
                    roc_auc_score(y_test_enc, y_proba_arr, multi_class="ovr", average="weighted")
                )

            roc_paths = plot_roc_curves(y_test_enc.values, y_proba_arr, display_labels, arts_dir)
            artifact_paths.extend(roc_paths)
            pr_path = plot_pr_curve(y_test_enc.values, y_proba_arr, display_labels, arts_dir)
            artifact_paths.append(pr_path)
        except Exception as e:
            logger.warning(f"  ROC/PR curves skipped: {e}")

        cm_path = plot_confusion_matrix(y_test_enc, y_pred_enc, display_labels, arts_dir)
        artifact_paths.append(cm_path)

        # Classification text report
        cr = classification_report(y_test_enc, y_pred_enc, target_names=display_labels, zero_division=0)
        cr_path = os.path.join(arts_dir, "classification_report.txt")
        with open(cr_path, "w") as f:
            f.write(f"Best Model: {best_model}\n")
            f.write(f"Eval Metric: {eval_metric}\n\n")
            f.write(cr)
        artifact_paths.append(cr_path)

    else:
        # ── Regression metrics ────────────────────────────────────────────
        import numpy as np
        # Standardized metric names (required: r2_score, rmse, mae, mse)
        # RMSE, MAE, MSE are calculated on delays (range ~ [-10, 84])
        metrics["rmse"]             = float(root_mean_squared_error(y_test, y_pred))
        metrics["mae"]              = float(mean_absolute_error(y_test, y_pred))
        metrics["mse"]              = float(mean_squared_error(y_test, y_pred))

        # Reconstruct actual arrival minutes from midnight to calculate R2 and MAPE (aligning with original metrics)
        if "Scheduled_Arrival_Time" in X_test.columns:
            sched_min = X_test["Scheduled_Arrival_Time"].values
            act_true = (sched_min + y_test) % 1440
            act_pred = (sched_min + y_pred) % 1440
            metrics["r2_score"] = float(r2_score(act_true, act_pred))
            metrics["explained_variance"] = float(explained_variance_score(act_true, act_pred))
            
            # Compute MAPE using actual arrival times as denominator (safely handling zero)
            diff_circular = (y_test - y_pred + 720) % 1440 - 720
            non_zero_mask = act_true > 1e-6
            if non_zero_mask.sum() > 0:
                metrics["mape"] = float(np.mean(np.abs(diff_circular[non_zero_mask]) / act_true[non_zero_mask]) * 100)
            else:
                metrics["mape"] = float("nan")
        else:
            metrics["r2_score"] = float(r2_score(y_test, y_pred))
            metrics["explained_variance"] = float(explained_variance_score(y_test, y_pred))
            # Standard MAPE
            mask = np.abs(y_test) > 1e-6
            if mask.sum() > 0:
                metrics["mape"] = float(np.mean(np.abs(y_test[mask] - y_pred[mask]) / y_test[mask]) * 100)
            else:
                metrics["mape"] = float("nan")

        p1 = plot_actual_vs_predicted(y_test, y_pred, arts_dir)
        p2 = plot_residual_distribution(y_test, y_pred, arts_dir)
        artifact_paths.extend([p1, p2])

        # ── Adjusted R² ────────────────────────────────────────────────────
        # Adjusted R² penalizes R² for the number of predictors used, so it
        # doesn't automatically increase as more features are added.
        n_obs = len(y_test)
        n_features = X_test.shape[1]
        r2_val = metrics["r2_score"]
        if n_obs - n_features - 1 > 0:
            metrics["adjusted_r2"] = float(1 - (1 - r2_val) * (n_obs - 1) / (n_obs - n_features - 1))
        else:
            metrics["adjusted_r2"] = float("nan")

    # ── AutoGluon feature importance ──────────────────────────────────────
    fi_subsample = cfg.get("evaluation", {}).get("feature_importance_subsample", 2000)
    if cfg.get("evaluation", {}).get("run_feature_importance", True):
        fi_path, fi_csv = plot_ag_feature_importance(
            predictor, test_data, target_col, arts_dir, subsample=fi_subsample
        )
        if fi_path:
            artifact_paths.append(fi_path)
        if fi_csv:
            artifact_paths.append(fi_csv)

    # ── Console metrics report ────────────────────────────────────────────
    logger.info("")
    logger.info("  ╔═══════════════════════════════════════════════╗")
    logger.info(f"  ║  Best Model : {best_model:<31}║")
    logger.info(f"  ║  Mode       : {problem_type.upper():<31}║")
    logger.info("  ╠═══════════════════════════════════════════════╣")
    for k, v in metrics.items():
        logger.info(f"  ║  {k:<22} : {v:<20.6f}║")
    logger.info("  ╚═══════════════════════════════════════════════╝")

    # ── Leaderboard from predictor ────────────────────────────────────────
    try:
        lb = predictor.leaderboard(silent=True)
        lb_path = os.path.join(arts_dir, "final_leaderboard.csv")
        lb.to_csv(lb_path, index=False)
        artifact_paths.append(lb_path)
    except Exception:
        pass

    # ── Evaluation report JSON ────────────────────────────────────────────
    eval_report = {
        "stage":         "evaluation",
        "best_model":    best_model,
        "problem_type":  problem_type,
        "eval_metric":   str(eval_metric),
        "target_col":    target_col,
        "test_rows":     len(y_test) if y_test is not None else 0,
        "metrics":       metrics,
        "predictor_path": predictor_path,
    }
    rpt_path = os.path.join(arts_dir, "evaluation_report.json")
    with open(rpt_path, "w") as f:
        json.dump(eval_report, f, indent=2)
    artifact_paths.append(rpt_path)

    # ── MLflow ────────────────────────────────────────────────────────────
    init_mlflow(cfg)
    with mlflow.start_run(run_name="Stage_5_Evaluation"):
        # Normalize problem_type to match traditional pipeline vocabulary
        # ("binary"/"multiclass" -> "classification") for cross-pipeline comparison.
        # AutoGluon's native problem_type ("binary"/"multiclass"/"regression")
        # is still used everywhere else in this script for branching logic.
        normalized_problem_type = (
            "classification" if problem_type in ("binary", "multiclass") else problem_type
        )
        safe_log_params({
            "selected_model": best_model,
            "problem_type":   normalized_problem_type,
            "eval_metric":    str(eval_metric),
            "target_col":     target_col,
        })
        safe_log_metrics(metrics)
        for p in artifact_paths:
            if p and os.path.exists(p):
                safe_log_artifact(p, "evaluation")

        # Log the entire AutoGluon predictor directory as a plain artifact
        # (keeps existing behavior — full directory browsable under Artifacts)
        try:
            mlflow.log_artifacts(predictor_path, artifact_path="autogluon_predictor")
            logger.info("  AutoGluon predictor directory logged to MLflow")
        except Exception as e:
            logger.warning(f"  Could not upload predictor dir to MLflow: {e}")
            logger.info("  Predictor is saved locally and loadable via TabularPredictor.load()")

        # ── Log as a proper MLflow Logged Model (pyfunc wrapper) ───────────
        # This is the piece that was missing: log_artifacts() above puts files
        # in the run's Artifacts browser, but does NOT register a "Logged
        # Model" entity — that's a separate MLflow concept requiring
        # mlflow.<flavor>.log_model() / mlflow.pyfunc.log_model(). AutoGluon
        # has no official flavor, so we use a custom pyfunc wrapper (see
        # mlflow_autogluon_flavor.py). This is what makes the model show up
        # in MLflow's "Models" tab/page, not just buried in Artifacts.
        try:
            signature = infer_signature(X_test, pd.Series(y_pred, name=target_col))
            input_example = X_test.head(5)
        except Exception:
            signature = None
            input_example = None

        log_autogluon_model(
            predictor_path=predictor_path,
            artifact_subpath="model",
            signature=signature,
            input_example=input_example,
        )

    logger.info(f"Stage 5 — Evaluation COMPLETE ✓  ({len(artifact_paths)} artifacts)")
    logger.info(f"")
    logger.info(f"  To load model later:")
    logger.info(f"    from autogluon.tabular import TabularPredictor")
    logger.info(f"    predictor = TabularPredictor.load('{predictor_path}')")
    logger.info(f"    predictions = predictor.predict(new_data)")
    return eval_report


if __name__ == "__main__":
    run_dynamic_evaluation()
