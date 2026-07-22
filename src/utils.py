"""
Shared utilities: config loader, MLflow helpers, logger.
"""
import os
import sys
import yaml
import logging
import mlflow
from datetime import datetime

# ─── Logging ────────────────────────────────────────────────────────────────

def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        fmt = logging.Formatter("[%(asctime)s] %(levelname)s  %(name)s — %(message)s",
                                datefmt="%H:%M:%S")
        sh = logging.StreamHandler(sys.stdout)
        sh.setFormatter(fmt)
        logger.addHandler(sh)
        logger.setLevel(logging.INFO)
    return logger

# ─── Config ─────────────────────────────────────────────────────────────────

def load_config(config_path: str | None = None) -> dict:
    if config_path is None:
        # works whether called from src/ or project root
        here = os.path.dirname(os.path.abspath(__file__))
        config_path = os.path.join(here, "..", "config.yaml")
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def base_dir(config: dict | None = None) -> str:
    here = os.path.dirname(os.path.abspath(__file__))
    return os.path.abspath(os.path.join(here, ".."))

# ─── MLflow helpers ─────────────────────────────────────────────────────────

def init_mlflow(config: dict):
    """Connect to MLflow; fall back to local SQLite DB if remote unreachable."""
    uri = config["mlflow"]["tracking_uri"]
    exp  = config["mlflow"]["experiment_name"]
    logger = get_logger("mlflow-init")
    try:
        mlflow.set_tracking_uri(uri)
        mlflow.set_experiment(exp)
        logger.info(f"MLflow → {uri}  |  experiment: {exp}")
    except Exception as exc:
        fallback_db = os.path.join(base_dir(), "mlflow_local.db")
        fallback_uri = f"sqlite:///{fallback_db}"
        logger.warning(f"Remote MLflow unreachable. Falling back to local SQLite: {fallback_db}")
        os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"
        mlflow.set_tracking_uri(fallback_uri)
        try:
            mlflow.set_experiment(exp)
        except Exception as exc2:
            # Last resort: allow file store
            fallback_file = os.path.join(base_dir(), "mlruns")
            os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"
            mlflow.set_tracking_uri(f"file://{fallback_file}")
            mlflow.set_experiment(exp)
            logger.warning(f"SQLite fallback also failed ({exc2}). Using file store: {fallback_file}")

def safe_log_artifact(local_path: str, artifact_subdir: str | None = None):
    """Log artifact, silently skip on network failures."""
    logger = get_logger("mlflow-artifact")
    try:
        mlflow.log_artifact(local_path, artifact_path=artifact_subdir)
    except Exception as exc:
        logger.warning(f"Could not upload artifact {local_path}: {exc}")

def safe_log_metrics(metrics: dict):
    try:
        mlflow.log_metrics({k: float(v) for k, v in metrics.items()})
    except Exception:
        pass

def safe_log_params(params: dict):
    try:
        mlflow.log_params({k: str(v) for k, v in params.items()})
    except Exception:
        pass
