"""
mlflow_autogluon_flavor.py
────────────────────────────
AutoGluon has no official MLflow model flavor (mlflow.autogluon does not
exist — confirmed against MLflow's flavor list and AutoGluon's own open
GitHub issue tracking this gap). Logging the predictor directory via
mlflow.log_artifacts() alone makes the files visible in the run's plain
Artifacts browser, but does NOT register the model as a first-class MLflow
"Logged Model" — that requires a pyfunc.PythonModel wrapper passed to
mlflow.pyfunc.log_model(). This is why "I can see metrics but not models"
happens: artifacts != logged models in MLflow's data model.

This module provides that wrapper so AutoGluon predictors show up in the
MLflow Models page like any other framework's model.
"""

import os
import pandas as pd
import mlflow.pyfunc


class AutoGluonPyFuncWrapper(mlflow.pyfunc.PythonModel):
    """
    Wraps an AutoGluon TabularPredictor for MLflow logging/serving.

    On load, finds the predictor directory under context.artifacts (the
    path MLflow downloads it to at load time, not the original training
    machine's path) and loads it via TabularPredictor.load().
    """

    def load_context(self, context):
        from autogluon.tabular import TabularPredictor
        predictor_dir = context.artifacts["predictor"]
        self.predictor = TabularPredictor.load(predictor_dir)
        self.problem_type = self.predictor.problem_type

    def predict(self, context, model_input: pd.DataFrame, params: dict = None):
        """
        Returns class predictions for classification, or point predictions
        for regression. Set params={"return_proba": True} to get class
        probabilities instead (classification only).
        """
        if params and params.get("return_proba") and self.problem_type in ("binary", "multiclass"):
            return self.predictor.predict_proba(model_input)
        return self.predictor.predict(model_input)


def log_autogluon_model(predictor_path: str, artifact_subpath: str = "model",
                         signature=None, input_example=None,
                         registered_model_name: str = None):
    """
    Log an AutoGluon predictor directory as a proper MLflow Logged Model
    using the pyfunc wrapper above. Returns True on success, False on
    failure (never raises — pipeline should continue even if this fails).
    """
    import logging
    logger = logging.getLogger("mlflow-autogluon-model")
    try:
        mlflow.pyfunc.log_model(
            artifact_path=artifact_subpath,
            python_model=AutoGluonPyFuncWrapper(),
            artifacts={"predictor": predictor_path},
            signature=signature,
            input_example=input_example,
            pip_requirements=[
                "autogluon.tabular",
                "pandas",
                "scikit-learn",
            ],
            registered_model_name=registered_model_name,
        )
        logger.info(f"  Logged Model registered in MLflow (pyfunc/AutoGluon): '{artifact_subpath}'")
        return True
    except Exception as exc:
        logger.warning(f"  Could not log AutoGluon model to MLflow Models: {exc}")
        return False
