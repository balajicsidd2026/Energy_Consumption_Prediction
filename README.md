# Energy Consumption Prediction

Cloned from the same MLOps pipeline used for **Cargo Damage Prediction** /
**SLA Breach Prediction**. Folder hierarchy, naming conventions, MLflow
integration, logging, exception-safety helpers, AutoGluon training/evaluation
flow, and Streamlit app structure are all unchanged — only the dataset,
target, feature schema, problem type, and business logic were adapted for
**Energy Consumption Prediction** (SAL Saudi Logistics Services — King
Abdulaziz International Airport, Jeddah).

## Usage

```bash
pip install -r requirements.txt
python run_pipeline.py          # runs all 5 stages end-to-end
streamlit run app.py            # launches the prediction + analytics UI
```

## Pipeline Stages (same order/names as the original project)

1. **data_ingestion** — loads `energy_consumption_prediction_dataset.csv`.
   `Year`/`Month`/`Quarter`/`Day_of_Week`/`Weekend` already exist in the raw
   data, so calendar derivation is skipped automatically (config-driven).
   Performs a **chronological** 70/15/15 train/validation/test split, sorted
   by `Operation_Date` — no random shuffling.
2. **eda** — unchanged, dataset/schema-agnostic exploratory analysis.
3. **preprocessing** — no leakage/ID columns to drop for this dataset
   (`Warehouse_ID` is a legitimate low-cardinality feature, not an
   identifier); writes `train_autogluon.csv` / `val_autogluon.csv` /
   `test_autogluon.csv`.
4. **tuning** — AutoGluon `TabularPredictor` trains against
   `Energy_Consumption_kWh` (regression). Internally this covers Linear
   Regression, Decision Tree, Random Forest, Extra Trees, Gradient Boosting,
   XGBoost, LightGBM, CatBoost, neural nets, and weighted ensembles — the
   exact model pool requested. `eval_metric="r2"` so model selection is
   driven by **Validation R² Score**, using the chronological validation
   split as `tuning_data`.
5. **evaluation** — MAE, MSE, RMSE, R², **Adjusted R²** (added for this
   project), MAPE, actual-vs-predicted / residual plots, permutation feature
   importance — logged to MLflow exactly as before (experiment renamed to
   `Energy_Consumption_Prediction`).

## What changed vs. the prior projects

| Aspect | SLA Breach Prediction | Energy Consumption Prediction |
|---|---|---|
| Target | `SLA_Breach` (binary) | `Energy_Consumption_kWh` (regression) |
| AutoGluon eval metric | `f1` | `r2` |
| Drop columns | leakage/ID columns present | none needed — no leakage/ID columns in this dataset |
| Calendar features | derived in Stage 1 | already present in raw data — derivation skipped |
| Regression-only metric | — | `adjusted_r2` added in `evaluation.py` |
| MLflow experiment | `SLA_Breach_Prediction` | `Energy_Consumption_Prediction` |
| Streamlit pages | Overview, Prediction | Overview, **Model Description**, Prediction, **Analytics Dashboard** |

Everything else — `src/utils.py`, `src/eda.py`,
`src/mlflow_autogluon_flavor.py`, `run_pipeline.py`, `requirements.txt`,
the ingestion/preprocessing/tuning/evaluation code paths — is reused
verbatim or with only config-driven, non-structural adjustments.

## Streamlit App

- **Overview** — use case, business objectives, key benefits.
- **Model Description** — selected model, metrics (R², Adjusted R², RMSE,
  MAE, MSE, MAPE), full leaderboard, permutation feature importance.
- **Prediction** — single-record manual entry or fetch-by-warehouse/date,
  plus batch CSV upload; both show the predicted kWh and top contributing
  features.
- **Analytics Dashboard** — KPI cards (avg/max/min energy), warehouse-wise
  average energy, monthly/yearly trends, weather/occupancy/temperature
  impact, energy distribution, correlation heatmap, feature importance, and
  prediction-distribution/residual plots computed live against the test set.
