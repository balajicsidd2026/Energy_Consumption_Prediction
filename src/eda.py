"""
STAGE 2 — EXPLORATORY DATA ANALYSIS (EDA)
─────────────────────────────────────────
Works for: regression, binary/multi-class classification, any data (real/synthetic)

Generates:
  1. Target distribution (histogram + box for regression; bar for classification)
  2. Missing data heatmap
  3. Numeric feature distributions (violin plots)
  4. Correlation matrix (Pearson + Spearman)
  5. Outlier summary (IQR method)
  6. Categorical feature value counts
  7. Feature–target relationships (scatter matrix / box plots)
  8. Skewness & kurtosis table
  9. Full ydata-profiling HTML report (optional, large datasets skip)
 10. Full EDA JSON metrics artifact

All charts saved as PNG + uploaded to MLflow.
"""

import os
import sys
import json
import warnings
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from scipy import stats

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(__file__))
from utils import load_config, base_dir, init_mlflow, safe_log_artifact, safe_log_metrics, safe_log_params, get_logger
import mlflow

logger = get_logger("eda")
sns.set_theme(style="darkgrid", palette="muted")
PALETTE = "viridis"


# ─── helpers ────────────────────────────────────────────────────────────────

def savefig(fig, path: str):
    fig.tight_layout()
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"  Saved: {os.path.basename(path)}")


def detect_problem_type(series: pd.Series, threshold: int = 15) -> str:
    if pd.api.types.is_numeric_dtype(series.dtype) and series.nunique() > threshold:
        return "regression"
    return "classification"


def drop_id_cols(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    id_cols = [c for c in df.columns if df[c].nunique() == len(df)
               and df[c].dtype == object]
    return df.drop(columns=id_cols), id_cols


def iqr_outliers(series: pd.Series) -> int:
    q1, q3 = series.quantile(0.25), series.quantile(0.75)
    iqr = q3 - q1
    return int(((series < q1 - 1.5 * iqr) | (series > q3 + 1.5 * iqr)).sum())


# ─── charts ─────────────────────────────────────────────────────────────────

def plot_target_distribution(df, target_col, problem_type, out_dir):
    fig, axes = plt.subplots(1, 2 if problem_type == "regression" else 1,
                             figsize=(12, 4))
    if problem_type == "regression":
        ax1, ax2 = axes
        ax1.hist(df[target_col].dropna(), bins=50, color="#4C72B0", edgecolor="white")
        ax1.set_title(f"Target Distribution — {target_col}")
        ax1.set_xlabel(target_col)
        ax2.boxplot(df[target_col].dropna(), vert=False, patch_artist=True,
                    boxprops=dict(facecolor="#4C72B0", alpha=0.7))
        ax2.set_title("Box Plot")
    else:
        ax = axes if not hasattr(axes, "__len__") else axes[0]
        vc = df[target_col].value_counts()
        ax.bar(vc.index.astype(str), vc.values, color=sns.color_palette(PALETTE, len(vc)))
        ax.set_title(f"Class Distribution — {target_col}")
        ax.set_xlabel("Class")
        ax.set_ylabel("Count")
        for i, v in enumerate(vc.values):
            ax.text(i, v + 5, str(v), ha="center", fontsize=8)
    path = os.path.join(out_dir, "01_target_distribution.png")
    savefig(fig, path)
    return path


def plot_missing_heatmap(df, out_dir):
    missing = df.isnull().sum()
    missing = missing[missing > 0].sort_values(ascending=False)
    if missing.empty:
        logger.info("  No missing values — skipping missing heatmap.")
        return None
    fig, ax = plt.subplots(figsize=(max(8, len(missing) * 0.5), 4))
    ax.bar(missing.index, missing.values, color="#DD8452")
    ax.set_title("Missing Value Count per Feature")
    ax.set_xlabel("Feature")
    ax.set_ylabel("Missing Count")
    plt.xticks(rotation=45, ha="right")
    path = os.path.join(out_dir, "02_missing_heatmap.png")
    savefig(fig, path)
    return path


def plot_correlation_matrix(df, numeric_cols, out_dir):
    if len(numeric_cols) < 2:
        return None
    corr = df[numeric_cols].corr(method="pearson")
    mask = np.triu(np.ones_like(corr, dtype=bool))
    fig, ax = plt.subplots(figsize=(max(8, len(numeric_cols) * 0.7),
                                    max(6, len(numeric_cols) * 0.6)))
    sns.heatmap(corr, mask=mask, annot=len(numeric_cols) <= 20,
                fmt=".2f", cmap="coolwarm", center=0,
                linewidths=0.5, ax=ax, cbar_kws={"shrink": 0.8})
    ax.set_title("Pearson Correlation Matrix")
    path = os.path.join(out_dir, "03_correlation_matrix.png")
    savefig(fig, path)
    return path


def plot_numeric_distributions(df, numeric_cols, target_col, out_dir):
    cols_to_plot = [c for c in numeric_cols if c != target_col][:20]  # cap at 20
    if not cols_to_plot:
        return None
    ncols = 4
    nrows = (len(cols_to_plot) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 4, nrows * 3))
    axes = np.array(axes).flatten()
    for i, col in enumerate(cols_to_plot):
        axes[i].hist(df[col].dropna(), bins=40, color="#55A868", edgecolor="white", alpha=0.85)
        axes[i].set_title(col, fontsize=9)
        sk = df[col].skew()
        axes[i].set_xlabel(f"skew={sk:.2f}", fontsize=8)
    for j in range(len(cols_to_plot), len(axes)):
        axes[j].set_visible(False)
    fig.suptitle("Numeric Feature Distributions", fontsize=14, y=1.01)
    path = os.path.join(out_dir, "04_numeric_distributions.png")
    savefig(fig, path)
    return path


def plot_categorical_distributions(df, cat_cols, out_dir):
    cols_to_plot = [c for c in cat_cols][:16]
    if not cols_to_plot:
        return None
    ncols = 4
    nrows = (len(cols_to_plot) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 4, nrows * 3))
    axes = np.array(axes).flatten()
    for i, col in enumerate(cols_to_plot):
        vc = df[col].value_counts().head(15)
        axes[i].barh(vc.index.astype(str), vc.values,
                     color=sns.color_palette(PALETTE, len(vc)))
        axes[i].set_title(col, fontsize=9)
        axes[i].invert_yaxis()
    for j in range(len(cols_to_plot), len(axes)):
        axes[j].set_visible(False)
    fig.suptitle("Categorical Feature Value Counts", fontsize=14, y=1.01)
    path = os.path.join(out_dir, "05_categorical_distributions.png")
    savefig(fig, path)
    return path


def plot_feature_target(df, numeric_cols, cat_cols, target_col, problem_type, out_dir):
    paths = []
    num_feat = [c for c in numeric_cols if c != target_col]

    if num_feat and problem_type == "regression":
        corr_vals = df[num_feat].corrwith(df[target_col]).abs().sort_values(ascending=False)
        top8 = corr_vals.head(8).index.tolist()
        ncols = 4
        nrows = (len(top8) + ncols - 1) // ncols
        fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 4, nrows * 3))
        axes = np.array(axes).flatten()
        for i, col in enumerate(top8):
            axes[i].scatter(df[col], df[target_col], alpha=0.3, s=5, color="#4C72B0")
            axes[i].set_xlabel(col, fontsize=8)
            axes[i].set_ylabel(target_col, fontsize=8)
            axes[i].set_title(f"r={df[col].corr(df[target_col]):.2f}", fontsize=8)
        for j in range(len(top8), len(axes)):
            axes[j].set_visible(False)
        fig.suptitle("Top Feature vs Target (Scatter)", fontsize=13, y=1.01)
        p = os.path.join(out_dir, "06a_feature_target_scatter.png")
        savefig(fig, p)
        paths.append(p)

    # Filter to low cardinality categories to prevent huge slow-down and unreadable plots
    low_card_cats = [c for c in cat_cols if df[c].nunique() <= 15]

    if low_card_cats and problem_type == "regression":
        top_cat = low_card_cats[:6]
        fig, axes = plt.subplots(1, len(top_cat), figsize=(len(top_cat) * 4, 4))
        if len(top_cat) == 1:
            axes = [axes]
        for i, col in enumerate(top_cat):
            grouped = df.groupby(col)[target_col].apply(lambda x: x.dropna().values)
            groups = grouped.values
            labels = [str(k) for k in grouped.index]
            axes[i].boxplot(groups, labels=labels, patch_artist=True)
            axes[i].set_title(f"{col} vs {target_col}", fontsize=8)
            plt.setp(axes[i].get_xticklabels(), rotation=30, ha="right", fontsize=7)
        fig.suptitle("Categorical Features vs Target (Box)", fontsize=13)
        p = os.path.join(out_dir, "06b_cat_target_box.png")
        savefig(fig, p)
        paths.append(p)

    if low_card_cats and problem_type == "classification":
        if df[target_col].nunique() <= 15:
            top_cat = low_card_cats[:6]
            fig, axes = plt.subplots(1, len(top_cat), figsize=(len(top_cat) * 4, 4))
            if len(top_cat) == 1:
                axes = [axes]
            for i, col in enumerate(top_cat):
                ct = pd.crosstab(df[col], df[target_col], normalize="index")
                ct.plot(kind="bar", stacked=True, ax=axes[i], legend=(i == 0))
                axes[i].set_title(f"{col}", fontsize=8)
                plt.setp(axes[i].get_xticklabels(), rotation=30, ha="right", fontsize=7)
            fig.suptitle("Categorical vs Target (Stacked %)", fontsize=13)
            p = os.path.join(out_dir, "06c_cat_target_stacked.png")
            savefig(fig, p)
            paths.append(p)
        else:
            logger.info(f"  Target '{target_col}' has high cardinality ({df[target_col].nunique()}) — skipping stacked bar charts.")

    return paths


def plot_outlier_summary(df, numeric_cols, out_dir):
    if not numeric_cols:
        return None
    outlier_counts = {c: iqr_outliers(df[c].dropna()) for c in numeric_cols}
    outlier_series = pd.Series(outlier_counts).sort_values(ascending=False)
    outlier_series = outlier_series[outlier_series > 0].head(20)
    if outlier_series.empty:
        return None
    fig, ax = plt.subplots(figsize=(max(8, len(outlier_series) * 0.6), 4))
    ax.bar(outlier_series.index, outlier_series.values, color="#C44E52")
    ax.set_title("Outlier Counts per Feature (IQR Method)")
    ax.set_xlabel("Feature")
    ax.set_ylabel("# Outliers")
    plt.xticks(rotation=45, ha="right")
    path = os.path.join(out_dir, "07_outlier_summary.png")
    savefig(fig, path)
    return path


def generate_html_report(df, target_col, out_dir, max_rows=20_000):
    """Use ydata-profiling if available."""
    try:
        from ydata_profiling import ProfileReport
        sample = df if len(df) <= max_rows else df.sample(max_rows, random_state=42)
        profile = ProfileReport(sample, title="EDA Profile Report",
                                explorative=True, minimal=len(df) > 10_000)
        html_path = os.path.join(out_dir, "eda_profile_report.html")
        profile.to_file(html_path)
        logger.info(f"  Profiling HTML saved: {os.path.basename(html_path)}")
        return html_path
    except ImportError:
        logger.info("  ydata-profiling not installed — skipping HTML report.")
        return None
    except Exception as e:
        logger.warning(f"  Profiling failed: {e}")
        return None


# ─── main ────────────────────────────────────────────────────────────────────

def run_automated_eda():
    logger.info("=" * 60)
    logger.info("STAGE 2: EXPLORATORY DATA ANALYSIS")
    logger.info("=" * 60)

    cfg     = load_config()
    bd      = base_dir()
    out_dir = os.path.join(bd, cfg["paths"]["artifacts_dir"], "eda")
    os.makedirs(out_dir, exist_ok=True)

    train_path = os.path.join(bd, cfg["paths"]["processed_dir"], "train.csv")
    if not os.path.exists(train_path):
        raise FileNotFoundError(f"train.csv not found — run data_ingestion first. ({train_path})")

    df = pd.read_csv(train_path, low_memory=False)
    df, id_cols = drop_id_cols(df)

    target_col   = cfg.get("data_schema", {}).get("target_column") or df.columns[-1]
    problem_type = detect_problem_type(df[target_col])
    numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns.tolist()
    cat_cols     = df.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
    # remove target from feature lists
    if target_col in cat_cols:
        cat_cols.remove(target_col)
    if target_col in numeric_cols:
        numeric_cols.remove(target_col)

    logger.info(f"Target: '{target_col}'  |  Mode: {problem_type.upper()}")
    logger.info(f"Numeric features : {len(numeric_cols)}")
    logger.info(f"Categorical features: {len(cat_cols)}")
    logger.info(f"Dropped ID cols  : {id_cols}")

    # ── Skewness / kurtosis table ─────────────────────────────────────────
    skew_kurt = pd.DataFrame({
        "skewness": df[numeric_cols].skew(),
        "kurtosis": df[numeric_cols].kurt(),
    }).round(4)
    sk_path = os.path.join(out_dir, "08_skewness_kurtosis.csv")
    skew_kurt.to_csv(sk_path)

    # ── Outlier counts ────────────────────────────────────────────────────
    outlier_counts = {c: iqr_outliers(df[c].dropna()) for c in numeric_cols}
    total_outliers  = sum(outlier_counts.values())

    # ── Generate charts ───────────────────────────────────────────────────
    artifact_paths = []

    p = plot_target_distribution(df, target_col, problem_type, out_dir)
    artifact_paths.append(p)

    p = plot_missing_heatmap(df, out_dir)
    if p: artifact_paths.append(p)

    p = plot_correlation_matrix(df, numeric_cols + [target_col] if pd.api.types.is_numeric_dtype(df[target_col].dtype) else numeric_cols, out_dir)
    if p: artifact_paths.append(p)

    p = plot_numeric_distributions(df, numeric_cols, target_col, out_dir)
    if p: artifact_paths.append(p)

    p = plot_categorical_distributions(df, cat_cols, out_dir)
    if p: artifact_paths.append(p)

    extra = plot_feature_target(df, numeric_cols, cat_cols, target_col, problem_type, out_dir)
    artifact_paths.extend(extra)

    p = plot_outlier_summary(df, numeric_cols, out_dir)
    if p: artifact_paths.append(p)

    artifact_paths.append(sk_path)

    # ── HTML profile (optional) ───────────────────────────────────────────
    html = generate_html_report(df, target_col, out_dir)
    if html:
        artifact_paths.append(html)

    # ── Metrics summary ───────────────────────────────────────────────────
    summary = {
        "stage": "eda",
        "target_column": target_col,
        "problem_type": problem_type,
        "n_rows": len(df),
        "n_numeric_features": len(numeric_cols),
        "n_categorical_features": len(cat_cols),
        "dropped_id_cols": id_cols,
        "total_missing": int(df.isnull().sum().sum()),
        "total_outliers_iqr": total_outliers,
        "high_skew_features": skew_kurt[skew_kurt["skewness"].abs() > 1].index.tolist(),
    }
    if problem_type == "regression":
        summary["target_mean"]   = float(df[target_col].mean())
        summary["target_std"]    = float(df[target_col].std())
        summary["target_median"] = float(df[target_col].median())
        summary["target_skew"]   = float(df[target_col].skew())
    else:
        summary["n_classes"]     = int(df[target_col].nunique())
        summary["class_counts"]  = df[target_col].value_counts().to_dict()

    json_path = os.path.join(out_dir, "eda_metrics.json")
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    artifact_paths.append(json_path)

    # ── MLflow ────────────────────────────────────────────────────────────
    init_mlflow(cfg)
    with mlflow.start_run(run_name="Stage_2_EDA"):
        safe_log_params({
            "target_column":       target_col,
            "problem_type":        problem_type,
            "n_numeric_features":  len(numeric_cols),
            "n_categorical_features": len(cat_cols),
        })
        safe_log_metrics({
            "total_missing":       int(df.isnull().sum().sum()),
            "total_outliers_iqr":  total_outliers,
            "n_high_skew_features": len(summary["high_skew_features"]),
        })
        for p in artifact_paths:
            safe_log_artifact(p, "eda")

    logger.info(f"Stage 2 — EDA COMPLETE ✓  ({len(artifact_paths)} artifacts saved)")
    return summary


if __name__ == "__main__":
    run_automated_eda()
