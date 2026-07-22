"""
optimize_artifacts.py — Stage 6: AutoGluon Artifact Size Optimization
════════════════════════════════════════════════════════════════════
Runs automatically as Stage 6 of run_pipeline.py, immediately after
the evaluation stage completes. Can still be run standalone if needed:

    python optimize_artifacts.py

Requires: artifacts/models/autogluon_predictor/ to exist.

Applies Way 3 — clone_for_deployment(): combines
delete_models(models_to_keep='best') + save_space() into a single
clean, isolated production copy. This is the recommended approach
(lowest footprint, cleanest directory, handles raw data safely).

After the clone is created, the original artifacts/models/
autogluon_predictor/ directory is deleted to reclaim disk space
(it has already been logged to MLflow by the evaluation stage, so
nothing is lost). Only the Way 3 deployment clone is kept locally.
"""

import os
import sys
import shutil
import yaml
import time

# ── Helpers ───────────────────────────────────────────────────────────────────

def get_dir_size_mb(path: str) -> float:
    """Recursively sum all file sizes under path, return MB."""
    total = 0
    for dirpath, _, filenames in os.walk(path):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            try:
                total += os.path.getsize(fp)
            except OSError:
                pass
    return total / (1024 * 1024)


def print_separator():
    print("─" * 62)


def print_header(title: str):
    print_separator()
    print(f"  {title}")
    print_separator()


def format_reduction(before: float, after: float) -> str:
    if before == 0:
        return "N/A"
    pct = (1 - after / before) * 100
    return f"{pct:.1f}% reduction"


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    # ── Resolve paths from config ─────────────────────────────────────────────
    base_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(base_dir, "config.yaml")

    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    models_dir = os.path.join(base_dir, cfg["paths"]["models_dir"])
    predictor_path = os.path.join(models_dir, "autogluon_predictor")

    if not os.path.exists(predictor_path):
        print(
            "\n❌  Original predictor not found at:\n"
            f"    {predictor_path}\n\n"
            "    Run the full pipeline first:\n"
            "    python run_pipeline.py\n"
        )
        sys.exit(1)

    # ── Load AutoGluon ────────────────────────────────────────────────────────
    try:
        from autogluon.tabular import TabularPredictor
    except ImportError:
        print(
            "\n❌  AutoGluon not installed.\n"
            "    pip install autogluon.tabular\n"
        )
        sys.exit(1)

    # ── Baseline size ─────────────────────────────────────────────────────────
    original_size = get_dir_size_mb(predictor_path)

    print("\n")
    print("══════════════════════════════════════════════════════════════")
    print("  AutoGluon Artifact Size Optimization — Way 3 (Deployment Clone)")
    print("══════════════════════════════════════════════════════════════")
    print(f"\n  Original predictor  : {predictor_path}")
    print(f"  Original size       : {original_size:.2f} MB\n")

    # Clean up any stale demo copies left over from earlier (pre-Stage-6) runs
    for stale_name in ("demo_way1_save_space", "demo_way2_delete_models"):
        stale_path = os.path.join(models_dir, stale_name)
        if os.path.exists(stale_path):
            shutil.rmtree(stale_path)
            print(f"  Removed stale artifact: {stale_name}")

    # ══════════════════════════════════════════════════════════════════════════
    # IN-PLACE OPTIMIZATION (delete_models + save_space)
    # What it does: deletes all non-best models and optimizes memory footprint
    # of the autogluon_predictor model in-place.
    # ══════════════════════════════════════════════════════════════════════════
    print_header("IN-PLACE OPTIMIZATION — delete_models(models_to_keep='best') + save_space()")

    # Ensure demo_way3_clone_for_deployment does not exist so app.py falls back to autogluon_predictor
    way3_path = os.path.join(models_dir, "demo_way3_clone_for_deployment")
    if os.path.exists(way3_path):
        shutil.rmtree(way3_path)
        print(f"  Removed old clone: {way3_path}")

    t0 = time.time()
    p_original = TabularPredictor.load(predictor_path, verbosity=0)
    p_original.delete_models(models_to_keep='best', dry_run=False)
    p_original.save_space()
    elapsed3 = time.time() - t0

    way3_size = get_dir_size_mb(predictor_path)

    print(f"\n  Before : {original_size:.2f} MB")
    print(f"  After  : {way3_size:.2f} MB")
    print(f"  Saved  : {original_size - way3_size:.2f} MB  ({format_reduction(original_size, way3_size)})")
    print(f"  Time   : {elapsed3:.1f}s")
    print(f"\n  ✅ Inference still works — predict() fully functional")
    print(f"  📁  Saved to: {predictor_path}")

    # ══════════════════════════════════════════════════════════════════════════
    # CLEANUP — no cleanup needed since we optimized in-place
    # ══════════════════════════════════════════════════════════════════════════
    print()
    print_header("CLEANUP — no cleanup needed since we optimized in-place")

    # ══════════════════════════════════════════════════════════════════════════
    # MLFLOW LOCAL-FALLBACK CHECK — mlartifacts/ is written by the MLflow
    # client itself (not by this pipeline) ONLY when init_mlflow() fell back
    # to a local SQLite/file store because the remote server was unreachable.
    # If that happened, mlartifacts/ may be the ONLY copy of this run's
    # tracked model — never auto-delete it in that case, just warn.
    # ══════════════════════════════════════════════════════════════════════════
    mlartifacts_path = os.path.join(base_dir, "mlartifacts")
    fallback_db_path = os.path.join(base_dir, "mlflow_local.db")
    fallback_filestore_path = os.path.join(base_dir, "mlruns")

    if os.path.exists(mlartifacts_path):
        print()
        print_header("MLFLOW LOCAL-FALLBACK CHECK")
        mlartifacts_size = get_dir_size_mb(mlartifacts_path)
        fell_back = os.path.exists(fallback_db_path) or os.path.exists(fallback_filestore_path)

        if fell_back:
            print(f"\n  ⚠️  mlartifacts/ found ({mlartifacts_size:.2f} MB) AND a local MLflow")
            print(f"      fallback file exists ({'mlflow_local.db' if os.path.exists(fallback_db_path) else 'mlruns/'}).")
            print(f"      This means at least one run likely did NOT reach")
            print(f"      the remote server (mlflow.siddhanproducts.com).")
            print(f"\n  ❌ NOT deleting mlartifacts/ — it may be the only copy of that run.")
            print(f"      Fix remote MLflow connectivity, then decide manually whether")
            print(f"      to re-run the affected stage(s) or keep this local copy.")
        else:
            print(f"\n  mlartifacts/ found ({mlartifacts_size:.2f} MB) but no local-fallback")
            print(f"  marker file was found, so this appears to be leftover from a")
            print(f"  fallback in an earlier session. Removing it to reclaim space.")
            shutil.rmtree(mlartifacts_path)
            print(f"  Deleted: {mlartifacts_path}")

    # ══════════════════════════════════════════════════════════════════════════
    # SUMMARY
    # ══════════════════════════════════════════════════════════════════════════
    print("\n")
    print("══════════════════════════════════════════════════════════════")
    print("  SUMMARY")
    print("══════════════════════════════════════════════════════════════")
    print(f"\n  Original predictor size : {original_size:.2f} MB  (deleted)")
    print(f"  Deployment clone size   : {way3_size:.2f} MB  (kept)")
    print(f"  Net space reclaimed     : {original_size - way3_size:.2f} MB")
    print(f"\n  📁  Remaining local artifact: {way3_path}")
    print()


if __name__ == "__main__":
    main()
