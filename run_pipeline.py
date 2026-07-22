"""
run_pipeline.py — Master Automated ML Pipeline Runner
══════════════════════════════════════════════════════
Usage:
    python run_pipeline.py                     # Run all stages
    python run_pipeline.py --start eda         # Resume from EDA
    python run_pipeline.py --only preprocessing tuning  # Run specific stages

Stages (in order):
    data_ingestion → eda → preprocessing → tuning → evaluation
"""

import subprocess
import sys
import os
import time
import argparse
from datetime import datetime

# Ensure UTF-8 output on all platforms
os.environ.setdefault("PYTHONIOENCODING", "utf-8")
os.environ.setdefault("MLFLOW_SILENT", "true")

import io
if hasattr(sys.stdout, 'buffer'):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
if hasattr(sys.stderr, 'buffer'):
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

STAGES = [
    "data_ingestion",
    "eda",
    "preprocessing",
    "tuning",
    "evaluation",
    "optimize_artifacts"
]

STAGE_DESCRIPTIONS = {
    "data_ingestion": "Load raw CSV, validate, stratified train/test split",
    "eda":            "Comprehensive exploratory data analysis + charts",
    "preprocessing":  "Feature engineering, encoding, scaling, outlier handling",
    "tuning":         "AutoML tournament — RandomizedSearch across model pool",
    "evaluation":     "Full metrics, charts, SHAP, save deployment .pkl",
    "optimize_artifacts": "Clone deployment model (Way 3) and delete original to save space",
}

STAGE_SCRIPT_OVERRIDES = {
    "optimize_artifacts": "optimize_artifacts.py",
}

def banner(title: str):
    width = 62
    print("\n" + "=" * width)
    print(f"  {title}")
    print("=" * width)


def run_stage(stage_name: str) -> bool:
    if stage_name in STAGE_SCRIPT_OVERRIDES:
        script = STAGE_SCRIPT_OVERRIDES[stage_name]
    else:
        script = os.path.join("src", f"{stage_name}.py")
    banner(f"STAGE: {stage_name.upper()}  |  {STAGE_DESCRIPTIONS[stage_name]}")
    print(f"  Started: {datetime.now().strftime('%H:%M:%S')}\n")

    t0 = time.time()
    env = {**os.environ, "PYTHONIOENCODING": "utf-8"}

    proc = subprocess.Popen(
        [sys.executable, script],
        cwd=os.path.dirname(os.path.abspath(__file__)),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        env=env,
        bufsize=1,
    )

    for line in proc.stdout:
        print(line, end="", flush=True)
    proc.wait()

    elapsed = time.time() - t0
    success = proc.returncode == 0

    status = "PASSED" if success else "FAILED"
    print(f"\n  {status}  —  {stage_name}  ({elapsed:.1f}s)")
    return success


def parse_args():
    args = argparse.ArgumentParser(description="Automated ML Pipeline")
    args.add_argument("--start", type=str, default=None,
                        help="Resume from this stage (skips earlier stages)")
    args.add_argument("--only", nargs="+", default=None,
                        help="Run only these specific stages")
    return args.parse_args()


def main():
    args = parse_args()

    if args.only:
        stages = [s for s in args.only if s in STAGES]
        if not stages:
            print(f"Unknown stages: {args.only}. Valid: {STAGES}")
            sys.exit(1)
    elif args.start:
        if args.start not in STAGES:
            print(f"Unknown stage '{args.start}'. Valid: {STAGES}")
            sys.exit(1)
        stages = STAGES[STAGES.index(args.start):]
    else:
        stages = STAGES

    banner(f"AUTOMATED ML PIPELINE  |  {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print(f"  Stages to run: {' -> '.join(stages)}\n")

    results = {}
    total_start = time.time()

    for stage in stages:
        ok = run_stage(stage)
        results[stage] = ok
        if not ok:
            banner("PIPELINE HALTED")
            print(f"  Stage '{stage}' failed. Fix the error and re-run with:")
            print(f"    python run_pipeline.py --start {stage}\n")
            _print_summary(results, time.time() - total_start)
            sys.exit(1)

    banner("ALL STAGES COMPLETE")
    _print_summary(results, time.time() - total_start)


def _print_summary(results: dict, elapsed: float):
    print("\n  Stage Summary:")
    for stage, ok in results.items():
        icon = "[OK]" if ok else "[FAIL]"
        print(f"    {icon}  {stage}")
    print(f"\n  Total time: {elapsed:.1f}s")
    print(f"  Artifacts: artifacts/")
    print(f"  Models   : artifacts/models/deployment_model.pkl\n")


if __name__ == "__main__":
    main()
