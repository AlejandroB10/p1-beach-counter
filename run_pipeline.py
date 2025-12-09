#!/usr/bin/env python3
"""Orchestrator to run Pipelines A, B, and C.

Pipeline A: scripts/run_pipeline_a.py (baseline grayscale+Canny)
Pipeline B: scripts/run_pipeline_b.py (teammate watershed/hue subtract)
Pipeline C: scripts/run_pipeline_c.py (zone-aware LAB+Canny)

Usage:
    python run_pipeline.py [--pipelines A,B,C] [--raw-dir data/raw] [--out-dir data/outputs] [--no-roi] [--no-zones]
"""

import argparse
import subprocess
import sys
from pathlib import Path


def run_cmd(cmd, name):
    print(f"\n=== Running {name}: {' '.join(map(str, cmd))}")
    result = subprocess.run(cmd, cwd=Path(__file__).parent)
    if result.returncode != 0:
        print(f"[WARN] {name} exited with code {result.returncode}")
    else:
        print(f"[OK] {name} completed")
    return result.returncode


def main():
    parser = argparse.ArgumentParser(description="Run Pipelines A, B, and C")
    parser.add_argument('--pipelines', default='A,B,C', help='Comma-separated list of pipelines to run')
    parser.add_argument('--raw-dir', default='data/raw', help='Raw images directory (used by Pipeline C)')
    parser.add_argument('--out-dir', default='data/outputs', help='Output directory (used by Pipeline C)')
    parser.add_argument('--no-roi', action='store_true', help='Disable ROI mask for Pipeline C')
    parser.add_argument('--no-zones', action='store_true', help='Disable water/sand zones for Pipeline C')
    args = parser.parse_args()

    pipelines = [p.strip().upper() for p in args.pipelines.split(',') if p.strip()]
    repo_root = Path(__file__).parent

    for p in pipelines:
        if p == 'A':
            cmd = [sys.executable, str(repo_root / "scripts" / "run_pipeline_a.py")]
            run_cmd(cmd, "Pipeline A")
        elif p == 'B':
            cmd = [sys.executable, str(repo_root / "scripts" / "run_pipeline_b.py")]
            run_cmd(cmd, "Pipeline B")
        elif p == 'C':
            cmd = [
                sys.executable,
                str(repo_root / "scripts" / "run_pipeline_c.py"),
            ]
            run_cmd(cmd, "Pipeline C")
        else:
            print(f"[WARN] Unknown pipeline '{p}', skipping.")


if __name__ == "__main__":
    raise SystemExit(main())
