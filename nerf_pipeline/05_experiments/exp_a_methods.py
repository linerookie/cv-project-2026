"""
실험 A — 방법론 비교 (NeRF / Instant-NGP / 3DGS)
동일 객체에 3가지 방법 학습 → 6개 지표 비교표 생성

사용법:
    python exp_a_methods.py --obj_dir ../data/objects/obj01_matte
    python exp_a_methods.py --all_objects  # 모든 객체 순차 실행
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PIPELINE_ROOT = Path(__file__).parent.parent
TRAIN_SCRIPT = PIPELINE_ROOT / "03_train" / "train.py"
EVAL_SCRIPT = PIPELINE_ROOT / "04_evaluate" / "evaluate.py"
RESULTS_DIR = PIPELINE_ROOT / "results" / "exp_a"


def run_for_object(obj_dir: Path):
    print(f"\n{'='*60}")
    print(f"실험 A: {obj_dir.name}")
    print(f"{'='*60}")

    # 3가지 방법 순차 학습
    for method in ["nerf", "ngp", "3dgs"]:
        cmd = [sys.executable, str(TRAIN_SCRIPT),
               "--method", method, "--obj_dir", str(obj_dir)]
        subprocess.run(cmd)

    # 평가
    out_csv = RESULTS_DIR / f"{obj_dir.name}_exp_a.csv"
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    subprocess.run([sys.executable, str(EVAL_SCRIPT),
                    "--obj_dir", str(obj_dir), "--out", str(out_csv)])
    return out_csv


def plot_comparison(csv_files: list, out_dir: Path):
    """방법별 지표 막대그래프 (논문 Table 1 대응)"""
    dfs = []
    for f in csv_files:
        df = pd.read_csv(f)
        dfs.append(df)
    combined = pd.concat(dfs)

    # 방법명 추출 (exp_name에서)
    def extract_method(name):
        for m in ["nerf", "ngp", "3dgs"]:
            if m in name:
                return m
        return name
    combined["method"] = combined["exp_name"].apply(extract_method)

    # 방법별 평균
    grouped = combined.groupby("method")[["psnr", "ssim", "lpips", "train_min", "render_fps", "model_mb"]].mean()

    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    metrics = [
        ("psnr", "PSNR (dB) ↑", True),
        ("ssim", "SSIM ↑", True),
        ("lpips", "LPIPS ↓", False),
        ("train_min", "학습 시간 (분) ↓", False),
        ("render_fps", "렌더링 FPS ↑", True),
        ("model_mb", "모델 크기 (MB) ↓", False),
    ]
    colors = {"nerf": "#4C72B0", "ngp": "#DD8452", "3dgs": "#55A868"}

    for ax, (metric, label, higher_better) in zip(axes.flat, metrics):
        vals = grouped[metric]
        bars = ax.bar(vals.index,
                      vals.values,
                      color=[colors.get(m, "gray") for m in vals.index])
        ax.set_title(label, fontsize=12)
        ax.set_xlabel("방법")
        for bar, v in zip(bars, vals.values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() * 1.01,
                    f"{v:.2f}", ha="center", va="bottom", fontsize=9)

    plt.suptitle("실험 A — 방법론 비교", fontsize=14, fontweight="bold")
    plt.tight_layout()
    out_path = out_dir / "exp_a_comparison.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n그래프 저장: {out_path}")

    # LaTeX 테이블 출력
    print("\n=== LaTeX Table (논문용) ===")
    print(grouped.to_latex(float_format="%.2f"))

    return grouped


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--obj_dir", type=Path, help="단일 객체 폴더")
    parser.add_argument("--all_objects", action="store_true", help="모든 객체 실행")
    args = parser.parse_args()

    data_root = PIPELINE_ROOT / "data" / "objects"
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    if args.all_objects:
        obj_dirs = [d for d in data_root.iterdir() if d.is_dir()]
    elif args.obj_dir:
        obj_dirs = [args.obj_dir]
    else:
        parser.error("--obj_dir 또는 --all_objects 필요")

    csv_files = []
    for obj_dir in obj_dirs:
        csv_f = run_for_object(obj_dir)
        if csv_f.exists():
            csv_files.append(csv_f)

    if csv_files:
        plot_comparison(csv_files, RESULTS_DIR)


if __name__ == "__main__":
    main()
