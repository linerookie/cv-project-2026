"""
실험 D — 배경 마스킹 Ablation
마스킹 O vs X → PSNR/SSIM/LPIPS 차이 측정 (동일 조건)

사용법:
    python exp_d_masking.py --obj_dir ../data/objects/obj01_matte --method nerf
    python exp_d_masking.py --obj_dir ../data/objects/obj01_matte --method all
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
MASK_SCRIPT = PIPELINE_ROOT / "02_preprocess" / "mask_background.py"
PREP_SCRIPT = PIPELINE_ROOT / "02_preprocess" / "prepare_dataset.py"
TRAIN_SCRIPT = PIPELINE_ROOT / "03_train" / "train.py"
EVAL_SCRIPT = PIPELINE_ROOT / "04_evaluate" / "evaluate.py"
RESULTS_DIR = PIPELINE_ROOT / "results" / "exp_d"

METHODS = ["nerf", "ngp", "3dgs"]


def run_ablation(obj_dir: Path, method: str) -> dict:
    print(f"\n[실험 D] {obj_dir.name} | {method.upper()}")

    # --- 마스킹 없이 ---
    print("\n  [1/2] 마스킹 없이 학습...")
    subprocess.run([sys.executable, str(TRAIN_SCRIPT),
                    "--method", method, "--obj_dir", str(obj_dir)])

    out_csv_raw = RESULTS_DIR / f"{obj_dir.name}_{method}_raw.csv"
    subprocess.run([sys.executable, str(EVAL_SCRIPT),
                    "--obj_dir", str(obj_dir), "--out", str(out_csv_raw)])

    # --- 배경 마스킹 ---
    print("\n  [2/2] 배경 마스킹 후 학습...")
    # 마스킹 수행
    subprocess.run([sys.executable, str(MASK_SCRIPT),
                    "--obj_dir", str(obj_dir)])
    # 마스킹 데이터 준비
    subprocess.run([sys.executable, str(PREP_SCRIPT),
                    "--obj_dir", str(obj_dir), "--masked"])
    # 마스킹으로 학습
    subprocess.run([sys.executable, str(TRAIN_SCRIPT),
                    "--method", method, "--obj_dir", str(obj_dir), "--masked"])

    out_csv_masked = RESULTS_DIR / f"{obj_dir.name}_{method}_masked.csv"
    subprocess.run([sys.executable, str(EVAL_SCRIPT),
                    "--obj_dir", str(obj_dir), "--out", str(out_csv_masked)])

    # 결과 비교
    result = {"method": method, "obj": obj_dir.name}
    for label, csv_path in [("raw", out_csv_raw), ("masked", out_csv_masked)]:
        if csv_path.exists():
            df = pd.read_csv(csv_path)
            if not df.empty:
                row = df.iloc[0]
                result[f"{label}_psnr"] = row.get("psnr", np.nan)
                result[f"{label}_ssim"] = row.get("ssim", np.nan)
                result[f"{label}_lpips"] = row.get("lpips", np.nan)

    if "raw_psnr" in result and "masked_psnr" in result:
        result["delta_psnr"] = result["masked_psnr"] - result["raw_psnr"]
        result["delta_ssim"] = result["masked_ssim"] - result["raw_ssim"]
        result["delta_lpips"] = result["raw_lpips"] - result["masked_lpips"]
        print(f"\n  결과: ΔPSNR={result['delta_psnr']:+.2f}dB  "
              f"ΔSSIM={result['delta_ssim']:+.4f}  "
              f"ΔLPIPS={result['delta_lpips']:+.4f}")

    return result


def plot_ablation(results: list, out_dir: Path):
    if not results:
        return

    df = pd.DataFrame(results)
    metrics = [
        ("delta_psnr", "ΔPSNR (dB)  마스킹이 높을수록 ↑"),
        ("delta_ssim", "ΔSSIM  마스킹이 높을수록 ↑"),
        ("delta_lpips", "ΔLPIPS (개선량)  마스킹이 높을수록 ↑"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    colors = {"nerf": "#4C72B0", "ngp": "#DD8452", "3dgs": "#55A868"}

    for ax, (metric, label) in zip(axes, metrics):
        if metric not in df.columns:
            continue
        for method in METHODS:
            subset = df[df["method"] == method][metric].dropna()
            if subset.empty:
                continue
            mean_val = subset.mean()
            ax.bar([method.upper()], [mean_val],
                   color=colors.get(method, "gray"), alpha=0.8)
            ax.text(METHODS.index(method), mean_val + 0.002,
                    f"{mean_val:+.3f}", ha="center", fontsize=9)

        ax.axhline(0, color="red", linestyle="--", alpha=0.5)
        ax.set_ylabel(label)
        ax.set_title(f"마스킹 효과 ({metric})")
        ax.grid(axis="y", alpha=0.3)

    plt.suptitle("실험 D — 배경 마스킹 Ablation", fontsize=14, fontweight="bold")
    plt.tight_layout()
    out_path = out_dir / "exp_d_masking_effect.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n그래프 저장: {out_path}")

    # 마스킹이 유의미한지 통계 요약
    print("\n=== 마스킹 효과 요약 ===")
    for _, metric_label in metrics[:1]:
        mean_delta = df["delta_psnr"].mean()
        print(f"  평균 ΔPSNR = {mean_delta:+.2f}dB  "
              f"({'마스킹 효과 있음 ✓' if mean_delta > 0.3 else '효과 미미'})")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--obj_dir", type=Path, required=True)
    parser.add_argument("--method", choices=["nerf", "ngp", "3dgs", "all"], default="nerf")
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    methods = METHODS if args.method == "all" else [args.method]

    results = []
    for method in methods:
        r = run_ablation(args.obj_dir, method)
        results.append(r)

    (RESULTS_DIR / f"{args.obj_dir.name}_exp_d.json").write_text(
        json.dumps(results, indent=2))

    plot_ablation(results, RESULTS_DIR)


if __name__ == "__main__":
    main()
