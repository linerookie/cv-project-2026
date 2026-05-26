"""
실험 C — 재질별 강건성 분석
5개 재질군 × 3개 방법 → 박스플롯 + 실패 모드 시각화

재질 5군:
  matte      - 무광 매트
  glossy     - 광택
  translucent - 반투명
  complex    - 복잡 텍스처
  fine       - 미세 디테일

사용법:
    python exp_c_materials.py --data_root ../data/objects
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
EVAL_SCRIPT = PIPELINE_ROOT / "04_evaluate" / "evaluate.py"
RESULTS_DIR = PIPELINE_ROOT / "results" / "exp_c"

MATERIAL_GROUPS = {
    "matte":       "무광 매트",
    "glossy":      "광택",
    "translucent": "반투명",
    "complex":     "복잡 텍스처",
    "fine":        "미세 디테일",
}

METHODS = ["nerf", "ngp", "3dgs"]


def detect_material(obj_dir: Path) -> str:
    name = obj_dir.name.lower()
    for key in MATERIAL_GROUPS:
        if key in name:
            return key
    return "unknown"


def collect_results(data_root: Path) -> pd.DataFrame:
    rows = []
    for obj_dir in sorted(data_root.iterdir()):
        if not obj_dir.is_dir():
            continue
        material = detect_material(obj_dir)
        eval_csv = obj_dir / "eval_results.csv"
        if not eval_csv.exists():
            print(f"  [건너뜀] {obj_dir.name}: eval_results.csv 없음")
            continue
        df = pd.read_csv(eval_csv)
        df["material"] = material
        df["obj_name"] = obj_dir.name
        # 방법 추출
        df["method"] = df["exp_name"].apply(
            lambda x: next((m for m in METHODS if m in x), "unknown"))
        rows.append(df)

    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)


def plot_boxplots(df: pd.DataFrame, out_dir: Path):
    """재질 × 방법 박스플롯 (PSNR, SSIM, LPIPS)"""
    materials = [m for m in MATERIAL_GROUPS if m in df["material"].values]
    mat_labels = [MATERIAL_GROUPS[m] for m in materials]
    methods = ["nerf", "ngp", "3dgs"]
    colors = {"nerf": "#4C72B0", "ngp": "#DD8452", "3dgs": "#55A868"}

    metrics = [("psnr", "PSNR (dB) ↑"), ("ssim", "SSIM ↑"), ("lpips", "LPIPS ↓")]

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    for ax, (metric, ylabel) in zip(axes, metrics):
        positions = []
        plot_data = []
        tick_positions = []
        tick_labels = []

        gap = len(methods) + 1.5
        for i, mat in enumerate(materials):
            center = i * gap
            tick_positions.append(center + len(methods) / 2)
            tick_labels.append(mat_labels[i])
            for j, method in enumerate(methods):
                pos = center + j
                positions.append(pos)
                subset = df[(df["material"] == mat) & (df["method"] == method)][metric].dropna()
                plot_data.append(subset.values if len(subset) > 0 else [np.nan])

        bp = ax.boxplot(plot_data, positions=positions, widths=0.6,
                        patch_artist=True, notch=False)
        for patch, pos in zip(bp["boxes"], positions):
            method_idx = int(pos) % len(methods)
            method = methods[method_idx % len(methods)]
            patch.set_facecolor(colors[method])
            patch.set_alpha(0.7)

        ax.set_xticks(tick_positions)
        ax.set_xticklabels(tick_labels, fontsize=9)
        ax.set_ylabel(ylabel)
        ax.set_title(f"{metric.upper()} 재질별 분포")
        ax.grid(axis="y", alpha=0.3)

        # 범례
        from matplotlib.patches import Patch
        legend_elements = [Patch(facecolor=colors[m], alpha=0.7, label=m.upper())
                           for m in methods]
        ax.legend(handles=legend_elements, loc="upper right", fontsize=8)

    plt.suptitle("실험 C — 재질별 강건성 분석", fontsize=14, fontweight="bold")
    plt.tight_layout()
    out_path = out_dir / "exp_c_boxplot.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"박스플롯 저장: {out_path}")


def identify_failure_cases(df: pd.DataFrame, out_dir: Path):
    """광택·반투명 객체에서 PSNR 하위 20% = 실패 사례"""
    hard_materials = ["glossy", "translucent"]
    failures = df[df["material"].isin(hard_materials)].copy()

    if failures.empty:
        return

    threshold_psnr = failures["psnr"].quantile(0.20)
    bad_cases = failures[failures["psnr"] < threshold_psnr]

    print(f"\n=== 실패 사례 (PSNR < {threshold_psnr:.1f}dB) ===")
    for _, row in bad_cases.iterrows():
        print(f"  {row['exp_name']}  재질={row['material']}  "
              f"PSNR={row['psnr']:.2f}  SSIM={row['ssim']:.4f}")

    out_path = out_dir / "failure_cases.json"
    out_path.write_text(bad_cases.to_json(orient="records", indent=2))
    print(f"실패 사례 저장: {out_path}")


def print_material_summary(df: pd.DataFrame):
    print("\n=== 재질별 평균 지표 ===")
    summary = df.groupby(["material", "method"])[["psnr", "ssim", "lpips"]].mean()
    print(summary.to_string(float_format="%.3f"))

    print("\n=== 가장 어려운 재질 (PSNR 기준) ===")
    by_mat = df.groupby("material")["psnr"].mean().sort_values()
    for mat, val in by_mat.items():
        print(f"  {mat:<14} {val:.2f} dB  ({MATERIAL_GROUPS.get(mat, '')})")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=Path,
                        default=PIPELINE_ROOT / "data" / "objects")
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    df = collect_results(args.data_root)

    if df.empty:
        print("[오류] 평가 결과 없음. 먼저 exp_a_methods.py로 모든 객체를 평가하세요.")
        return

    df.to_csv(RESULTS_DIR / "exp_c_all.csv", index=False)
    print(f"전체 데이터: {len(df)}행")

    print_material_summary(df)
    plot_boxplots(df, RESULTS_DIR)
    identify_failure_cases(df, RESULTS_DIR)


if __name__ == "__main__":
    main()
