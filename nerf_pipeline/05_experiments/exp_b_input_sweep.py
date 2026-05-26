"""
실험 B — 입력 이미지 수 Sweep
{10, 20, 30, 45, 60, 90}장으로 각각 학습 → PSNR/학습시간 곡선 출력

사용법:
    python exp_b_input_sweep.py --obj_dir ../data/objects/obj01_matte --method nerf
    python exp_b_input_sweep.py --obj_dir ../data/objects/obj01_matte --method all
"""

import argparse
import json
import random
import shutil
import subprocess
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PIPELINE_ROOT = Path(__file__).parent.parent
TRAIN_SCRIPT = PIPELINE_ROOT / "03_train" / "train.py"
EVAL_SCRIPT = PIPELINE_ROOT / "04_evaluate" / "evaluate.py"
RESULTS_DIR = PIPELINE_ROOT / "results" / "exp_b"

SWEEP_COUNTS = [10, 20, 30, 45, 60, 90]


def create_subset(obj_dir: Path, n: int, seed: int = 42) -> Path:
    """n장 서브셋 폴더 생성"""
    all_imgs = sorted((obj_dir / "images").glob("*.jpg")) + \
               sorted((obj_dir / "images").glob("*.png"))

    if len(all_imgs) < n:
        print(f"[경고] 이미지 {len(all_imgs)}장 < {n}장 요청, 전체 사용")
        n = len(all_imgs)

    random.seed(seed)
    selected = sorted(random.sample(all_imgs, n))

    subset_dir = obj_dir / f"subset_{n:03d}"
    (subset_dir / "images").mkdir(parents=True, exist_ok=True)
    for p in selected:
        shutil.copy2(p, subset_dir / "images" / p.name)

    return subset_dir


def sweep_one_method(obj_dir: Path, method: str) -> list:
    results = []
    for n in SWEEP_COUNTS:
        total_imgs = len(list((obj_dir / "images").glob("*.jpg")) +
                         list((obj_dir / "images").glob("*.png")))
        if n > total_imgs:
            print(f"  [건너뜀] {n}장 > 보유 {total_imgs}장")
            continue

        subset_dir = create_subset(obj_dir, n)
        print(f"\n--- {method.upper()} | {n}장 ---")

        # 전처리
        subprocess.run([sys.executable,
                        str(PIPELINE_ROOT / "02_preprocess" / "run_colmap.py"),
                        "--obj_dir", str(subset_dir)])
        subprocess.run([sys.executable,
                        str(PIPELINE_ROOT / "02_preprocess" / "prepare_dataset.py"),
                        "--obj_dir", str(subset_dir)])

        # 학습
        subprocess.run([sys.executable, str(TRAIN_SCRIPT),
                        "--method", method, "--obj_dir", str(subset_dir)])

        # 평가
        out_csv = RESULTS_DIR / f"{obj_dir.name}_{method}_n{n:03d}.csv"
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        subprocess.run([sys.executable, str(EVAL_SCRIPT),
                        "--obj_dir", str(subset_dir), "--out", str(out_csv)])

        if out_csv.exists():
            import pandas as pd
            df = pd.read_csv(out_csv)
            if not df.empty:
                row = df.iloc[0].to_dict()
                row["n_images"] = n
                results.append(row)

    return results


def plot_sweep(all_results: dict, out_dir: Path):
    """PSNR vs 이미지 수 곡선 + 학습시간 곡선"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    colors = {"nerf": "#4C72B0", "ngp": "#DD8452", "3dgs": "#55A868"}

    for method, results in all_results.items():
        if not results:
            continue
        ns = [r["n_images"] for r in results]
        psnrs = [r["psnr"] for r in results]
        ssims = [r["ssim"] for r in results]
        times = [r["train_min"] for r in results]
        c = colors.get(method, "gray")

        axes[0].plot(ns, psnrs, "o-", color=c, label=method.upper(), linewidth=2)
        axes[1].plot(ns, ssims, "s-", color=c, label=method.upper(), linewidth=2)
        axes[2].plot(ns, times, "^-", color=c, label=method.upper(), linewidth=2)

    axes[0].set_xlabel("입력 이미지 수")
    axes[0].set_ylabel("PSNR (dB) ↑")
    axes[0].set_title("PSNR vs 입력 수")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].set_xlabel("입력 이미지 수")
    axes[1].set_ylabel("SSIM ↑")
    axes[1].set_title("SSIM vs 입력 수")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    axes[2].set_xlabel("입력 이미지 수")
    axes[2].set_ylabel("학습 시간 (분)")
    axes[2].set_title("학습 시간 vs 입력 수")
    axes[2].legend()
    axes[2].grid(alpha=0.3)

    plt.suptitle("실험 B — 입력 이미지 수 Sweep", fontsize=14, fontweight="bold")
    plt.tight_layout()
    out_path = out_dir / "exp_b_sweep.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n그래프 저장: {out_path}")

    # Diminishing return 지점 분석
    print("\n=== Diminishing Return 분석 ===")
    for method, results in all_results.items():
        if len(results) < 3:
            continue
        ns = [r["n_images"] for r in results]
        psnrs = [r["psnr"] for r in results]
        diffs = np.diff(psnrs)
        threshold = 0.2  # dB
        dr_idx = next((i for i, d in enumerate(diffs) if d < threshold), len(diffs))
        if dr_idx < len(ns) - 1:
            print(f"  {method.upper()}: {ns[dr_idx+1]}장부터 한계 효용 (ΔPSNR < {threshold}dB)")
        else:
            print(f"  {method.upper()}: 90장까지 계속 향상됨")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--obj_dir", type=Path, required=True)
    parser.add_argument("--method", choices=["nerf", "ngp", "3dgs", "all"], default="all")
    args = parser.parse_args()

    methods = ["nerf", "ngp", "3dgs"] if args.method == "all" else [args.method]
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    all_results = {}
    for method in methods:
        print(f"\n{'='*60}")
        print(f"실험 B: {method.upper()} — 입력 수 Sweep")
        all_results[method] = sweep_one_method(args.obj_dir, method)

    # 결과 저장
    (RESULTS_DIR / "exp_b_raw.json").write_text(json.dumps(all_results, indent=2))
    plot_sweep(all_results, RESULTS_DIR)


if __name__ == "__main__":
    main()
