"""
정량 평가 — PSNR / SSIM / LPIPS + 렌더링 FPS + 모델 크기

nerfstudio 학습 완료 후 outputs/ 폴더를 읽어 6개 지표를 CSV로 출력

사용법:
    python evaluate.py --obj_dir ../data/objects/obj01_matte
    python evaluate.py --obj_dir ../data/objects/obj01_matte --render  # 새 렌더링 수행
"""

import argparse
import csv
import json
import os
import subprocess
import time
from pathlib import Path

import cv2
import lpips
import numpy as np
import torch
from skimage.metrics import peak_signal_noise_ratio as psnr_fn
from skimage.metrics import structural_similarity as ssim_fn


def load_images_from_dir(d: Path, ext=("*.png", "*.jpg")):
    imgs = []
    for e in ext:
        imgs.extend(sorted(d.glob(e)))
    return imgs


def compute_psnr(gt: np.ndarray, pred: np.ndarray) -> float:
    return psnr_fn(gt, pred, data_range=255)


def compute_ssim(gt: np.ndarray, pred: np.ndarray) -> float:
    return ssim_fn(gt, pred, channel_axis=2, data_range=255)


def compute_lpips_score(gt_t: torch.Tensor, pred_t: torch.Tensor, loss_fn) -> float:
    with torch.no_grad():
        d = loss_fn(gt_t, pred_t)
    return float(d.mean())


def img_to_tensor(img: np.ndarray) -> torch.Tensor:
    t = torch.from_numpy(img).permute(2, 0, 1).float() / 127.5 - 1.0
    return t.unsqueeze(0)


def measure_fps(render_dir: Path, n_frames: int = 30) -> float:
    imgs = load_images_from_dir(render_dir)[:n_frames]
    if not imgs:
        return 0.0
    t0 = time.time()
    for p in imgs:
        cv2.imread(str(p))
    elapsed = time.time() - t0
    return len(imgs) / elapsed


def get_model_size_mb(ckpt_dir: Path) -> float:
    total = 0
    for ext in ["*.ckpt", "*.pt", "*.pth", "*.splat", "*.ply"]:
        for f in ckpt_dir.rglob(ext):
            total += f.stat().st_size
    return total / 1024 / 1024


def find_latest_config(exp_dir: Path):
    """nerfstudio 중첩 출력 구조에서 최신 config.yml 탐색"""
    configs = sorted(exp_dir.rglob("config.yml"))
    return configs[-1] if configs else None


def ns_render(exp_dir: Path, out_dir: Path):
    """nerfstudio로 test set 렌더링"""
    config = find_latest_config(exp_dir)
    if config is None:
        return False

    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ns-render", "dataset",
        "--load-config", str(config),
        "--rendered-output-names", "rgb",
        "--output-path", str(out_dir),
        "--split", "test",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    return result.returncode == 0


def evaluate_one(exp_dir: Path, test_images_dir: Path, loss_fn) -> dict:
    exp_name = exp_dir.name
    render_dir = exp_dir / "renders"
    do_render = not (render_dir.exists() and any(render_dir.iterdir()))

    if do_render:
        print(f"  렌더링 중: {exp_name}")
        ns_render(exp_dir, render_dir)

    # ns-render는 {out_dir}/test/rgb/ 에 저장
    render_rgb_dir = render_dir / "test" / "rgb"
    pred_dir = render_rgb_dir if render_rgb_dir.exists() else render_dir

    gt_imgs = load_images_from_dir(test_images_dir)
    pred_imgs = load_images_from_dir(pred_dir)

    if not gt_imgs or not pred_imgs:
        print(f"  [경고] {exp_name}: GT={len(gt_imgs)}, Pred={len(pred_imgs)} — 건너뜀")
        print(f"  pred_dir: {pred_dir}")
        return None

    n = min(len(gt_imgs), len(pred_imgs))
    psnr_list, ssim_list, lpips_list = [], [], []

    for i in range(n):
        gt = cv2.imread(str(gt_imgs[i]))
        pred = cv2.imread(str(pred_imgs[i]))
        if gt is None or pred is None:
            continue
        pred = cv2.resize(pred, (gt.shape[1], gt.shape[0]))
        gt_rgb = cv2.cvtColor(gt, cv2.COLOR_BGR2RGB)
        pred_rgb = cv2.cvtColor(pred, cv2.COLOR_BGR2RGB)

        psnr_list.append(compute_psnr(gt_rgb, pred_rgb))
        ssim_list.append(compute_ssim(gt_rgb, pred_rgb))

        gt_t = img_to_tensor(gt_rgb)
        pred_t = img_to_tensor(pred_rgb)
        lpips_list.append(compute_lpips_score(gt_t, pred_t, loss_fn))

    latest_config = find_latest_config(exp_dir)
    ckpt_dir = latest_config.parent / "nerfstudio_models" if latest_config else exp_dir
    model_mb = get_model_size_mb(ckpt_dir) if ckpt_dir.exists() else 0.0

    train_log = exp_dir / ".." / ".." / "train_log.json"
    train_sec = 0.0
    if train_log.exists():
        logs = json.loads(train_log.read_text())
        for log in logs:
            if log.get("exp_name", "") in str(exp_dir):
                train_sec = log.get("train_time_sec", 0.0)

    fps = measure_fps(pred_dir)

    return {
        "exp_name": exp_name,
        "n_frames": n,
        "psnr": float(np.mean(psnr_list)),
        "ssim": float(np.mean(ssim_list)),
        "lpips": float(np.mean(lpips_list)),
        "train_min": round(train_sec / 60, 2),
        "render_fps": round(fps, 2),
        "model_mb": round(model_mb, 2),
    }


def save_results(results: list, out_csv: Path):
    if not results:
        return
    keys = list(results[0].keys())
    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(results)
    print(f"\n결과 저장: {out_csv}")


def print_table(results: list):
    print("\n" + "="*80)
    print(f"{'실험명':<30} {'PSNR':>7} {'SSIM':>7} {'LPIPS':>7} {'학습(분)':>9} {'FPS':>6} {'MB':>7}")
    print("-"*80)
    for r in results:
        print(f"{r['exp_name']:<30} {r['psnr']:>7.2f} {r['ssim']:>7.4f} {r['lpips']:>7.4f} "
              f"{r['train_min']:>9.1f} {r['render_fps']:>6.1f} {r['model_mb']:>7.1f}")
    print("="*80)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--obj_dir", type=Path, required=True)
    parser.add_argument("--render", action="store_true", help="강제 재렌더링")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    outputs_dir = args.obj_dir / "outputs"
    test_dir = args.obj_dir / "ns_data" / "images" / "test"
    # fallback: images 전체에서 test split 사용
    if not test_dir.exists():
        test_dir = args.obj_dir / "images"

    print("[LPIPS 모델 로딩...]")
    loss_fn = lpips.LPIPS(net="alex", verbose=False)

    results = []
    for exp_dir in sorted(outputs_dir.iterdir()):
        if not exp_dir.is_dir():
            continue
        print(f"\n평가: {exp_dir.name}")
        r = evaluate_one(exp_dir, test_dir, loss_fn)
        if r:
            results.append(r)

    if not results:
        print("[경고] 평가 가능한 실험 없음. 먼저 train.py를 실행하세요.")
        return

    print_table(results)

    out_csv = args.out or (args.obj_dir / "eval_results.csv")
    save_results(results, out_csv)


if __name__ == "__main__":
    main()
