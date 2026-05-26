"""
NeRF / Instant-NGP / 3DGS 학습 통합 래퍼 (nerfstudio 기반)

디바이스:
  - Apple M5 (MPS): nerfacto(NeRF), splatfacto(3DGS) 지원
  - GPU (CUDA):     vanilla-nerf, instant-ngp, splatfacto 모두 지원

사용법:
    python train.py --method nerf    --obj_dir ../data/objects/obj01_matte
    python train.py --method ngp     --obj_dir ../data/objects/obj01_matte
    python train.py --method 3dgs    --obj_dir ../data/objects/obj01_matte
    python train.py --method all     --obj_dir ../data/objects/obj01_matte
    python train.py --method nerf    --obj_dir ../data/objects/obj01_matte --masked
"""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path


METHOD_MAP = {
    "nerf": "nerfacto",
    "ngp":  "instant-ngp",
    "3dgs": "splatfacto",
}

# M5 MPS에서 동작 가능한 방법 (instant-ngp은 CUDA 필요)
MPS_COMPATIBLE = {"nerf", "3dgs"}

TRAIN_STEPS = {
    "nerf": 30000,
    "ngp":  30000,
    "3dgs": 30000,
}


def detect_device() -> str:
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
    except ImportError:
        pass
    return "cpu"


def get_ns_data_dir(obj_dir: Path, masked: bool) -> Path:
    d = obj_dir / ("ns_data_masked" if masked else "ns_data")
    if not d.exists():
        sys.exit(f"[오류] nerfstudio 데이터 없음: {d}\n먼저 prepare_dataset.py를 실행하세요.")
    return d


def train_one(method: str, obj_dir: Path, masked: bool, device: str,
              extra_steps: int = None) -> dict:
    ns_method = METHOD_MAP[method]
    data_dir = get_ns_data_dir(obj_dir, masked)
    suffix = "_masked" if masked else ""
    exp_name = f"{obj_dir.name}_{method}{suffix}"
    out_dir = obj_dir / "outputs" / exp_name

    if device == "mps" and method not in MPS_COMPATIBLE:
        print(f"\n[경고] {method} (instant-ngp) 은 CUDA 전용입니다.")
        print("  → Google Colab 노트북: nerf_pipeline/03_train/colab_train.ipynb 사용")
        return {"method": method, "status": "skipped_no_cuda"}

    steps = extra_steps or TRAIN_STEPS[method]

    cmd = [
        "ns-train", ns_method,
        "--data", str(data_dir),
        "--output-dir", str(out_dir),
        "--experiment-name", exp_name,
        "--max-num-iterations", str(steps),
        "--steps-per-eval-all-images", "5000",
        "--vis", "wandb" if False else "tensorboard",
    ]

    # MPS 디바이스 설정
    if device == "mps":
        cmd += ["--pipeline.device", "mps"]

    print(f"\n[학습 시작] {exp_name}")
    print(f"  방법: {ns_method}  |  디바이스: {device}  |  스텝: {steps}")
    print(f"  데이터: {data_dir}")

    t_start = time.time()
    result = subprocess.run(cmd, capture_output=False, text=True)
    elapsed = time.time() - t_start

    status = "done" if result.returncode == 0 else "error"
    print(f"\n[완료] {exp_name} — {elapsed/60:.1f}분 ({status})")

    return {
        "method": method,
        "ns_method": ns_method,
        "exp_name": exp_name,
        "out_dir": str(out_dir),
        "train_time_sec": elapsed,
        "steps": steps,
        "masked": masked,
        "status": status,
    }


def save_run_log(obj_dir: Path, results: list):
    log_path = obj_dir / "train_log.json"
    log_path.write_text(json.dumps(results, indent=2))
    print(f"\n학습 로그 저장: {log_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", choices=["nerf", "ngp", "3dgs", "all"], required=True)
    parser.add_argument("--obj_dir", type=Path, required=True)
    parser.add_argument("--masked", action="store_true", help="마스킹 데이터 사용 (실험 D)")
    parser.add_argument("--steps", type=int, default=None, help="학습 스텝 수 오버라이드")
    args = parser.parse_args()

    device = detect_device()
    print(f"[디바이스] {device}")

    methods = ["nerf", "ngp", "3dgs"] if args.method == "all" else [args.method]
    results = []
    for m in methods:
        r = train_one(m, args.obj_dir, args.masked, device, args.steps)
        results.append(r)

    save_run_log(args.obj_dir, results)

    print("\n=== 학습 요약 ===")
    for r in results:
        status = r.get("status", "?")
        if status == "done":
            mins = r["train_time_sec"] / 60
            print(f"  {r['method']:6s}: {mins:.1f}분  → {r['out_dir']}")
        elif status == "skipped_no_cuda":
            print(f"  {r['method']:6s}: [건너뜀] CUDA 필요 — Colab 사용")
        else:
            print(f"  {r['method']:6s}: [오류]")


if __name__ == "__main__":
    main()
