"""
전체 파이프라인 실행기 — 객체 하나를 처음부터 끝까지 처리

단계:
  1. COLMAP SfM
  2. nerfstudio 데이터 변환
  3. 배경 마스킹 (선택)
  4. 학습 (nerf / ngp / 3dgs)
  5. 평가

사용법:
    python run_pipeline.py --obj_dir data/objects/obj01_matte --methods all
    python run_pipeline.py --obj_dir data/objects/obj02_glossy --methods nerf 3dgs --skip_colmap
"""

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).parent

def step(name: str, cmd: list, skip: bool = False):
    if skip:
        print(f"\n[건너뜀] {name}")
        return
    print(f"\n{'='*60}")
    print(f"[단계] {name}")
    print(f"{'='*60}")
    result = subprocess.run(cmd)
    if result.returncode != 0:
        print(f"[오류] {name} 실패. 계속 진행합니다.")


def main():
    parser = argparse.ArgumentParser(description="NeRF 비교 연구 전체 파이프라인")
    parser.add_argument("--obj_dir", type=Path, required=True, help="객체 폴더 (images/ 포함)")
    parser.add_argument("--methods", nargs="+",
                        choices=["nerf", "ngp", "3dgs", "all"], default=["nerf", "3dgs"])
    parser.add_argument("--camera_params", type=Path, default=None,
                        help="calibrate.py 출력 JSON")
    parser.add_argument("--skip_colmap", action="store_true")
    parser.add_argument("--skip_mask", action="store_true", help="마스킹 건너뜀")
    parser.add_argument("--masked_train", action="store_true", help="마스킹 데이터로 학습")
    args = parser.parse_args()

    py = sys.executable
    obj = args.obj_dir.resolve()

    print(f"\n{'='*60}")
    print(f"파이프라인 시작: {obj.name}")
    print(f"방법: {args.methods}  |  마스킹: {not args.skip_mask}")
    print(f"{'='*60}")

    # 1. COLMAP
    colmap_cmd = [py, str(ROOT / "02_preprocess" / "run_colmap.py"),
                  "--obj_dir", str(obj), "--gpu"]
    if args.camera_params:
        colmap_cmd += ["--camera_params", str(args.camera_params)]
    step("COLMAP SfM", colmap_cmd, skip=args.skip_colmap)

    # 2. 데이터 변환 (원본)
    step("nerfstudio 변환 (원본)",
         [py, str(ROOT / "02_preprocess" / "prepare_dataset.py"),
          "--obj_dir", str(obj)])

    # 3. 배경 마스킹
    step("배경 마스킹 (rembg)",
         [py, str(ROOT / "02_preprocess" / "mask_background.py"),
          "--obj_dir", str(obj)],
         skip=args.skip_mask)

    if not args.skip_mask:
        step("nerfstudio 변환 (마스킹)",
             [py, str(ROOT / "02_preprocess" / "prepare_dataset.py"),
              "--obj_dir", str(obj), "--masked"])

    # 4. 학습
    methods = ["nerf", "ngp", "3dgs"] if "all" in args.methods else args.methods
    for method in methods:
        train_cmd = [py, str(ROOT / "03_train" / "train.py"),
                     "--method", method, "--obj_dir", str(obj)]
        if args.masked_train and not args.skip_mask:
            train_cmd.append("--masked")
        step(f"학습 ({method.upper()})", train_cmd)

    # 5. 평가
    step("정량 평가 (PSNR/SSIM/LPIPS/FPS/MB)",
         [py, str(ROOT / "04_evaluate" / "evaluate.py"),
          "--obj_dir", str(obj)])

    print(f"\n{'='*60}")
    print(f"[완료] {obj.name}")
    print(f"결과: {obj / 'eval_results.csv'}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
