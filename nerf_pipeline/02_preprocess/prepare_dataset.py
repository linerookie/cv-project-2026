"""
COLMAP 결과 → nerfstudio transforms.json 변환 + train/test split
nerfstudio의 ns-process-data 래퍼 (공식 권장 경로)

사용법:
    python prepare_dataset.py --obj_dir ../data/objects/obj01_matte
    python prepare_dataset.py --obj_dir ../data/objects/obj01_matte --masked  # 마스킹 버전
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path


def ns_process(obj_dir: Path, use_masked: bool = False):
    """ns-process-data colmap: COLMAP 결과를 nerfstudio 형식으로 변환"""
    images_dir = obj_dir / ("images_masked" if use_masked else "images")
    colmap_dir = obj_dir / "sparse" / "0"
    out_dir = obj_dir / ("ns_data_masked" if use_masked else "ns_data")

    if not colmap_dir.exists():
        sys.exit(f"[오류] COLMAP 결과 없음: {colmap_dir}  먼저 run_colmap.py를 실행하세요.")

    if use_masked and not images_dir.exists():
        sys.exit("[오류] images_masked/ 없음  먼저 mask_background.py를 실행하세요.")

    print(f"[변환] COLMAP → nerfstudio transforms.json")
    print(f"  이미지: {images_dir}")
    print(f"  COLMAP: {colmap_dir}")
    print(f"  출력:   {out_dir}")

    cmd = [
        "ns-process-data", "colmap",
        "--data", str(obj_dir),
        "--output-dir", str(out_dir),
        "--colmap-model-path", str(colmap_dir),
        "--images-per-equirect", "0",
    ]

    if use_masked:
        cmd += ["--images", str(images_dir)]

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        # 직접 변환 시도
        print("[경고] ns-process-data 실패, 직접 변환 시도...")
        direct_convert(obj_dir, images_dir, colmap_dir, out_dir)
        return out_dir

    print("[완료] transforms.json 생성")
    _print_split_stats(out_dir)
    return out_dir


def direct_convert(obj_dir: Path, images_dir: Path, colmap_dir: Path, out_dir: Path):
    """COLMAP TXT → transforms.json 직접 변환 (nerfstudio 호환 형식)"""
    import numpy as np

    cameras_txt = colmap_dir / "cameras.txt"
    images_txt = colmap_dir / "images.txt"

    if not cameras_txt.exists():
        sys.exit("[오류] cameras.txt 없음. 먼저 run_colmap.py --out_txt를 실행하세요.")

    # 카메라 내부 파라미터 파싱
    cam_params = {}
    for line in cameras_txt.read_text().splitlines():
        if line.startswith("#") or not line.strip():
            continue
        parts = line.split()
        cam_id = int(parts[0])
        model = parts[1]
        w, h = int(parts[2]), int(parts[3])
        params = [float(p) for p in parts[4:]]
        if model == "PINHOLE":
            fx, fy, cx, cy = params
            k1 = k2 = 0.0
        elif model == "OPENCV":
            fx, fy, cx, cy, k1, k2 = params[:6]
        else:
            fx = fy = float(params[0])
            cx, cy = w / 2, h / 2
            k1 = k2 = 0.0
        cam_params[cam_id] = {"w": w, "h": h, "fl_x": fx, "fl_y": fy,
                               "cx": cx, "cy": cy, "k1": k1, "k2": k2}

    # 이미지 외부 파라미터 파싱
    frames = []
    lines = [l for l in images_txt.read_text().splitlines()
             if not l.startswith("#") and l.strip()]

    i = 0
    while i < len(lines):
        parts = lines[i].split()
        if len(parts) < 9:
            i += 1
            continue
        img_id = int(parts[0])
        qw, qx, qy, qz = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
        tx, ty, tz = float(parts[5]), float(parts[6]), float(parts[7])
        cam_id = int(parts[8])
        name = parts[9]

        # 쿼터니언 → 회전행렬
        R = np.array([
            [1-2*(qy**2+qz**2), 2*(qx*qy-qz*qw), 2*(qx*qz+qy*qw)],
            [2*(qx*qy+qz*qw), 1-2*(qx**2+qz**2), 2*(qy*qz-qx*qw)],
            [2*(qx*qz-qy*qw), 2*(qy*qz+qx*qw), 1-2*(qx**2+qy**2)],
        ])
        t = np.array([tx, ty, tz])

        # COLMAP: world→cam / nerfstudio: cam→world (c2w)
        c2w = np.eye(4)
        c2w[:3, :3] = R.T
        c2w[:3, 3] = -R.T @ t

        # nerfstudio 좌표 변환 (y, z 축 반전)
        c2w[0:3, 1:3] *= -1

        frames.append({
            "file_path": str(images_dir / name),
            "transform_matrix": c2w.tolist(),
        })
        i += 2

    if not frames:
        sys.exit("[오류] 파싱된 프레임 없음")

    # 기본 카메라 파라미터 (첫 번째 카메라 사용)
    cam = list(cam_params.values())[0]

    # train / test 8:2 split
    import random
    random.seed(42)
    random.shuffle(frames)
    n_test = max(1, len(frames) // 5)
    test_frames = frames[:n_test]
    train_frames = frames[n_test:]

    out_dir.mkdir(parents=True, exist_ok=True)
    for split, split_frames in [("train", train_frames), ("test", test_frames), ("", frames)]:
        suffix = f"_{split}" if split else ""
        transforms = {**cam, "frames": split_frames}
        (out_dir / f"transforms{suffix}.json").write_text(
            json.dumps(transforms, indent=2))

    print(f"[완료] transforms.json 생성 (train: {len(train_frames)}, test: {n_test})")
    return out_dir


def _print_split_stats(out_dir: Path):
    for fname in ["transforms_train.json", "transforms_test.json", "transforms.json"]:
        p = out_dir / fname
        if p.exists():
            data = json.loads(p.read_text())
            print(f"  {fname}: {len(data['frames'])}프레임")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--obj_dir", type=Path, required=True)
    parser.add_argument("--masked", action="store_true", help="마스킹된 이미지 사용 (실험 D)")
    args = parser.parse_args()

    ns_process(args.obj_dir, args.masked)


if __name__ == "__main__":
    main()
