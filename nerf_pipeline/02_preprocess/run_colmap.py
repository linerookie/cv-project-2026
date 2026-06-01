"""
COLMAP SfM 파이프라인 래퍼 (수업 6주차 — Structure from Motion)
SIFT 매칭 → Essential Matrix 분해 → 번들 조정 → sparse point cloud

사용법:
    python run_colmap.py --obj_dir ../data/objects/obj01_matte
    python run_colmap.py --obj_dir ../data/objects/obj01_matte --camera_params ../01_capture/camera_params.json
"""

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path


def run(cmd: list, desc: str):
    print(f"\n[COLMAP] {desc}")
    print("  " + " ".join(str(c) for c in cmd))
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(result.stderr[-2000:])
        sys.exit(f"[오류] {desc} 실패")
    return result.stdout


def build_camera_model_arg(params_path: Path) -> list:
    """캘리브레이션 결과를 COLMAP camera_model 인자로 변환"""
    if not params_path or not params_path.exists():
        return []
    data = json.loads(params_path.read_text())
    K = data["K"]
    d = data["dist"]
    fx, fy = K[0][0], K[1][1]
    cx, cy = K[0][2], K[1][2]
    k1, k2, p1, p2 = d[0], d[1], d[2], d[3]
    return [
        "--ImageReader.camera_model", "OPENCV",
        "--ImageReader.camera_params",
        f"{fx},{fy},{cx},{cy},{k1},{k2},{p1},{p2}",
        "--ImageReader.single_camera", "1",
    ]


def run_colmap_pipeline(obj_dir: Path, camera_params: Path = None, use_gpu: bool = True):
    images_dir = obj_dir / "images"
    db_path = obj_dir / "database.db"
    sparse_dir = obj_dir / "sparse"
    sparse_dir.mkdir(exist_ok=True)

    if not images_dir.exists() or not any(images_dir.iterdir()):
        sys.exit(f"[오류] 이미지 없음: {images_dir}")

    img_count = len(list(images_dir.glob("*.jpg")) + list(images_dir.glob("*.png")))
    print(f"[SfM 시작] {obj_dir.name} — {img_count}장 이미지")

    gpu_flag = "1" if use_gpu else "0"
    camera_args = build_camera_model_arg(camera_params)

    # 1단계: Feature extraction (SIFT)
    run(
        ["colmap", "feature_extractor",
         "--database_path", str(db_path),
         "--image_path", str(images_dir),
         "--FeatureExtraction.use_gpu", gpu_flag,
         "--SiftExtraction.max_image_size", "3200",
         ] + camera_args,
        "Feature extraction (SIFT)"
    )

    # 2단계: Feature matching
    run(
        ["colmap", "exhaustive_matcher",
         "--database_path", str(db_path),
         "--FeatureMatching.use_gpu", gpu_flag,
         ],
        "Feature matching (Exhaustive)"
    )

    # 3단계: Sparse reconstruction (Bundle Adjustment)
    run(
        ["colmap", "mapper",
         "--database_path", str(db_path),
         "--image_path", str(images_dir),
         "--output_path", str(sparse_dir),
         "--Mapper.ba_global_function_tolerance", "1e-6",
         ],
        "Sparse reconstruction + Bundle Adjustment"
    )

    # 4단계: 모델 변환 (nerfstudio transforms.json 용)
    model_dir = sparse_dir / "0"
    if not model_dir.exists():
        sys.exit("[오류] sparse/0 없음 — 매칭 실패. 이미지가 충분한지 확인하세요.")

    run(
        ["colmap", "model_converter",
         "--input_path", str(model_dir),
         "--output_path", str(model_dir),
         "--output_type", "TXT",
         ],
        "Model → TXT 변환"
    )

    # 요약 출력
    cameras_txt = model_dir / "cameras.txt"
    images_txt = model_dir / "images.txt"
    points_txt = model_dir / "points3D.txt"

    if cameras_txt.exists():
        cam_lines = [l for l in cameras_txt.read_text().splitlines() if not l.startswith("#") and l.strip()]
        img_lines = [l for l in images_txt.read_text().splitlines() if not l.startswith("#") and l.strip()]
        pts_lines = [l for l in points_txt.read_text().splitlines() if not l.startswith("#") and l.strip()]

        reg_imgs = len(img_lines) // 2
        pts3d = len(pts_lines)
        print(f"\n[SfM 완료]")
        print(f"  등록된 이미지: {reg_imgs} / {img_count}")
        print(f"  3D 포인트:    {pts3d}")
        if reg_imgs < img_count * 0.8:
            print(f"  [경고] 등록률 {reg_imgs/img_count*100:.0f}% < 80% — 이미지 품질 확인 필요")

    print(f"\n결과 저장: {sparse_dir}")
    return sparse_dir


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--obj_dir", type=Path, required=True)
    parser.add_argument("--camera_params", type=Path, default=None,
                        help="calibrate.py 출력 JSON (없으면 COLMAP 자동 추정)")
    parser.add_argument("--gpu", action="store_true", help="GPU 사용 (CUDA 환경 전용)")
    args = parser.parse_args()

    run_colmap_pipeline(args.obj_dir, args.camera_params, args.gpu)


if __name__ == "__main__":
    main()
