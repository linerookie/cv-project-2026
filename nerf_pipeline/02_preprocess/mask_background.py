"""
배경 마스킹 전처리 — 실험 D (Masking Ablation)
rembg (U2Net 기반) 로 객체 전경 분리 → RGBA PNG 저장

사용법:
    python mask_background.py --obj_dir ../data/objects/obj01_matte
    python mask_background.py --obj_dir ../data/objects/obj01_matte --preview  # 결과 미리보기
"""

import argparse
from pathlib import Path

import cv2
import numpy as np
from rembg import remove
from PIL import Image
import io


def process_image(img_path: Path, out_dir: Path, preview: bool = False) -> Path:
    with open(img_path, "rb") as f:
        input_bytes = f.read()

    output_bytes = remove(input_bytes)
    rgba = Image.open(io.BytesIO(output_bytes)).convert("RGBA")

    out_path = out_dir / (img_path.stem + ".png")
    rgba.save(out_path)

    if preview:
        img_cv = cv2.imread(str(img_path))
        mask_arr = np.array(rgba)[:, :, 3]
        mask_3ch = cv2.cvtColor(mask_arr, cv2.COLOR_GRAY2BGR)
        fg = cv2.bitwise_and(img_cv, img_cv, mask=mask_arr)
        # 배경을 흰색으로 채워서 시각화
        bg = np.ones_like(img_cv) * 255
        bg[mask_arr > 128] = img_cv[mask_arr > 128]
        preview_img = np.hstack([img_cv, bg])
        cv2.imshow("원본 | 마스킹 결과", preview_img)
        cv2.waitKey(500)

    return out_path


def mask_all(obj_dir: Path, preview: bool = False):
    images_dir = obj_dir / "images"
    out_dir = obj_dir / "images_masked"
    out_dir.mkdir(exist_ok=True)

    imgs = sorted(images_dir.glob("*.jpg")) + sorted(images_dir.glob("*.png"))
    if not imgs:
        print(f"[오류] 이미지 없음: {images_dir}")
        return

    print(f"[마스킹] {obj_dir.name} — {len(imgs)}장 처리 중...")
    print("  모델: rembg U2Net (첫 실행 시 모델 다운로드 ~170MB)")

    for i, p in enumerate(imgs, 1):
        out = process_image(p, out_dir, preview)
        print(f"  [{i:3d}/{len(imgs)}] {p.name} → {out.name}")

    if preview:
        cv2.destroyAllWindows()

    print(f"\n[완료] {len(imgs)}장 → {out_dir}")
    print("  학습 시 --data.mask_path images_masked/ 옵션으로 사용")


def compare_mask_stats(obj_dir: Path):
    """마스킹 전후 픽셀 비율 요약"""
    orig_dir = obj_dir / "images"
    mask_dir = obj_dir / "images_masked"

    if not mask_dir.exists():
        print("[오류] 먼저 마스킹을 실행하세요.")
        return

    fg_ratios = []
    for p in sorted(mask_dir.glob("*.png"))[:10]:
        img = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
        if img.shape[2] == 4:
            alpha = img[:, :, 3]
            fg_ratio = (alpha > 128).sum() / alpha.size
            fg_ratios.append(fg_ratio)

    if fg_ratios:
        print(f"\n[마스크 통계] 평균 전경 비율: {np.mean(fg_ratios)*100:.1f}%")
        print(f"  배경 제거 비율: {(1-np.mean(fg_ratios))*100:.1f}%")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--obj_dir", type=Path, required=True)
    parser.add_argument("--preview", action="store_true", help="처리 결과 미리보기")
    parser.add_argument("--stats", action="store_true", help="마스크 통계 출력")
    args = parser.parse_args()

    mask_all(args.obj_dir, args.preview)
    if args.stats:
        compare_mask_stats(args.obj_dir)


if __name__ == "__main__":
    main()
