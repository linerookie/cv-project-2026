"""
소품 촬영 가이드 — 웹캠으로 다시점 영상 수집
- 60~90장 / 객체
- 수평 360도 + 상단 45도 앙각 포함 권장 궤도

사용법:
    python capture_guide.py --obj matte_vase --out ../data/objects/obj01_matte
"""

import argparse
import time
from pathlib import Path

import cv2
import numpy as np


ORBIT_TARGETS = [
    ("수평 0°",   12, "카메라를 수평으로 유지하며 360도 촬영"),
    ("앙각 30°",  12, "카메라를 30도 위로 기울여 360도 촬영"),
    ("앙각 60°",   8, "카메라를 60도 위로 기울여 360도 촬영"),
    ("클로즈업",   8, "특징적인 재질 부분에 가까이 접근하여 촬영"),
]


def capture_session(obj_name: str, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[오류] 웹캠을 열 수 없습니다. 스마트폰으로 촬영 후 images/ 폴더에 복사하세요.")
        return

    count = 0
    orbit_idx = 0
    orbit_name, orbit_target, orbit_guide = ORBIT_TARGETS[orbit_idx]
    orbit_count = 0
    last_capture = 0

    print(f"\n[촬영 시작] {obj_name}")
    print(f"총 목표: {sum(t for _, t, _ in ORBIT_TARGETS)}장")
    print("SPACE: 촬영  |  n: 다음 궤도  |  q: 완료\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        overlay = frame.copy()

        # 격자 가이드라인
        h, w = frame.shape[:2]
        cx, cy = w // 2, h // 2
        cv2.line(overlay, (cx, 0), (cx, h), (200, 200, 200), 1)
        cv2.line(overlay, (0, cy), (w, cy), (200, 200, 200), 1)
        cv2.rectangle(overlay, (cx - w//6, cy - h//6), (cx + w//6, cy + h//6), (200, 200, 200), 1)

        alpha = 0.7
        display = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)

        # HUD
        cv2.rectangle(display, (0, 0), (w, 110), (0, 0, 0), -1)
        cv2.putText(display, f"[{orbit_idx+1}/{len(ORBIT_TARGETS)}] {orbit_name}: {orbit_guide}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 1)
        cv2.putText(display, f"이 구간: {orbit_count}/{orbit_target}장  |  전체: {count}장",
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
        cv2.putText(display, "SPACE: 촬영  n: 다음 구간  q: 완료",
                    (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)

        # 촬영 직후 플래시 효과
        if time.time() - last_capture < 0.15:
            display = cv2.addWeighted(display, 0.4, np.ones_like(display) * 255, 0.6, 0)

        cv2.imshow(f"촬영: {obj_name}", display)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

        if key == ord("n"):
            if orbit_idx < len(ORBIT_TARGETS) - 1:
                orbit_idx += 1
                orbit_name, orbit_target, orbit_guide = ORBIT_TARGETS[orbit_idx]
                orbit_count = 0
                print(f"\n→ 다음 구간: {orbit_name} — {orbit_guide}")

        if key == ord(" "):
            fname = out_dir / f"{obj_name}_{count:04d}.jpg"
            cv2.imwrite(str(fname), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
            count += 1
            orbit_count += 1
            last_capture = time.time()
            print(f"  저장: {fname.name}  (구간 {orbit_count}/{orbit_target})")

    cap.release()
    cv2.destroyAllWindows()

    print(f"\n[완료] {count}장 저장 → {out_dir}")
    if count < 60:
        print(f"[권장] {60 - count}장 추가 촬영 권장 (목표: 60~90장)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--obj", required=True, help="객체 이름 (예: matte_vase)")
    parser.add_argument("--out", type=Path, required=True, help="저장 폴더")
    args = parser.parse_args()

    capture_session(args.obj, args.out / "images")


if __name__ == "__main__":
    main()
