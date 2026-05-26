"""
카메라 캘리브레이션 — Zhang's Method (수업 2~3주차)
체커보드 영상에서 내부 파라미터 K와 왜곡 계수 추정

사용법:
    python calibrate.py --images ./checkerboard_imgs --out ./camera_params.json
    python calibrate.py --live  # 웹캠 실시간 캘리브레이션
"""

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np


BOARD_W = 9   # 체커보드 내부 코너 가로
BOARD_H = 6   # 체커보드 내부 코너 세로
SQUARE_MM = 25.0  # 한 칸 실제 크기 (mm)


def find_corners(img_path: Path):
    img = cv2.imread(str(img_path))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    found, corners = cv2.findChessboardCorners(gray, (BOARD_W, BOARD_H), None)
    if not found:
        return None, gray.shape[::-1]
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
    return corners, gray.shape[::-1]


def calibrate_from_images(img_dir: Path) -> dict:
    obj_p = np.zeros((BOARD_W * BOARD_H, 3), np.float32)
    obj_p[:, :2] = np.mgrid[0:BOARD_W, 0:BOARD_H].T.reshape(-1, 2) * SQUARE_MM

    obj_points, img_points = [], []
    img_size = None
    valid_count = 0

    imgs = sorted(img_dir.glob("*.jpg")) + sorted(img_dir.glob("*.png"))
    print(f"[캘리브레이션] {len(imgs)}장 이미지 처리 중...")

    for p in imgs:
        corners, size = find_corners(p)
        if corners is None:
            print(f"  ✗ {p.name} — 코너 미검출")
            continue
        obj_points.append(obj_p)
        img_points.append(corners)
        img_size = size
        valid_count += 1
        print(f"  ✓ {p.name}")

    if valid_count < 10:
        sys.exit(f"[오류] 유효 이미지 {valid_count}장 < 10장. 더 촬영하세요.")

    print(f"\n유효 이미지: {valid_count}장 → 캘리브레이션 실행 중...")
    rms, K, dist, rvecs, tvecs = cv2.calibrateCamera(
        obj_points, img_points, img_size, None, None
    )

    print(f"\n[결과]")
    print(f"  RMS 재투영 오차: {rms:.4f} px  (< 1.0 권장)")
    print(f"  fx={K[0,0]:.1f}  fy={K[1,1]:.1f}")
    print(f"  cx={K[0,2]:.1f}  cy={K[1,2]:.1f}")
    print(f"  왜곡계수: {dist.ravel()}")

    return {
        "rms": float(rms),
        "K": K.tolist(),
        "dist": dist.ravel().tolist(),
        "img_size": list(img_size),
        "board": {"w": BOARD_W, "h": BOARD_H, "square_mm": SQUARE_MM},
    }


def calibrate_live() -> dict:
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        sys.exit("[오류] 웹캠을 열 수 없습니다.")

    obj_p = np.zeros((BOARD_W * BOARD_H, 3), np.float32)
    obj_p[:, :2] = np.mgrid[0:BOARD_W, 0:BOARD_H].T.reshape(-1, 2) * SQUARE_MM

    obj_points, img_points = [], []
    img_size = None
    collected = 0

    print("[실시간 캘리브레이션]")
    print("  체커보드를 여러 각도로 보여주세요.")
    print("  SPACE: 현재 프레임 저장  |  q: 완료 (20장 이상 권장)")

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        found, corners = cv2.findChessboardCorners(gray, (BOARD_W, BOARD_H), None)

        display = frame.copy()
        if found:
            cv2.drawChessboardCorners(display, (BOARD_W, BOARD_H), corners, found)
            cv2.putText(display, "FOUND — SPACE to capture", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        else:
            cv2.putText(display, "Searching...", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        cv2.putText(display, f"Captured: {collected}", (20, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 200, 0), 2)
        cv2.imshow("Camera Calibration", display)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key == ord(" ") and found:
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            obj_points.append(obj_p)
            img_points.append(corners)
            img_size = gray.shape[::-1]
            collected += 1
            print(f"  캡처 {collected}장 완료")

    cap.release()
    cv2.destroyAllWindows()

    if collected < 10:
        sys.exit(f"[오류] {collected}장 수집 < 10장. 다시 시도하세요.")

    rms, K, dist, _, _ = cv2.calibrateCamera(obj_points, img_points, img_size, None, None)
    print(f"\nRMS: {rms:.4f} px  |  fx={K[0,0]:.1f}, fy={K[1,1]:.1f}")

    return {
        "rms": float(rms),
        "K": K.tolist(),
        "dist": dist.ravel().tolist(),
        "img_size": list(img_size),
        "board": {"w": BOARD_W, "h": BOARD_H, "square_mm": SQUARE_MM},
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--images", type=Path, help="체커보드 이미지 폴더")
    parser.add_argument("--live", action="store_true", help="웹캠 실시간 캘리브레이션")
    parser.add_argument("--out", type=Path, default=Path("camera_params.json"))
    args = parser.parse_args()

    if args.live:
        params = calibrate_live()
    elif args.images:
        params = calibrate_from_images(args.images)
    else:
        parser.error("--images 또는 --live 중 하나를 선택하세요.")

    args.out.write_text(json.dumps(params, indent=2, ensure_ascii=False))
    print(f"\n캘리브레이션 파라미터 저장: {args.out}")

    if params["rms"] > 1.0:
        print("[경고] RMS > 1.0px — 더 많은 이미지 또는 다양한 각도 필요")


if __name__ == "__main__":
    main()
