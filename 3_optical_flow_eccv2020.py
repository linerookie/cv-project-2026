"""
Dense Optical Flow Visualization
Paper: "RAFT: Recurrent All-Pairs Field Transforms for Optical Flow"
Authors: Teed, Z. & Deng, J.
Conference: ECCV 2020 (European Conference on Computer Vision) - Best Paper Award
Citations: ~4,000+

핵심 아이디어:
- 4D correlation volume 구축: 모든 픽셀 쌍의 유사도 계산
- GRU 기반 반복적 flow 업데이트 (수렴까지 반복)
- 작은 물체와 빠른 움직임에서 이전 SOTA 대비 큰 성능 향상

이 구현: cv2.calcOpticalFlowFarneback (Farneback 2003) 사용
- RAFT 대비 단순하지만 순수 OpenCV로 즉시 실행 가능
- 시각화 방식(HSV color coding)은 RAFT 논문의 표준 flow visualization과 동일

optical flow 컬러 의미:
- 색상(Hue) = 움직임 방향 (0°=오른쪽, 90°=위쪽, ...)
- 밝기(Value) = 움직임 크기 (밝을수록 빠른 움직임)

사용법:
- 's': sparse/dense 모드 전환 (Lucas-Kanade / Farneback)
- 'q': 종료
"""

import cv2
import numpy as np

cap = cv2.VideoCapture(0)
ret, prev_frame = cap.read()
prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)

# Shi-Tomasi 코너 검출 (sparse LK용) - CVPR 1994
feature_params = dict(maxCorners=200, qualityLevel=0.01, minDistance=10, blockSize=7)
lk_params = dict(winSize=(21, 21), maxLevel=3,
                 criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))

mode = "dense"  # "dense" or "sparse"
tracks = []     # sparse 모드 트랙
TRACK_LEN = 15  # 트레일 길이

# Dense optical flow를 HSV 컬러맵으로 변환하는 함수 (RAFT 논문 시각화 방식)
def flow_to_color(flow):
    h, w = flow.shape[:2]
    fx, fy = flow[:, :, 0], flow[:, :, 1]
    angle = np.arctan2(fy, fx) + np.pi  # 0 ~ 2π
    magnitude = np.sqrt(fx**2 + fy**2)

    hsv = np.zeros((h, w, 3), dtype=np.uint8)
    hsv[:, :, 0] = angle * 90 / np.pi   # Hue: 방향 (0~179)
    hsv[:, :, 1] = 255                   # Saturation: 100%
    hsv[:, :, 2] = np.clip(magnitude * 12, 0, 255).astype(np.uint8)  # Value: 속도

    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

print("=== Optical Flow (ECCV 2020 RAFT 개념 / Farneback 구현) ===")
print("논문: Teed & Deng, ECCV 2020 Best Paper")
print("s: dense/sparse 모드 전환")
print("q: 종료")

frame_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame_count += 1

    if mode == "dense":
        # ─── Dense Optical Flow (Farneback 2003) ───
        # pyr_scale: 피라미드 스케일
        # levels: 피라미드 레벨
        # winsize: 평균화 윈도우 크기
        # iterations: 각 레벨 반복 횟수
        # poly_n: 픽셀 주변 영역 크기
        # poly_sigma: Gaussian 표준편차
        flow = cv2.calcOpticalFlowFarneback(
            prev_gray, gray,
            None,
            pyr_scale=0.5,
            levels=3,
            winsize=15,
            iterations=3,
            poly_n=5,
            poly_sigma=1.2,
            flags=0
        )

        flow_vis = flow_to_color(flow)

        # 원본 + flow 오버레이
        out = cv2.addWeighted(frame, 0.4, flow_vis, 0.6, 0)

        # 움직임 크기 평균 (화면 전체 움직임 척도)
        magnitude = np.sqrt(flow[:, :, 0]**2 + flow[:, :, 1]**2)
        avg_motion = float(np.mean(magnitude))

        cv2.putText(out, f"Dense Flow | Motion: {avg_motion:.2f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        cv2.putText(out, "Color = Direction, Brightness = Speed", (10, 58),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
        cv2.putText(out, "ECCV 2020 Best Paper - RAFT (Farneback approx.)", (10, out.shape[0] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (200, 200, 200), 1)

    else:
        # ─── Sparse Optical Flow (Lucas-Kanade pyramidal) ───
        # 매 10프레임마다 새 feature 검출
        if frame_count % 10 == 0 or len(tracks) < 20:
            corners = cv2.goodFeaturesToTrack(gray, mask=None, **feature_params)
            if corners is not None:
                for c in corners:
                    tracks.append([c.reshape(1, 2).tolist()])

        out = frame.copy()

        if len(tracks) > 0:
            # 이전 포인트와 현재 포인트 준비
            p0 = np.array([t[-1] for t in tracks], dtype=np.float32)
            p1, status, _ = cv2.calcOpticalFlowPyrLK(prev_gray, gray, p0, None, **lk_params)

            if p1 is not None:
                good_new = p1[status.ravel() == 1]
                good_old = p0[status.ravel() == 1]

                new_tracks = []
                for i, (new, old) in enumerate(zip(good_new, good_old)):
                    nx, ny = new.ravel()
                    ox, oy = old.ravel()

                    if i < len(tracks) and status.ravel()[i] == 1:
                        track = tracks[i] + [[[int(nx), int(ny)]]]
                        if len(track) > TRACK_LEN:
                            track = track[-TRACK_LEN:]
                        new_tracks.append(track)

                        # 트레일 그리기
                        for j in range(1, len(track)):
                            p_old = tuple(track[j-1][0])
                            p_new = tuple(track[j][0])
                            alpha = j / len(track)
                            color = (int(255 * alpha), int(100 * alpha), int(200 * (1 - alpha)))
                            cv2.line(out, p_old, p_new, color, 1)
                        cv2.circle(out, (int(nx), int(ny)), 3, (0, 255, 0), -1)

                tracks = new_tracks

        cv2.putText(out, f"Sparse LK Flow | Tracks: {len(tracks)}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        cv2.putText(out, "ECCV 2020 Best Paper - RAFT (LK approx.)", (10, out.shape[0] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (200, 200, 200), 1)

    cv2.putText(out, f"Mode: {'DENSE' if mode == 'dense' else 'SPARSE'}  [s: switch]",
                (10, out.shape[0] - 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 200, 255), 1)

    cv2.imshow("Optical Flow (ECCV 2020)", out)
    prev_gray = gray.copy()

    key = cv2.waitKey(1) & 0xFF
    if key == ord('s'):
        mode = "sparse" if mode == "dense" else "dense"
        tracks = []
        print(f"Mode: {mode}")
    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
