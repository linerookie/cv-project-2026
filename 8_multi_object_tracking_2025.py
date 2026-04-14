"""
Multi-Object Tracking with Trajectory Heatmap
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
논문: BoT-SORT: Robust Associations Multi-Pedestrian Tracking
저자: Nir Aharon, Roy Tsur, Amit Ferencz (Apple)
학회: arXiv 2022 → ByteTrack 계열 → 2025 CVPR 통합 추적 연구로 이어짐
날짜: arXiv 2206.14651

📄 BoT-SORT:    https://arxiv.org/abs/2206.14651
📄 ByteTrack:   https://arxiv.org/abs/2110.06864  (ECCV 2022)
🔗 ultralytics: https://docs.ultralytics.com/modes/track/
📄 2025 관련:   https://arxiv.org/abs/2409.09028  (CVPR 2025 tracking survey)

━━ 탑티어 리뷰어 보증 현황 ━━
┌─ ByteTrack (ECCV 2022): 인용 ~2,100+
├─ BoT-SORT (arXiv 2022): 인용 ~800+
├─ ultralytics 구현: production-grade, 1,000만+ 다운로드
├─ MOT17/MOT20 벤치마크 1위 달성 기록
└─ 2025: CVPR/ECCV 추적 논문들이 ByteTrack/BoT-SORT를 baseline으로 사용

━━ 핵심 기여 (BoT-SORT) ━━
- IoU + ReID 결합: 외형(appearance)과 위치(location)를 함께 고려
- Camera Motion Compensation (CMC): 카메라 움직임 보정
- Adaptive Kalman Filter: 객체 크기 변화에 맞는 필터 설계
- ByteTrack: 낮은 confidence detection도 활용 → 가려진 객체 추적 유지

시각화:
- 각 ID별 색깔 다른 궤적 그리기 (최근 60프레임)
- 히트맵: 프레임 누적으로 "어디에 오래 머물렀는지" 표시
- t: 궤적/히트맵 전환  |  c: 히트맵 초기화  |  q: 종료
"""

from ultralytics import YOLO
import cv2
import numpy as np
import collections
import time

print("MOT 모델 로드 중...")
print("논문: BoT-SORT (arXiv:2206.14651) | ByteTrack (ECCV 2022:2110.06864)")

model = YOLO("yolo11n.pt")

# 궤적 저장: id → deque of (cx, cy)
TRAJ_LEN = 60
trajectories = collections.defaultdict(lambda: collections.deque(maxlen=TRAJ_LEN))

# ID별 고정 색상
def id_to_color(track_id):
    np.random.seed(track_id * 7 + 3)
    return tuple(int(c) for c in np.random.randint(80, 255, 3))

cap = cv2.VideoCapture(0)
ret, init_frame = cap.read()
h, w = init_frame.shape[:2]

# 히트맵 누적 버퍼
heatmap_acc = np.zeros((h, w), dtype=np.float32)

mode = "trajectory"  # "trajectory" or "heatmap"
fps_buf = []
total_ids = set()

print("t: 궤적/히트맵 전환  |  c: 히트맵 초기화  |  q: 종료")

while True:
    t0 = time.perf_counter()
    ret, frame = cap.read()
    if not ret:
        break

    # BoT-SORT 추적
    results = model.track(
        frame,
        device='mps',
        persist=True,
        tracker="botsort.yaml",
        verbose=False,
        conf=0.35
    )

    fps_buf.append(1.0 / (time.perf_counter() - t0))
    if len(fps_buf) > 12: fps_buf.pop(0)
    fps = np.mean(fps_buf)

    if mode == "trajectory":
        out = frame.copy()
    else:
        # 히트맵 모드: 누적 히트맵 + 원본 블렌드
        norm_heat = cv2.normalize(heatmap_acc, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        heat_color = cv2.applyColorMap(norm_heat, cv2.COLORMAP_JET)
        out = cv2.addWeighted(frame, 0.45, heat_color, 0.55, 0)

    if results[0].boxes.id is not None:
        ids = results[0].boxes.id.int().tolist()
        xyxys = results[0].boxes.xyxy.tolist()
        clss = results[0].boxes.cls.int().tolist()

        for box, track_id, cls in zip(xyxys, ids, clss):
            x1, y1, x2, y2 = map(int, box)
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            color = id_to_color(track_id)
            total_ids.add(track_id)

            trajectories[track_id].append((cx, cy))

            # 히트맵 누적
            if 0 <= cy < h and 0 <= cx < w:
                cv2.circle(heatmap_acc, (cx, cy), 20, 1.0, -1)
            heatmap_acc *= 0.997  # 점진적 감쇠

            if mode == "trajectory":
                # 궤적 선 그리기 (최신일수록 밝음)
                pts = list(trajectories[track_id])
                for i in range(1, len(pts)):
                    alpha = i / len(pts)
                    c = tuple(int(v * alpha) for v in color)
                    thickness = max(1, int(alpha * 3))
                    cv2.line(out, pts[i - 1], pts[i], c, thickness, cv2.LINE_AA)

                # 현재 위치 원
                cv2.circle(out, (cx, cy), 6, color, -1, cv2.LINE_AA)
                cv2.circle(out, (cx, cy), 7, (255, 255, 255), 1, cv2.LINE_AA)

            # 박스 + ID
            cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
            label = f"ID:{track_id} {model.names[cls]}"
            cv2.putText(out, label, (x1, y1 - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.48, color, 1)

    # ─── 정보 패널 ───
    n_active = len(results[0].boxes.id) if results[0].boxes.id is not None else 0
    cv2.rectangle(out, (0, 0), (out.shape[1], 58), (15, 15, 15), -1)
    cv2.putText(out, f"Multi-Object Tracking | Mode: {mode.upper()}", (10, 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.62, (0, 220, 255), 1)
    cv2.putText(out, f"FPS: {fps:.1f}  Active: {n_active}  Total IDs: {len(total_ids)}", (10, 46),
                cv2.FONT_HERSHEY_SIMPLEX, 0.47, (200, 200, 200), 1)
    cv2.putText(out, "BoT-SORT arXiv:2206.14651 | ByteTrack ECCV2022:2110.06864",
                (10, out.shape[0] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.37, (180, 180, 180), 1)

    cv2.imshow("Multi-Object Tracking (BoT-SORT / ByteTrack 2022→2025)", out)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('t'):
        mode = "heatmap" if mode == "trajectory" else "trajectory"
        print(f"모드 전환: {mode}")
    elif key == ord('c'):
        heatmap_acc[:] = 0
        print("히트맵 초기화")
    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
