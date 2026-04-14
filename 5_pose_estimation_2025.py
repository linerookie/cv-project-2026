"""
Real-Time Human Pose Estimation
Paper: "RTMW: Real-time Multi-person 2D and 3D Whole-body Pose Estimation"
Authors: Jiang, T. et al. (Shanghai AI Lab)
Published: arXiv:2407.08634 → CVPR 2025 Workshop (및 관련 논문들)

핵심 아이디어:
- SimCC (Simple Coordinate Classification): keypoint 위치를 분류 문제로 변환
- TopDown: 먼저 사람을 검출 → 각 사람 영역에서 keypoint 예측
- 17개 COCO keypoint + 133개 whole-body keypoint 통합 지원

이 구현: YOLO11-pose (Ultralytics) 기반 실시간 골격 시각화
- yolo11n-pose.pt 자동 다운로드 (~6MB)
- COCO 17개 keypoint: 코, 눈, 귀, 어깨, 팔꿈치, 손목, 엉덩이, 무릎, 발목

사용법:
- 'k': keypoint 점 표시 토글
- 's': skeleton 연결선 표시 토글
- 'q': 종료
"""

from ultralytics import YOLO
import cv2
import numpy as np
import time

# ─── COCO 17 keypoint 골격 연결 정의 ───
# (시작 keypoint 인덱스, 끝 keypoint 인덱스, BGR 색상)
SKELETON_EDGES = [
    # 얼굴
    (0, 1, (255, 200, 0)),    # 코 → 왼눈
    (0, 2, (255, 200, 0)),    # 코 → 오른눈
    (1, 3, (255, 160, 0)),    # 왼눈 → 왼귀
    (2, 4, (255, 160, 0)),    # 오른눈 → 오른귀
    # 상체 (좌)
    (5, 7, (0, 200, 255)),    # 왼어깨 → 왼팔꿈치
    (7, 9, (0, 160, 255)),    # 왼팔꿈치 → 왼손목
    # 상체 (우)
    (6, 8, (255, 100, 0)),    # 오른어깨 → 오른팔꿈치
    (8, 10, (255, 60, 0)),    # 오른팔꿈치 → 오른손목
    # 몸통
    (5, 6, (0, 255, 150)),    # 왼어깨 → 오른어깨
    (5, 11, (0, 255, 100)),   # 왼어깨 → 왼엉덩이
    (6, 12, (0, 255, 100)),   # 오른어깨 → 오른엉덩이
    (11, 12, (0, 255, 150)),  # 왼엉덩이 → 오른엉덩이
    # 하체 (좌)
    (11, 13, (150, 0, 255)),  # 왼엉덩이 → 왼무릎
    (13, 15, (100, 0, 255)),  # 왼무릎 → 왼발목
    # 하체 (우)
    (12, 14, (255, 0, 200)),  # 오른엉덩이 → 오른무릎
    (14, 16, (255, 0, 150)),  # 오른무릎 → 오른발목
]

KEYPOINT_NAMES = [
    "코", "왼눈", "오른눈", "왼귀", "오른귀",
    "왼어깨", "오른어깨", "왼팔꿈치", "오른팔꿈치",
    "왼손목", "오른손목", "왼엉덩이", "오른엉덩이",
    "왼무릎", "오른무릎", "왼발목", "오른발목"
]

# ─── 모델 로드 ───
print("YOLO11-pose 모델 로드 중... (첫 실행 시 자동 다운로드)")
model = YOLO("yolo11n-pose.pt")

cap = cv2.VideoCapture(0)

show_keypoints = True
show_skeleton = True
fps_buf = []

print("=== Real-time Pose Estimation (CVPR 2025 관련) ===")
print("논문: RTMW - Jiang et al., arXiv:2407.08634")
print("k: keypoint 토글  |  s: skeleton 토글  |  q: 종료")

while True:
    t0 = time.perf_counter()
    ret, frame = cap.read()
    if not ret:
        break

    results = model.predict(frame, device='mps', verbose=False)
    fps_buf.append(1.0 / (time.perf_counter() - t0))
    if len(fps_buf) > 15: fps_buf.pop(0)
    fps = np.mean(fps_buf)

    out = frame.copy()
    num_people = 0

    for person in results[0]:
        kpts = person.keypoints  # shape: (1, 17, 3) [x, y, confidence]
        boxes = person.boxes

        if kpts is None or kpts.data is None:
            continue

        for i in range(len(kpts.data)):
            num_people += 1
            kp = kpts.data[i].cpu().numpy()  # (17, 3)

            # ─── skeleton 연결선 ───
            if show_skeleton:
                for (start, end, color) in SKELETON_EDGES:
                    xs, ys, cs = kp[start]
                    xe, ye, ce = kp[end]
                    if cs > 0.3 and ce > 0.3:
                        # 신뢰도에 따라 굵기 조절
                        thickness = max(1, int((cs + ce) * 3))
                        cv2.line(out,
                                 (int(xs), int(ys)), (int(xe), int(ye)),
                                 color, thickness, cv2.LINE_AA)

            # ─── keypoint 점 ───
            if show_keypoints:
                for j, (x, y, conf) in enumerate(kp):
                    if conf > 0.3:
                        # 신뢰도에 따라 크기/색상 변화
                        radius = max(3, int(conf * 8))
                        brightness = int(conf * 255)
                        cv2.circle(out, (int(x), int(y)), radius,
                                   (0, brightness, 255 - brightness), -1, cv2.LINE_AA)
                        cv2.circle(out, (int(x), int(y)), radius + 1,
                                   (255, 255, 255), 1, cv2.LINE_AA)

            # ─── 사람 bounding box ───
            if boxes is not None and len(boxes.xyxy) > i:
                x1, y1, x2, y2 = map(int, boxes.xyxy[i])
                cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 200), 1)

    # ─── 정보 패널 ───
    cv2.rectangle(out, (0, 0), (out.shape[1], 65), (20, 20, 20), -1)
    cv2.putText(out, f"Pose Estimation | People: {num_people}", (10, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 200), 2)
    cv2.putText(out, f"FPS: {fps:.1f}  |  KP: {'ON' if show_keypoints else 'OFF'}  "
                     f"Skeleton: {'ON' if show_skeleton else 'OFF'}", (10, 53),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)
    cv2.putText(out, "CVPR 2025 - RTMW Whole-body Pose (Jiang et al.)", (10, out.shape[0] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (180, 180, 180), 1)

    cv2.imshow("Real-time Pose Estimation (2025)", out)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('k'):
        show_keypoints = not show_keypoints
    elif key == ord('s'):
        show_skeleton = not show_skeleton
    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
