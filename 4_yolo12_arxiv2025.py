"""
YOLOv12: Attention-Centric Real-Time Object Detectors
Authors: Tian, Y., Ye, Q., Ruan, R., et al.
arXiv: 2502.12524 (February 2025)

핵심 혁신 (이전 YOLO와의 차이):
- 기존 YOLO: convolution 중심 neck (PANet, CSP 등)
- YOLOv12: Area Attention 기반 neck으로 전환
  - R=2 비율의 지역 어텐션으로 global attention 근사
  - FlashAttention 활용으로 메모리 효율 최적화
  - R-ELAN (Residual Efficient Layer Aggregation Networks)

이 파일에서는 yolo11n.pt vs yolo12n.pt를 나란히 비교해
Attention 기반 아키텍처의 실질적 차이를 눈으로 확인합니다.
"""

from ultralytics import YOLO
import cv2
import time
import numpy as np

# ─── 두 모델 로드 (비교용) ───
print("모델 로드 중...")
model_v12 = YOLO("yolo12n.pt")   # arXiv 2025 - Attention 기반
model_v11 = YOLO("yolo11n.pt")   # 기존 Convolution 기반

cap = cv2.VideoCapture(0)

# FPS 측정 버퍼
fps_v11_buf, fps_v12_buf = [], []

print("=== YOLOv12 vs YOLOv11 비교 (arXiv Feb 2025) ===")
print("논문: Tian et al., arXiv:2502.12524")
print("q: 종료")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # ─── YOLO11 추론 ───
    t0 = time.perf_counter()
    r11 = model_v11.predict(frame, device='mps', verbose=False)
    fps_v11_buf.append(1.0 / (time.perf_counter() - t0))
    if len(fps_v11_buf) > 15: fps_v11_buf.pop(0)

    # ─── YOLO12 추론 ───
    t0 = time.perf_counter()
    r12 = model_v12.predict(frame, device='mps', verbose=False)
    fps_v12_buf.append(1.0 / (time.perf_counter() - t0))
    if len(fps_v12_buf) > 15: fps_v12_buf.pop(0)

    fps_v11 = np.mean(fps_v11_buf)
    fps_v12 = np.mean(fps_v12_buf)

    # ─── 시각화 ───
    vis_v11 = frame.copy()
    vis_v12 = frame.copy()

    def draw_boxes(img, result, color):
        for box in result[0].boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            cls_name = result[0].names[int(box.cls[0])]
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            label = f"{cls_name} {conf:.2f}"
            (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(img, (x1, y1 - lh - 4), (x1 + lw + 4, y1), color, -1)
            cv2.putText(img, label, (x1 + 2, y1 - 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        return img

    # YOLO11: 파란색 박스, YOLO12: 초록색 박스
    draw_boxes(vis_v11, r11, (255, 80, 0))
    draw_boxes(vis_v12, r12, (0, 220, 80))

    n11 = len(r11[0].boxes)
    n12 = len(r12[0].boxes)

    # 상단 정보 패널
    for img, fps, n, label, color in [
        (vis_v11, fps_v11, n11, "YOLO11 (Conv)", (255, 80, 0)),
        (vis_v12, fps_v12, n12, "YOLO12 (Attention)", (0, 220, 80))
    ]:
        cv2.rectangle(img, (0, 0), (img.shape[1], 70), (20, 20, 20), -1)
        cv2.putText(img, label, (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        cv2.putText(img, f"FPS: {fps:.1f}  |  Detections: {n}", (10, 52),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)

    # 하단 논문 정보
    cv2.putText(vis_v12, "arXiv:2502.12524 (2025) - Area Attention", (10, vis_v12.shape[0] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (180, 180, 180), 1)

    # 좌우 합치기
    divider = np.full((frame.shape[0], 3, 3), [0, 200, 255], dtype=np.uint8)
    combined = np.hstack([vis_v11, divider, vis_v12])

    cv2.imshow("YOLOv11 vs YOLOv12 (arXiv 2025)", combined)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
