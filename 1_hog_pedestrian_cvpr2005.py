"""
HOG + SVM Pedestrian Detection
Paper: "Histograms of Oriented Gradients for Human Detection"
Authors: Dalal, N. & Triggs, B.
Conference: CVPR 2005 (IEEE Conference on Computer Vision and Pattern Recognition)
Citations: ~30,000+ (딥러닝 이전 보행자 검출의 표준)

핵심 아이디어:
- 이미지를 8x8 픽셀 셀로 나눔
- 각 셀에서 gradient 방향 히스토그램 계산 (9개 방향 bin)
- 인접한 셀들을 16x16 블록으로 묶어 normalization
- 추출된 HOG feature vector를 SVM으로 분류

OpenCV 내장: cv2.HOGDescriptor() - Dalal & Triggs의 SVM 가중치가 그대로 포함됨
"""

import cv2
import numpy as np
import time

# ─── Dalal & Triggs (CVPR 2005)의 HOG + SVM 보행자 감지기 ───
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())  # 논문 모델 그대로

cap = cv2.VideoCapture(0)

fps_buffer = []

print("=== HOG Pedestrian Detection (CVPR 2005) ===")
print("논문: Dalal & Triggs, CVPR 2005")
print("q: 종료")

while True:
    t0 = time.time()
    ret, frame = cap.read()
    if not ret:
        break

    # HOG detectMultiScale: sliding window + image pyramid
    # winStride: 윈도우 이동 간격 (클수록 빠름, 작을수록 정밀)
    # padding: 각 윈도우 주변 패딩
    # scale: 이미지 피라미드 축소 비율 (1.05 = 5%씩 줄임)
    boxes, weights = hog.detectMultiScale(
        frame,
        winStride=(8, 8),
        padding=(4, 4),
        scale=1.05
    )

    # 결과 시각화
    for i, (x, y, w, h) in enumerate(boxes):
        confidence = float(weights[i])
        # 신뢰도에 따라 색상 변화 (낮음=노랑, 높음=초록)
        color = (0, int(min(255, confidence * 80)), 255 - int(min(255, confidence * 80)))
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        cv2.putText(frame, f"{confidence:.2f}", (x, y - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    # FPS 계산
    elapsed = time.time() - t0
    fps_buffer.append(1.0 / elapsed if elapsed > 0 else 0)
    if len(fps_buffer) > 10:
        fps_buffer.pop(0)
    fps = np.mean(fps_buffer)

    # 정보 오버레이
    cv2.putText(frame, f"HOG Pedestrians: {len(boxes)}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    cv2.putText(frame, f"FPS: {fps:.1f}", (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 1)
    cv2.putText(frame, "CVPR 2005 - Dalal & Triggs HOG+SVM", (10, frame.shape[0] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)

    cv2.imshow("HOG Pedestrian Detection (CVPR 2005)", frame)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
