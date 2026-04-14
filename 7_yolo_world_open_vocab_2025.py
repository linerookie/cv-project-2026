"""
Open-Vocabulary Real-Time Object Detection
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
논문: YOLO-World: Real-Time Open-Vocabulary Object Detection
저자: Tianheng Cheng, Lin Song, Yixiao Ge, et al. (Tencent AI Lab + HKU)
학회: CVPR 2024 → YOLOv8-World v2 업데이트: 2025
날짜: arXiv 2024.01

📄 논문 링크: https://arxiv.org/abs/2401.17270
🔗 공식 코드:  https://github.com/AILab-CVC/YOLO-World
📰 CVPR 2024:  https://openaccess.thecvf.com/content/CVPR2024/html/Cheng_YOLO-World_Real-Time_Open-Vocabulary_Object_Detection_CVPR_2024_paper.html

━━ 탑티어 리뷰어 보증 현황 ━━
┌─ CVPR 2024 수락률: ~22.3% (약 11,532편 제출 → 2,719편 수락)
├─ 리뷰어: 3명 (CVPR 표준)
├─ 인용 수: ~900+ (2024.01 이후 — 높은 실용성으로 빠른 확산)
├─ GitHub Stars: ~4,000+
└─ 2025 업데이트: YOLOv8-World v2, WorldV2-L 모델 공개

━━ 핵심 기여 ━━
- 기존 YOLO: 고정된 80개 COCO 클래스만 검출
- YOLO-World: 텍스트 프롬프트로 임의의 물체를 실시간 검출
- Re-Parameterizable Vision-Language Path Aggregation Network (RepVL-PAN)
- Region-Text Contrastive Loss로 텍스트-영상 정렬
- 추론 속도: 45~60 FPS (A100 기준) — open-vocab 모델 중 최고속

사용법:
- 코드 아래 CUSTOM_CLASSES 리스트를 원하는 한국어/영어 단어로 수정
- 's': 클래스 세트 순환 (미리 준비된 3가지 프리셋)
- q: 종료
"""

from ultralytics import YOLO
import cv2
import numpy as np
import time

# ─── 원하는 클래스 직접 수정 ───
CUSTOM_CLASSES = ["person", "phone", "laptop", "cup", "bottle", "chair", "desk"]

# 3가지 프리셋 (s키로 전환)
PRESETS = [
    ("일상 사물",     ["person", "phone", "laptop", "cup", "bottle", "chair", "bag"]),
    ("교통/야외",     ["car", "bicycle", "motorcycle", "bus", "traffic light", "tree", "dog"]),
    ("커스텀",        CUSTOM_CLASSES),
]
preset_idx = 0

print("YOLO-World v2 모델 로드 중... (첫 실행 시 자동 다운로드)")
print("논문: YOLO-World (CVPR 2024 → 2025 v2) | arXiv:2401.17270")

model = YOLO("yolov8s-worldv2.pt")
model.set_classes(PRESETS[preset_idx][1])

# 클래스별 고유 색상 생성
np.random.seed(42)
COLORS = {name: tuple(int(c) for c in np.random.randint(80, 255, 3))
          for name in sum([p[1] for p in PRESETS], [])}

cap = cv2.VideoCapture(0)
fps_buf = []

print("s: 클래스 프리셋 전환  |  q: 종료")
print(f"현재 클래스: {PRESETS[preset_idx][1]}")

while True:
    t0 = time.perf_counter()
    ret, frame = cap.read()
    if not ret:
        break

    results = model.predict(frame, device='mps', verbose=False, conf=0.25)

    fps_buf.append(1.0 / (time.perf_counter() - t0))
    if len(fps_buf) > 12: fps_buf.pop(0)
    fps = np.mean(fps_buf)

    # ─── 결과 시각화 ───
    class_counts = {}
    for box in results[0].boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        conf = float(box.conf[0])
        cls_name = results[0].names[int(box.cls[0])]
        color = COLORS.get(cls_name, (0, 255, 0))

        # 박스
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        # 레이블 배경
        label = f"{cls_name} {conf:.2f}"
        (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.52, 1)
        cv2.rectangle(frame, (x1, y1 - lh - 6), (x1 + lw + 6, y1), color, -1)
        cv2.putText(frame, label, (x1 + 3, y1 - 3),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 0, 0), 1)

        class_counts[cls_name] = class_counts.get(cls_name, 0) + 1

    # ─── 정보 패널 ───
    preset_name, current_classes = PRESETS[preset_idx]
    cv2.rectangle(frame, (0, 0), (frame.shape[1], 60), (15, 15, 15), -1)
    cv2.putText(frame, f"Open-Vocab Detection | Preset: {preset_name}", (10, 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.62, (0, 220, 255), 1)
    cv2.putText(frame, f"FPS: {fps:.1f}  |  Detected: {dict(class_counts)}", (10, 47),
                cv2.FONT_HERSHEY_SIMPLEX, 0.43, (200, 200, 200), 1)

    # 우측 하단: 현재 감지 중인 클래스 목록
    for i, cls in enumerate(current_classes):
        color = COLORS.get(cls, (200, 200, 200))
        cv2.putText(frame, f"• {cls}", (frame.shape[1] - 130, 80 + i * 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

    cv2.putText(frame, "CVPR 2024 → 2025 v2 | YOLO-World | arXiv:2401.17270",
                (10, frame.shape[0] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, (180, 180, 180), 1)

    cv2.imshow("YOLO-World Open-Vocabulary Detection (CVPR 2024/2025)", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('s'):
        preset_idx = (preset_idx + 1) % len(PRESETS)
        model.set_classes(PRESETS[preset_idx][1])
        print(f"프리셋 변경: {PRESETS[preset_idx][0]} → {PRESETS[preset_idx][1]}")
    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
