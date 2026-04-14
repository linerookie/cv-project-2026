from ultralytics import YOLO
import cv2

# YOLOv12 대신 검증된 YOLO11 사용 (이 모델은 즉시 작동합니다)
model = YOLO("yolo11n.pt") 

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("❌ 카메라 권한을 확인하세요 (시스템 설정 -> 보안 -> 카메라)")
    exit()

while True:
    ret, frame = cap.read()
    if not ret: break

    try:
        # 맥북 GPU(mps) 가속 사용
        results = model.predict(frame, device='mps', conf=0.25, verbose=False)
        
        # 시각화
        annotated_frame = results[0].plot()
        cv2.imshow("YOLO11 Real-time Detection", annotated_frame)
    except Exception as e:
        print(f"❌ 에러 발생: {e}")
        break

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
