import cv2
from ultralytics import YOLO

# 1. 모델 로드
yolo_model = YOLO("yolo11n.pt")
detector = cv2.FaceDetectorYN.create("face_detection_yunet_2023mar.onnx", "", (320, 320))
recognizer = cv2.FaceRecognizerSF.create("face_recognition_sface_2021dec.onnx", "")

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret: break

    # YOLO로 사람 찾기
    results = yolo_model.predict(frame, device='mps', classes=[0], verbose=False) # class 0은 person
    
    for r in results[0].boxes.data:
        x1, y1, x2, y2, conf, cls = r.tolist()
        person_crop = frame[int(y1):int(y2), int(x1):int(x2)]
        
        if person_crop.size > 0:
            # 잘라낸 사람 영역에서 얼굴 찾기
            h, w, _ = person_crop.shape
            detector.setInputSize((w, h))
            _, faces = detector.detect(person_crop)
            
            if faces is not None:
                # 얼굴이 발견되면 SFace로 특징 추출 (이게 정밀 데이터가 됨)
                face_feature = recognizer.feature(recognizer.alignCrop(person_crop, faces[0]))
                cv2.putText(frame, "Face Captured!", (int(x1), int(y1)-10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    cv2.imshow("Umbrella System Alpha", frame)
    if cv2.waitKey(1) == ord('q'): break

cap.release()
cv2.destroyAllWindows()
