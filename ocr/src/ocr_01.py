import cv2
import easyocr
import matplotlib.pyplot as plt
import time
import sys
import torch


def stop(text):
    if text == '정지' or text == 'STOP':
        print('정지 신호 감지 3초 후 정지.')
        time.sleep(3)
        print("정지합니다.")
        cap.release()
        cv2.destroyAllWindows()
        sys.exit()


# OCR 모델 준비
reader = easyocr.Reader(['ko', 'en'], gpu=True)

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: 카메라를 열 수 없습니다.")
    exit()

THRESHOLD = 0.5
# frame_count = 0
# OCR_INTERVAL = 5

while True:
    ret, frame = cap.read()

    if not ret:
        print("Error: 프레임을 읽을 수 없습니다.")
        break

    # # # OCR 적용
    # if frame_count % OCR_INTERVAL == 0:
    #     result = reader.readtext(frame)

    result = reader.readtext(frame)

    # 인식된 텍스트 표시
    for bbox, text, conf in result:
        if conf >= THRESHOLD:
            print(f"[{conf:.2f}] {text}")
            pt1 = tuple(map(int, bbox[0]))
            pt2 = tuple(map(int, bbox[2]))
            cv2.rectangle(frame, pt1, pt2, (0, 255, 0), 2)
            cv2.putText(frame, text, pt1, cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            stop(text)

    # 프레임 화면에 표시
    cv2.imshow('Webcam + OCR', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 자원 정리
cap.release()
cv2.destroyAllWindows()

print(torch.cuda.is_available())  # True가 나와야 GPU 사용 가능
print(torch.__version__)
print(torch.version.cuda)
