import cv2
import os
import csv
import numpy as np

# CLAHE 설정 (LAB 색 공간에서 L 채널에 적용)
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

# CSV 파일 경로
csv_file = 'color_labels.csv'
if not os.path.exists(csv_file):
    with open(csv_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['image_path'])

# 저장할 폴더
save_folder = '../cloth'
os.makedirs(save_folder, exist_ok=True)

# 웹캠 켜기
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Camera not accessible")
    exit()

# 영상 크기 가져오기
ret, frame = cap.read()
if not ret:
    print("Can't read frame")
    cap.release()
    exit()

h, w, _ = frame.shape

# ROI 설정: 영상 중앙 기준 100x100
roi_size = 100
x = w // 2 - roi_size // 2
y = h // 2 - roi_size // 2

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # ROI 사각형 표시
    cv2.rectangle(frame, (x, y), (x + roi_size, y + roi_size), (0, 255, 0), 2)
    cv2.imshow('Video', frame)

    key = cv2.waitKey(1)
    if key == ord('q'):
        break

    elif key == ord(' '):  # 스페이스바를 누르면 ROI 영역 저장

        roi = frame[y:y + roi_size, x:x + roi_size]

        # 🎯 조명에 강한 정규화
        # LAB 색공간은 명도(L)를 분리해서 조정 가능.
        # CLAHE는 밝은 부분과 어두운 부분을 모두 보정해서 조명에 강한 색상 보정 효과가 있어요.
        # RGB에서 바로 조정하면 색상 왜곡이 생길 수 있음.

        lab = cv2.cvtColor(roi, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        l_eq = clahe.apply(l)
        lab_eq = cv2.merge((l_eq, a, b))
        img1 = cv2.cvtColor(lab_eq, cv2.COLOR_LAB2BGR)

        # ✅ 가우시안 블러로 주름/섀도우 완화
        img1 = cv2.GaussianBlur(img1, (5, 5), 0)

        # 저장할 파일 번호 결정
        existing_files = [f for f in os.listdir(save_folder) if f.endswith('.jpg')]
        next_number = len(existing_files) + 1
        filename = f"{next_number}.jpg"
        file_path = os.path.join(save_folder, filename)
        cv2.imwrite(file_path, img1)

        # CSV 저장
        with open(csv_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([file_path])

        print(f"Saved: {file_path}")

cap.release()
cv2.destroyAllWindows()
