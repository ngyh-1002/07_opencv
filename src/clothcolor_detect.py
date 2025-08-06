import cv2
import os
import csv
import numpy as np
from collections import Counter

# --------------------------
# 설정
csv_file = 'color_labels.csv'
save_folder = '../cloth'
os.makedirs(save_folder, exist_ok=True)

# CLAHE 설정
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

# CSV 초기화 (없으면 헤더 작성)
if not os.path.exists(csv_file):
    with open(csv_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['image_path', 'cluster_id', 'cluster_name'])

# 기존 CSV에서 클러스터 정보 로딩
cluster_names = {}  # {cluster_id: cluster_name}
if os.path.exists(csv_file):
    with open(csv_file, 'r', newline='') as f:
        reader = csv.reader(f)
        next(reader, None)  # skip header
        for row in reader:
            if len(row) == 3:
                _, cid, cname = row
                cluster_names[int(cid)] = cname

# --------------------------
# 웹캠 열기
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Camera not accessible")
    exit()

ret, frame = cap.read()
if not ret:
    print("Can't read frame")
    cap.release()
    exit()

h, w, _ = frame.shape
roi_size = 100
x = w // 2 - roi_size // 2
y = h // 2 - roi_size // 2

while True:
    ret, frame = cap.read()
    if not ret:
        break

    cv2.rectangle(frame, (x, y), (x + roi_size, y + roi_size), (0, 255, 0), 2)
    cv2.imshow('Video', frame)

    key = cv2.waitKey(1)
    if key == ord('q'):
        break

    elif key == ord(' '):  # 스페이스바로 ROI 저장
        roi = frame[y:y + roi_size, x:x + roi_size]

        # CLAHE + 가우시안 블러
        lab = cv2.cvtColor(roi, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        l_eq = clahe.apply(l)
        lab_eq = cv2.merge((l_eq, a, b))
        img1 = cv2.cvtColor(lab_eq, cv2.COLOR_LAB2BGR)
        img1 = cv2.GaussianBlur(img1, (5, 5), 0)

        # 저장 번호 확인
        existing_files = [f for f in os.listdir(save_folder) if f.endswith('.jpg')]
        next_number = len(existing_files) + 1
        file_path = os.path.join(save_folder, f"{next_number}.jpg")
        cv2.imwrite(file_path, img1)

        # 클러스터 정보 결정
        cluster_id = (next_number - 1) // 30

        # cluster_name이 이미 등록돼 있지 않다면 사용자에게 입력받기
        if cluster_id not in cluster_names:
            print(f"\n🔵 새 클러스터 감지됨: ID {cluster_id} (30개 단위 저장 기준)")
            cluster_name = input(f"이 클러스터 [{cluster_id}]의 색상 이름을 입력하세요: ")
            cluster_names[cluster_id] = cluster_name
        else:
            cluster_name = cluster_names[cluster_id]

        # CSV에 저장
        with open(csv_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([file_path, cluster_id, cluster_name])

        print(f"✅ 저장 완료: {file_path} | Cluster {cluster_id}: {cluster_name}")

cap.release()
cv2.destroyAllWindows()
