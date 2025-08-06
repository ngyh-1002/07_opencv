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

        # 🎯 조명에 강한 정규화
        # LAB 색공간은 명도(L)를 분리해서 조정 가능.
        # CLAHE는 밝은 부분과 어두운 부분을 모두 보정해서 조명에 강한 색상 보정 효과가 있어요.
        # RGB에서 바로 조정하면 색상 왜곡이 생길 수 있음.
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
    elif key == ord('d'):  # d 키: ROI 처리 후 KNN 분류 수행
        roi = frame[y:y + roi_size, x:x + roi_size]

        # 🎯 조명에 강한 정규화
        # LAB 색공간은 명도(L)를 분리해서 조정 가능.
        # CLAHE는 밝은 부분과 어두운 부분을 모두 보정해서 조명에 강한 색상 보정 효과가 있어요.
        # RGB에서 바로 조정하면 색상 왜곡이 생길 수 있음.
        # CLAHE + 블러 처리 (저장된 이미지와 동일한 전처리)
        lab = cv2.cvtColor(roi, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        l_eq = clahe.apply(l)
        lab_eq = cv2.merge((l_eq, a, b))
        processed_roi = cv2.cvtColor(lab_eq, cv2.COLOR_LAB2BGR)
        processed_roi = cv2.GaussianBlur(processed_roi, (5, 5), 0)

        # ROI 이미지로부터 특징 추출
        roi_feature = extract_feature(processed_roi)

        # 학습 데이터가 있을 경우 KNN 분류 수행
        if train_features is not None and len(train_features) > 0:
            cluster_label = knn_classify(roi_feature, train_features, train_labels, k=3)
            result_text = f"Cluster: {cluster_label}"
            print(f"ROI 이미지 분류 결과 -> {result_text}")
        else:
            result_text = "No training data"
            print("학습 데이터가 없어 분류할 수 없습니다.")

        # 결과 텍스트를 ROI 이미지에 오버레이하고 표시
        display_img = processed_roi.copy()
        cv2.putText(display_img, result_text, (5, 20), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (0, 0, 255), 2, cv2.LINE_AA)
        cv2.imshow('Processed ROI', display_img)

cap.release()
cv2.destroyAllWindows()
