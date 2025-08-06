import cv2
import os
import csv
import numpy as np
from collections import Counter

# -----------------------------
# HSV 히스토그램 특징 추출 함수
def extract_feature(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1, 2], None, [8, 8, 8],
                        [0, 180, 0, 256, 0, 256])
    hist = cv2.normalize(hist, hist).flatten()
    return hist

# -----------------------------
# KNN 분류 함수
def knn_classify(feature, train_features, train_labels, k=3):
    if len(train_features) == 0:
        return None
    distances = np.linalg.norm(train_features - feature, axis=1)
    knn_indices = distances.argsort()[:k]
    knn_labels = train_labels[knn_indices]
    vote = Counter(knn_labels)
    return vote.most_common(1)[0][0]

# -----------------------------
# 기본 경로 설정
csv_file = 'color_labels.csv'
save_folder = '../cloth'
os.makedirs(save_folder, exist_ok=True)

clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

# -----------------------------
# CSV 헤더 초기화
if not os.path.exists(csv_file):
    with open(csv_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['image_path', 'cluster_id', 'cluster_name'])

# -----------------------------
# CSV 로드: 클러스터 정보 및 학습 데이터 준비
cluster_names = {}
train_features = []
train_labels = []

with open(csv_file, 'r', newline='') as f:
    reader = csv.reader(f)
    next(reader, None)
    for row in reader:
        if len(row) == 3:
            path, cid, cname = row
            cid = int(cid)
            cluster_names[cid] = cname
            if os.path.exists(path):
                img = cv2.imread(path)
                feat = extract_feature(img)
                train_features.append(feat)
                train_labels.append(cid)

train_features = np.array(train_features)
train_labels = np.array(train_labels)

# -----------------------------
# 웹캠 초기화
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

    # ROI 사각형
    cv2.rectangle(frame, (x, y), (x + roi_size, y + roi_size), (0, 255, 0), 2)
    cv2.imshow('Video', frame)

    key = cv2.waitKey(1)
    if key == ord('q'):
        break

    elif key == ord(' '):  # 이미지 저장 및 클러스터 이름 입력
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
        img1 = cv2.cvtColor(lab_eq, cv2.COLOR_LAB2BGR)
        img1 = cv2.GaussianBlur(img1, (5, 5), 0)

        # 이미지 저장
        existing_files = [f for f in os.listdir(save_folder) if f.endswith('.jpg')]
        next_number = len(existing_files) + 1
        file_path = os.path.join(save_folder, f"{next_number}.jpg")
        cv2.imwrite(file_path, img1)

        # 클러스터 ID 계산
        cluster_id = (next_number - 1) // 30
        if cluster_id not in cluster_names:
            print(f"\n🟢 새 클러스터 ID 감지됨: {cluster_id}")
            cname = input(f"클러스터 [{cluster_id}]의 색상 이름을 입력하세요: ")
            cluster_names[cluster_id] = cname
        else:
            cname = cluster_names[cluster_id]

        # CSV 기록
        with open(csv_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([file_path, cluster_id, cname])

        # 학습 데이터에 즉시 반영
        feature = extract_feature(img1)
        train_features = np.append(train_features, [feature], axis=0)
        train_labels = np.append(train_labels, cluster_id)

        print(f"✅ 저장 완료: {file_path} | Cluster {cluster_id}: {cname}")

    elif key == ord('d'):  # KNN으로 클러스터 분류
        roi = frame[y:y + roi_size, x:x + roi_size]
        lab = cv2.cvtColor(roi, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        l_eq = clahe.apply(l)
        lab_eq = cv2.merge((l_eq, a, b))
        img1 = cv2.cvtColor(lab_eq, cv2.COLOR_LAB2BGR)
        img1 = cv2.GaussianBlur(img1, (5, 5), 0)

        roi_feature = extract_feature(img1)

        if len(train_features) > 0:
            cluster_id = knn_classify(roi_feature, train_features, train_labels)
            cname = cluster_names.get(cluster_id, "Unknown")
            result_text = f"Cluster: {cluster_id} ({cname})"
            print(f"🎯 인식된 클러스터 → {result_text}")
        else:
            result_text = "No training data"
            print("❌ 학습 데이터가 없습니다.")

        display_img = img1.copy()
        cv2.putText(display_img, result_text, (5, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        cv2.imshow('Processed ROI', display_img)

cap.release()
cv2.destroyAllWindows()

