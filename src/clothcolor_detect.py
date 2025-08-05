import cv2
import numpy as np
import csv
import os

def save_colors_to_csv(colors, filename='colors.csv'):
    file_exists = os.path.isfile(filename)
    with open(filename, mode='a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(['B', 'G', 'R'])
        for color in colors:
            writer.writerow(color.tolist())
    print(f"CSV 파일 '{filename}'에 색상 추가 저장 완료!")

def load_colors_from_csv(filename='colors.csv'):
    colors = []
    if os.path.isfile(filename):
        with open(filename, newline='', encoding='utf-8') as f:
            reader = csv.reader(f)
            next(reader)  # 헤더 건너뜀
            for row in reader:
                b, g, r = map(int, row)
                colors.append([b, g, r])
    return np.array(colors, dtype=np.uint8)

def knn_predict(train_colors, query_color, k=1):
    distances = np.linalg.norm(train_colors - query_color, axis=1)
    nearest_idx = np.argpartition(distances, k-1)[:k]
    return nearest_idx[0]

# --- 색상 인덱스를 사람이 보기 쉽게 매핑 ---
color_labels = ['카키', '하얀', '검정']  # 저장된 순서에 맞게!

# 초기 설정
img1 = None
img2 = None
win_name = 'Camera Matching'
K = 1
max_save_count = 30
save_count = 0

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

train_colors = load_colors_from_csv('../colors.csv')

criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10 , 1.0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    res = frame.copy()
    cv2.imshow(win_name, res)
    key = cv2.waitKey(1) & 0xFF

    if key == 27:  # ESC
        break

    elif key == ord(' '):  # ROI 선택 + 저장
        if save_count < max_save_count:
            x, y, w, h = cv2.selectROI(win_name, frame, False)
            if w and h:
                img1 = frame[y:y+h, x:x+w]
                data = img1.reshape((-1,3)).astype(np.float32)
                ret, label, center = cv2.kmeans(data, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
                center = np.uint8(center)
                save_colors_to_csv(center, '../colors.csv')

                # CSV 갱신
                train_colors = load_colors_from_csv('../colors.csv')
                save_count += 1
                print(f"[{save_count}/{max_save_count}] ROI 저장 완료.")
        else:
            print("⚠️ 저장 횟수 초과: 더 이상 저장되지 않습니다.")

    elif key == ord('d'):  # ROI 선택 후 예측
        if len(train_colors) > 0:
            x, y, w, h = cv2.selectROI(win_name, frame, False)
            if w and h:
                img2 = frame[y:y+h, x:x+w]
                data = img2.reshape((-1,3)).astype(np.float32)
                ret, label, center = cv2.kmeans(data, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
                center = np.uint8(center)
                print(f"추출된 대표 색상 (BGR): {center[0]}")

                pred_idx = knn_predict(train_colors, center[0])
                nearest_color = train_colors[pred_idx]

                if pred_idx < len(color_labels):
                    predicted_label = color_labels[pred_idx]
                else:
                    predicted_label = f"Unknown (index={pred_idx})"

                print(f"🎯 예측 결과: {predicted_label} (인덱스 {pred_idx}, 색상 {nearest_color})")
        else:
            print("⚠️ 학습 데이터가 없습니다. 먼저 색상을 저장하세요.")

cap.release()
cv2.destroyAllWindows()
