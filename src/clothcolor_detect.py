import cv2
import numpy as np
import csv
import os

def save_color_and_image(colors, roi_img, index, filename='colors.csv', image_dir='images'):
    os.makedirs(image_dir, exist_ok=True)
    image_path = os.path.join(image_dir, f"color_{index:02d}.jpg")
    cv2.imwrite(image_path, roi_img)

    file_exists = os.path.isfile(filename)
    with open(filename, mode='a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(['B', 'G', 'R', 'image_path'])
        for color in colors:
            writer.writerow(color.tolist() + [image_path])
    print(f"✅ [{index}] 색상 및 이미지 저장 완료: {image_path}")

# 초기 설정
img1 = None
win_name = 'Camera Matching'
K = 1  # 대표 색상 개수
max_save_count = 30
save_count = 0
csv_path = '../colors.csv'
image_folder = '../img'

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    res = frame.copy()

    if img1 is not None:
        data = img1.reshape((-1, 3)).astype(np.float32)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10 , 1.0)
        ret, label, center = cv2.kmeans(data, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
        center = np.uint8(center)

    cv2.imshow(win_name, res)
    key = cv2.waitKey(1) & 0xFF

    # ESC 종료
    if key == 27:
        break

    # space: ROI 저장
    elif key == ord(' '):
        if save_count < max_save_count:
            x, y, w, h = cv2.selectROI(win_name, frame, False)
            if w and h:
                img1 = frame[y:y+h, x:x+w]
                print(f"[{save_count+1}/{max_save_count}] ROI 선택됨: ({x}, {y}, {w}, {h})")

                data = img1.reshape((-1, 3)).astype(np.float32)
                ret, label, center = cv2.kmeans(data, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
                center = np.uint8(center)
                save_color_and_image(center, img1, save_count + 1, filename=csv_path, image_dir=image_folder)
                save_count += 1
        else:
            print("❌ 저장 횟수 초과 (30회). 더 이상 저장할 수 없습니다.")

    # d: 디텍션 모드
    elif key == ord('d'):
        print("🎯 디텍션 모드 시작")

        if not os.path.isfile(csv_path):
            print("❌ CSV 파일 없음.")
            continue

        # 색상 불러오기
        colors = []
        with open(csv_path, newline='', encoding='utf-8') as f:
            reader = csv.reader(f)
            next(reader)  # 헤더 건너뛰기
            for row in reader:
                if len(row) >= 4:
                    bgr = list(map(int, row[:3]))
                    colors.append(np.array(bgr, dtype=np.uint8))

        print(f"🎨 불러온 색상 수: {len(colors)}개")

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            output = frame.copy()

            for color in colors:
                lower = np.clip(color - 20, 0, 255)
                upper = np.clip(color + 20, 0, 255)

                mask = cv2.inRange(frame, lower, upper)
                result = cv2.bitwise_and(output, output, mask=mask)

                contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                for cnt in contours:
                    if cv2.contourArea(cnt) > 500:
                        x, y, w, h = cv2.boundingRect(cnt)
                        cv2.rectangle(output, (x, y), (x+w, y+h), (0, 255, 0), 2)

            cv2.imshow("Detection Mode", output)
            k = cv2.waitKey(1) & 0xFF
            if k == 27:
                print("🛑 디텍션 모드 종료")
                break

cap.release()
cv2.destroyAllWindows()