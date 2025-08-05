import cv2
import csv
import os

# 저장 경로 및 CSV 설정
save_dir = '../cloth'
csv_path = '../colors.csv'

# 디렉터리 없으면 생성
os.makedirs(save_dir, exist_ok=True)

# CSV 헤더 작성 (없을 때만)
if not os.path.isfile(csv_path):
    with open(csv_path, mode='w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['image_path'])

# 초기 변수
img1 = None
win_name = 'Camera Matching'
save_count = 0
max_save_count = 30

# 카메라 초기화
cap = cv2.VideoCapture(0)              
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

while cap.isOpened():       
    ret, frame = cap.read() 
    if not ret:
        break

    # ROI 선택 전 상태
    res = frame.copy()

    cv2.imshow(win_name, res)
    key = cv2.waitKey(1) & 0xFF

    if key == 27:  # ESC 키
        break          
    
    elif key == ord(' '):  # 스페이스바 누를 때 ROI 저장
        if save_count < max_save_count:
            x, y, w, h = cv2.selectROI(win_name, frame, False)
            if w and h:
                img1 = frame[y:y+h, x:x+w]
                img_name = f"roi_{save_count+1:02d}.jpg"
                img_path = os.path.join(save_dir, img_name)

                # 이미지 저장
                cv2.imwrite(img_path, img1)

                # CSV에 경로 저장
                with open(csv_path, mode='a', newline='', encoding='utf-8') as f:
                    writer = csv.writer(f)
                    writer.writerow([img_path])

                print(f"[{save_count+1}/{max_save_count}] ROI 저장됨: {img_path}")
                save_count += 1
        else:
            print("⚠️ 저장 횟수 초과: 더 이상 저장되지 않습니다.")

cap.release()                          
cv2.destroyAllWindows()