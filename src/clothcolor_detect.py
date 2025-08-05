import cv2
import numpy as np
import csv

def save_colors_to_csv(colors, filename='colors.csv'):
    """
    colors: np.ndarray, shape=(K, 3) -- BGR 색상 배열
    """
    file_exists = os.path.isfile(filename)

    with open(filename, mode='a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        
        # 파일이 없을 때만 헤더 작성
        if not file_exists:
            writer.writerow(['B', 'G', 'R'])
        
        for color in colors:
            writer.writerow(color.tolist())

    print(f"CSV 파일 '{filename}'에 색상 추가 저장 완료!"))

# 예시 사용법
# save_colors_to_csv(center, 'colors.csv')


# 초기 설정
img1 = None
win_name = 'Camera Matching'
K = 1  # 대표 색상 개수

cap = cv2.VideoCapture(0)              
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

while cap.isOpened():       
    ret, frame = cap.read() 
    if not ret:
        break
        
    if img1 is None:
        res = frame.copy()
    else:
        # 1. K-means를 위한 전처리
        data = img1.reshape((-1, 3)).astype(np.float32)

        # 2. K-means clustering
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10 , 1.0)        
        ret, label, center = cv2.kmeans(data, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

        center = np.uint8(center)
        save_colors_to_csv(center,'../colors.csv')
        
    # 결과 출력
    cv2.imshow(win_name, res)
    key = cv2.waitKey(1) & 0xFF
    
    if key == 27:
        break          
    elif key == ord(' '):
        x, y, w, h = cv2.selectROI(win_name, frame, False)
        if w and h:
            img1 = frame[y:y+h, x:x+w]
            print("ROI 선택됨: (%d, %d, %d, %d)" % (x, y, w, h))

cap.release()                          
cv2.destroyAllWindows()



