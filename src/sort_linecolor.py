# 3채널 컬러 영상은 하나의 색상을 위해서 24비트(8X3)
# 16777216가지의 색상을 표현 가능

# 모든색을 다 사용하지 않고 비슷한 그룹의 색상을 지어서 같은 색상으로 처리
# 처리 용량 간소화

import numpy as np
import cv2

# 차선도로 색 지정
Y = (209,252,255)
B = (127,125,114)
W = (235,238,236)

# Y, B, W를 병합 ---③
data = np.vstack((Y,B,W)).astype(np.float32)
# K = 18 # 군집화 개수
K = 8
img = cv2.imread('../img/load_line.jpg')
resized = cv2.resize(img, (640, 480))
data = resized.reshape((-1, 3)).astype(np.float32)
# k-means는 2차원 배열만 처리할수있으므로 reshape으로 2차원배열로 변환 
# EX) img.shape = (480, 640, 3)인 3차원이므로 reshape으로 바꿈
# 데이터를 잃으면 안되므로 -1을 넣어 자동으로 480X640X3/3을 계산해 데이터를 유지
# 데이터 평균을 구할때 소수점 이하값을 가질수 있으므로 변환
# 반복 중지 조건
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10 , 1.0)

# 평균 클러스터링 적용
ret, label, center = cv2.kmeans(data, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

# 중심값을 정수형으로 변환

center = np.uint8(center)
print(center)

# 각레이블에 해당하는 중심값을 픽셀 값 선택
res = center[label.flatten()]
# 원본 영상의 형태로 변환
res = res.reshape((resized.shape))

# 결과 출력
merged = np.hstack((resized, res))
cv2.imshow('Kmeans color', merged)
cv2.waitKey(0)
cv2.destroyAllWindows()