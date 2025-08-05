import numpy as np, cv2
import matplotlib.pyplot as plt

# 0~150 임의의 2수, 25개 ---①
a = np.random.randint(0,150,(25,2))
# 128~255 임의의 2수, 25개  ---②
b = np.random.randint(128, 255,(25,2))
# a, b를 병합 ---③
data = np.vstack((a,b)).astype(np.float32)
# 중지 요건 ---④
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
# 평균 클러스터링 적용 ---⑤
# data : 처리 대상 데이터
# K : 원하는 묶음개수
# 결과 데이터 
# 반복 종료 조건
# 매번 다른 초기 레이블로 실행할 횟수
# 초기 중앙점
ret,label,center=cv2.kmeans(data,2,None,criteria,10,cv2.KMEANS_RANDOM_CENTERS)

# centers와 target_colors 간 거리 계산 (Kx3 행렬)
dist_matrix = np.linalg.norm(centers[:, None] - target_colors[None, :], axis=2)

# 각 클러스터 중심에 가장 가까운 target color 인덱스 구하기
closest_target_idx = np.argmin(dist_matrix, axis=1)

# 클러스터 인덱스별 픽셀 수 세기
unique, counts = np.unique(labels, return_counts=True)

# target_colors별 픽셀 수 누적
pixel_counts = np.zeros(len(target_colors), dtype=int)
for cluster_idx, count in zip(unique, counts):
    target_idx = closest_target_idx[cluster_idx]
    pixel_counts[target_idx] += count

total_pixels = data.shape[0]
pixel_ratios = pixel_counts / total_pixels

# 결과 출력
print("색상별 픽셀 수와 비율:")
for i, (count, ratio) in enumerate(zip(pixel_counts, pixel_ratios)):
    print(f"{color_names[i]}: 픽셀 수 = {count}, 비율 = {ratio:.4f}")

# 시각화
plt.figure(figsize=(6,6))
plt.pie(pixel_ratios, labels=color_names, colors=target_colors/255,
        autopct='%.1f%%', startangle=90, counterclock=False)
plt.title('이미지 내 3가지 기준 색상 비율')
plt.show()