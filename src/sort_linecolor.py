# 3채널 컬러 영상은 하나의 색상을 위해서 24비트(8X3)
# 16777216가지의 색상을 표현 가능

# 모든색을 다 사용하지 않고 비슷한 그룹의 색상을 지어서 같은 색상으로 처리
# 처리 용량 간소화

import numpy as np
import cv2
import matplotlib.pyplot as plt

# 기준 색상 (BGR)
Y = np.array([209, 252, 255], dtype=np.float32)
B = np.array([127, 125, 114], dtype=np.float32)
W = np.array([235, 238, 236], dtype=np.float32)
target_colors = np.vstack((Y, B, W))
color_names = ['Yellow', 'Black', 'White']

# 이미지 불러오기 및 리사이즈
img = cv2.imread('../img/load_line.jpg')
resized = cv2.resize(img, (640, 480))
data = resized.reshape((-1, 3)).astype(np.float32)

# k-means 조건
K = 20
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10 , 1.0)

# k-means 실행
ret, labels, centers = cv2.kmeans(data, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

# centers와 target_colors 간 거리 계산
dist_matrix = np.linalg.norm(centers[:, None] - target_colors[None, :], axis=2)
closest_target_idx = np.argmin(dist_matrix, axis=1)

# 클러스터별 픽셀 수
unique, counts = np.unique(labels, return_counts=True)

# 각 target_color에 해당하는 픽셀 수 누적
pixel_counts = np.zeros(len(target_colors), dtype=int)
for cluster_idx, count in zip(unique, counts):
    target_idx = closest_target_idx[cluster_idx]
    pixel_counts[target_idx] += count

# 전체 대비 비율
total_pixels = data.shape[0]
pixel_ratios = pixel_counts / total_pixels

# 가장 가까운 클러스터 색상 (BGR → RGB for plotting)
cluster_colors_bgr = [centers[closest_target_idx == i][0] for i in range(3)]
cluster_colors_rgb = [c[::-1]/255 for c in cluster_colors_bgr]  # BGR → RGB → 0~1

# 중심값 정수형 변환 및 결과 이미지 만들기
center = np.uint8(centers)
res = center[labels.flatten()].reshape(resized.shape)
res_rgb = cv2.cvtColor(res, cv2.COLOR_BGR2RGB)
resized_rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
merged_rgb = np.hstack((resized_rgb, res_rgb))  # 좌우 합치기

# ------------------------------
# 하나의 matplotlib 창에 모든 시각화
# ------------------------------
plt.figure(figsize=(15, 6))

# subplot 1: 이미지 비교
plt.subplot(1, 3, 1)
plt.imshow(merged_rgb)
plt.axis('off')
plt.title('Original (Left) | KMeans (Right)')

# subplot 2: 도넛 차트 (실제 클러스터 색상 사용)
plt.subplot(1, 3, 2)
plt.pie(pixel_ratios, labels=color_names, colors=cluster_colors_rgb,
        autopct='%.1f%%', startangle=90, counterclock=False, wedgeprops=dict(width=0.4))
plt.title('Color Proportion by Cluster')

# subplot 3: 막대 그래프 (실제 클러스터 색상 사용)
plt.subplot(1, 3, 3)
bars = plt.bar(range(3), pixel_counts, color=cluster_colors_rgb)
plt.xticks(range(3), color_names)
plt.ylabel("Pixel Count")
plt.title("Pixel Count for Each Target Color")
plt.grid(axis='y', linestyle='--', alpha=0.5)

plt.tight_layout()
plt.show()