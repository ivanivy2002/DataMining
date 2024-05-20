import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, dendrogram

# 定义距离矩阵
# distance_matrix = np.array([
#     [0.0000, 0.2357, 0.2218, 0.3688, 0.3421, 0.2347],
#     [0.2357, 0.0000, 0.1483, 0.2042, 0.1388, 0.2540],
#     [0.2218, 0.1483, 0.0000, 0.1513, 0.2843, 0.1100],
#     [0.3688, 0.2042, 0.1513, 0.0000, 0.2932, 0.2216],
#     [0.3421, 0.1388, 0.2843, 0.2932, 0.0000, 0.3921],
#     [0.2347, 0.2540, 0.1100, 0.2216, 0.3921, 0.0000]
# ])
similarity_matrix = [
    [1.00, 0.10, 0.41, 0.55, 0.35],
    [0.10, 1.00, 0.64, 0.47, 0.98],
    [0.41, 0.64, 1.00, 0.44, 0.85],
    [0.55, 0.47, 0.44, 1.00, 0.76],
    [0.35, 0.98, 0.85, 0.76, 1.00]
]
distance_matrix = 1 - np.array(similarity_matrix)

n = len(distance_matrix)
labels = [f"p{i}" for i in range(1, n + 1)]

# scipy的linkage方法需要一维的距离向量
condensed_distance_matrix = distance_matrix[np.triu_indices(n, 1)]

# method = 'single'
method = 'complete'
# 进行单链聚类
Z = linkage(condensed_distance_matrix, method=method)

# 绘制树状图
plt.figure(figsize=(10, 7))
dendrogram(Z, labels=labels)
plt.title("Single Linkage Dendrogram")
plt.xlabel("Points")
plt.ylabel("Distance")
plt.show()
