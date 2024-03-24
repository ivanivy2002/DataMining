# hw19_CENTROIDS.py
import numpy as np
from sklearn import datasets
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
import time

# 定义debugger宏
DEBUG = False

iris = datasets.load_iris()
X = iris.data
# 数据集的长度: 150
# 选取一个cluster centroid
# 1. 随机选择一组点作为cluster centroid, 假设是k个
p = 0.03  # 选取的比例
k = int(p * len(X))


def dist(a, b):
    return np.linalg.norm(a - b)


# 目标: 对每个点找到最近的C中的点
def ori_kmeans(X, k, max=200):
    # a. 直接使用kmeans
    kmeans = KMeans(n_clusters=k, max_iter=max).fit(X)
    lb = kmeans.labels_
    cent = kmeans.cluster_centers_
    # 找到每个数据点最近的簇中心
    nearest_centers = kmeans.predict(X)
    # 如果DEBUG为True, 则打印出最近的簇中心
    if DEBUG:
        print(nearest_centers)
    return cent, lb


def tri_kmeans(X, k, max=200):
    n = len(X)
    cent = X[np.random.choice(range(n), k, replace=False)]


def triangle_inequality(X, C):
    # c. 使用三角不等式加快计算
    # 预先计算好C内所有对
    d = np.zeros((k, k))
    for i in range(k):
        for j in range(i + 1, k):
            d[i][j] = dist(C[i], C[j])
            # print(f"{i} {j} {d[i][j]}")
    start_time = time.time()
    nearest_centers = []
    for i in range(n):
        min_dist = np.inf
        min_idx = 0
        for j in range(k):
            # 使用三角不等式
            if d[min_idx][j] > 2 * min_dist:
                continue
            dist = np.linalg.norm(X[i] - C[j])
            if dist < min_dist:
                min_dist = dist
                min_idx = j
        nearest_centers.append(min_idx)
    end_time = time.time()
    time_triangle = end_time - start_time
    if DEBUG:
        print(nearest_centers)
    return time_triangle


ori_cent, ori_lb = ori_kmeans(X, k)
# tri_cent, tri_lb = tri_kmeans(X, k)
print(f"ori_cent:\n {ori_cent}")
# print(f"tri_cent: {tri_cent}")
# print(f"三角不等式: {triangle_inequality(X, k)}")
