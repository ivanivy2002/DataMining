# hw19_TIME.py
import numpy as np
from sklearn import datasets
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
import time

# 定义debugger宏
# DEBUG = True
DEBUG = False


def dist(a, b):
    return np.linalg.norm(a - b)


iris = datasets.load_iris()
X = iris.data
# 数据集的长度: 150
# 选取一个cluster centroid
# 1. 随机选择一组点作为cluster centroid, 假设是k个
p = 0.03  # 选取的比例
k = int(p * len(X))
if DEBUG:
    print(k)
# 划分X, 选出k个点作为cluster centroid
C_idx = np.random.choice(range(len(X)), k, replace=False)
C = X[C_idx]
if DEBUG:
    print(C)
# 将原来的数据集去除cluster centroid 作为新的数据集
X = np.delete(X, C_idx, axis=0)
n = len(X)


# 目标: 对每个点找到最近的C中的点
def kmeans_clustering(X, C):
    # a. 直接使用kmeans
    start_time = time.time()
    kmeans = KMeans(n_clusters=k, init=C, n_init=1).fit(X)
    # 找到每个数据点最近的簇中心
    nearest_centers = kmeans.predict(X)
    end_time = time.time()
    time_kmeans = end_time - start_time
    # 如果DEBUG为True, 则打印出最近的簇中心
    if DEBUG:
        print(nearest_centers)
    return time_kmeans


def brute(X, C):
    # b. 直接遍历
    start_time = time.time()
    nearest_centers = []
    for i in range(n):
        min_dist = np.inf
        min_idx = -1
        for j in range(k):
            d = dist(X[i], C[j])
            if d < min_dist:
                min_dist = d
                min_idx = j
        nearest_centers.append(min_idx)
    end_time = time.time()
    time_brute = end_time - start_time
    if DEBUG:
        print(nearest_centers)
    return time_brute


def triangle_inequality(X, C):
    # c. 使用三角不等式加快计算
    # 预先计算好C内所有对
    d = np.zeros((k, k))
    for i in range(k):
        for j in range(i + 1, k):
            d[j][i] = d[i][j] = dist(C[i], C[j])
            # print(f"{i} {j} {d[i][j]} {d[j][i]}")
    start_time = time.time()
    nearest_centers = []
    for i in range(n):
        min_dist = np.inf
        min_idx = 0
        for j in range(k):
            # 使用三角不等式
            if d[min_idx][j] > 2 * min_dist:
                continue
            tmp = dist(X[i], C[j])
            if tmp < min_dist:
                min_dist = tmp
                min_idx = j
        nearest_centers.append(min_idx)
    end_time = time.time()
    time_triangle = end_time - start_time
    if DEBUG:
        print(nearest_centers)
    return time_triangle


print(f"KMeans: {kmeans_clustering(X, C)}")
print(f"遍历: {brute(X, C)}")
print(f"三角不等式: {triangle_inequality(X, C)}")


def cdist_clustering(X, C):
    # d. 使用cdist
    start_time = time.time()
    d = cdist(X, C)
    nearest_centers = np.argmin(d, axis=1)
    end_time = time.time()
    time_cdist = end_time - start_time
    if DEBUG:
        print(nearest_centers)
    return time_cdist


print(f"cdist: {cdist_clustering(X, C)}")
