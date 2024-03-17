import utils as u

data = {
    'x1': (1.5, 1.7),
    'x2': (2, 1.9),
    'x3': (1.6, 1.8),
    'x4': (1.2, 1.5),
    'x5': (1.5, 1.0)
}
# 新数据
x = (1.4, 1.6)
# 基于不同的距离度量方法计算相似度排序
# 欧几里得距离
similarity_euclidean = {}
for k, v in data.items():
    similarity_euclidean[k] = (u.minkowski_distance(v, x, 2))
    sorted(similarity_euclidean.items(), key=lambda item: item[1])
print("欧几里得距离：", similarity_euclidean)
