# hw2_8_similarity.py
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
print("(a)")
similarity_euclidean = {}
for k, v in data.items():
    similarity_euclidean[k] = (u.format_nf(u.minkowski_distance(v, x, 2), 4))
similarity_euclidean = sorted(similarity_euclidean.items(), key=lambda item: item[1])
u.print_2i_list(similarity_euclidean, "欧氏距离排序：")
# 曼哈顿距离
similarity_manhattan = {}
for k, v in data.items():
    similarity_manhattan[k] = (u.format_nf(u.minkowski_distance(v, x, 1), 1))
similarity_manhattan = sorted(similarity_manhattan.items(), key=lambda item: item[1])
u.print_2i_list(similarity_manhattan, "曼哈顿距离排序：")
# 上确界距离
similarity_upper_bound = {}
for k, v in data.items():
    similarity_upper_bound[k] = u.format_nf(u.upper_bound_distance(v, x), 1)
similarity_upper_bound = sorted(similarity_upper_bound.items(), key=lambda item: item[1])
u.print_2i_list(similarity_upper_bound, "上确界距离排序：")
# 余弦相似度
similarity_cosine = {}
for k, v in data.items():
    similarity_cosine[k] = u.format_nf(u.cosine_similarity(v, x), 5)
similarity_cosine = sorted(similarity_cosine.items(), key=lambda item: item[1], reverse=True)
u.print_2i_list(similarity_cosine, "余弦相似度排序：")

print("(b)")
# 让每个数据的范数为1
data_normalized = {}
for k, v in data.items():
    data_normalized[k] = u.normalize(v)
# 新数据归一化
x_normalized = u.normalize(x)
print("归一化数据：")
for k, v in data_normalized.items():
    print(k, v)
print("归一化新数据：", x_normalized)
sim_euclidean_normalized = {}
for k, v in data_normalized.items():
    sim_euclidean_normalized[k] = u.format_nf(u.minkowski_distance(v, x_normalized, 2), 5)
sim_euclidean_normalized = sorted(sim_euclidean_normalized.items(), key=lambda item: item[1])
u.print_2i_list(sim_euclidean_normalized, "归一化欧氏距离排序：")
