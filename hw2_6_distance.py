import numpy as np
import utils as u


def minkowski_distance(tuple1, tuple2, h):
    return np.power(np.sum(np.power(np.abs(np.array(tuple1) - np.array(tuple2)), h)), 1 / h)


# 元组
t1 = (22, 1, 42, 10)
t2 = (20, 0, 36, 8)

euclidean_distance = minkowski_distance(t1, t2, 2)
manhattan_distance = minkowski_distance(t1, t2, 1)
q = 3
minkowski_distance = minkowski_distance(t1, t2, q)
# 上确界距离
upper_bound = np.max(np.abs(np.array(t1) - np.array(t2)))
print("欧式距离：", u.format_nf(euclidean_distance, 4))
print("曼哈顿距离：", manhattan_distance)
print("闵可夫斯基距离：", u.format_nf(minkowski_distance, 4))
print("上确界距离：", upper_bound)
