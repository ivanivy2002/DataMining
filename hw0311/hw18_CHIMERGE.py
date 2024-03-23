# hw18_CHIMERGE.py
import numpy as np
from sklearn import datasets

iris = datasets.load_iris()
X = iris.data
y = iris.target

DEBUG = False


# 数据集的长度: 150
# ChiMerge算法

def ChiMerge(attr, lbs, max_intervals=6):
    # 首先对属性值和标签按属性值进行排序
    sorted_idx = np.argsort(attr)  # 返回排序后的索引
    attr = attr[sorted_idx]
    lbs = lbs[sorted_idx]

    # 计算每个区间的频数
    intv_freq = {}  # interval frequency
    L = len(attr)
    for i in range(L):
        intv_lb = (attr[i],)  # interval label # 以元组的形式存储区间
        if intv_lb in intv_freq:  # 如果区间已经存在, 则频数加1
            intv_freq[intv_lb][lbs[i]] = intv_freq[intv_lb].get(lbs[i], 0) + 1
        else:
            intv_freq[intv_lb] = {lbs[i]: 1}
    # print(intv_freq)
    # 合并相邻区间直到满足停止条件
    while len(intv_freq) > max_intervals:
        if DEBUG:
            print(intv_freq)
        chi_values = {}
        # 计算相邻区间的卡方值
        # for i in range(L - 1):
        for i in range(len(intv_freq) - 1):
            intv_lb1 = list(intv_freq.keys())[i]
            intv_lb2 = list(intv_freq.keys())[i + 1]
            freq1 = intv_freq[intv_lb1] if intv_lb1 in intv_freq else {0: 0, 1: 0, 2: 0}
            freq2 = intv_freq[intv_lb2] if intv_lb2 in intv_freq else {0: 0, 1: 0, 2: 0}

            # 计算卡方值
            total_freq = {k: freq1.get(k, 0) + freq2.get(k, 0) for k in set(freq1) | set(freq2)}
            expected_freq = {k: (freq1.get(k, 0) + freq2.get(k, 0)) * sum(total_freq.values()) / L
                             for k in total_freq}
            # 处理期望频率为零的情况
            expected_freq = {k: v if v != 0 else 1e-10 for k, v in expected_freq.items()}
            chi_values[(intv_lb1, intv_lb2)] = sum(
                {k: ((freq1.get(k, 0) - expected_freq.get(k, 0)) ** 2) / expected_freq.get(k, 1) for k in
                 total_freq}.values())

        # 找到最小卡方值的相邻区间
        min_chi_pair = min(chi_values, key=chi_values.get)
        min_chi_value = chi_values[min_chi_pair]
        l_b = min_chi_pair[0]
        u_b = min_chi_pair[1]
        # 合并最小卡方值的相邻区间
        intv_freq[l_b] = {
            k: intv_freq[l_b].get(k, 0) + intv_freq[u_b].get(k, 0) for k in
            set(intv_freq[l_b]) | set(intv_freq[u_b])}
        del intv_freq[u_b]
        if DEBUG:
            print(f"Merge {min_chi_pair} with chi value {min_chi_value}")
        # 在合并区间后，更新attr字典

    # 计算分裂点和最终的区间
    split_points = []
    intervals = []
    for key in intv_freq:
        split_points.append(key[0])

    for i in range(len(split_points) - 1):
        intervals.append((split_points[i], split_points[i + 1]))
    intervals.append((split_points[-1], split_points[-1]))
    return split_points, intervals


# 对每个数值属性应用 ChiMerge 方法
num_attributes = X.shape[1]
intervals = {}  # 存储每个属性的最终区间
split_points = {}  # 存储每个属性的分裂点

for ia in range(num_attributes):
    attribute_values = X[:, ia]  # 取出第i个属性的所有值
    labels = y
    split_point, interval = ChiMerge(attribute_values, labels)
    split_points[ia] = split_point
    intervals[ia] = interval

# 打印结果
for ia in range(num_attributes):
    print(f"Attribute {ia + 1}:")
    print(f"Split Point: {split_points[ia]}")
    print(f"Intervals: {intervals[ia]}")
