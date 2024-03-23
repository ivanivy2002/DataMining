import numpy as np


# 定义函数，将数字格式化为整数
def format_integer(num):
    return "{:.0f}".format(num)


# 定义函数，将数字格式化为2位小数
def format_2f(num):
    return "{:.2f}".format(num)


# 将整个数组都格式化为2位小数
def format_nf_list(li, n):
    formatted = [format_nf(item, n) for item in li]
    formatted = [float(item) for item in formatted]
    return formatted

def z_score_normalize(data):
    mean = np.mean(data)
    std = np.std(data)
    return [(item - mean) / std for item in data]


def format_nf(num, n):
    return "{:.{}f}".format(num, n)


import matplotlib.pyplot as plt


def plot_box(data, labels, title):
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.boxplot(data, patch_artist=True, labels=labels)
    plt.title(title)
    plt.show()


def plot_scatter(x, y, title, xl, yl):
    plt.scatter(x, y)
    plt.title(title)
    plt.xlabel(xl)
    plt.ylabel(yl)
    plt.show()


def minkowski_distance(tuple1, tuple2, h):
    return np.power(np.sum(np.power(np.abs(np.array(tuple1) - np.array(tuple2)), h)), 1 / h)


def upper_bound_distance(t1, t2):
    return np.max(np.abs(np.array(t1) - np.array(t2)))


def cosine_similarity(t1, t2):
    return np.dot(t1, t2) / (np.linalg.norm(t1) * np.linalg.norm(t2))


def normalize(x):
    # 使范数为1
    return x / np.linalg.norm(x)


def print_2i_list(li, text):
    print(text)
    for item in li:
        print(f"{item[0]}: {item[1]}")
    print()


def jaccard_similarity(t1, t2):
    intersection = np.sum(np.minimum(t1, t2))
    union = np.sum(np.maximum(t1, t2))
    return intersection / union


def mean_nf(x, n):
    return format_nf(np.mean(x), n)


jaccard = jaccard_similarity
cos = cosine_similarity
minkowski = minkowski_distance
