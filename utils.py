import numpy as np
# 定义函数，将数字格式化为整数
def format_integer(num):
    return "{:.0f}".format(num)


# 定义函数，将数字格式化为2位小数
def format_2f(num):
    return "{:.2f}".format(num)


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
