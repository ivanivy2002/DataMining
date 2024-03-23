# hw0311_8_19.py
import utils as u
import numpy as np


def correlation_coefficient(x, y):
    x_mean = np.mean(x)
    y_mean = np.mean(y)
    covariance = np.sum((x - x_mean) * (y - y_mean))
    denominator = np.sqrt(np.sum(np.power(x - x_mean, 2)) * np.sum(np.power(y - y_mean, 2)))
    if denominator == 0:
        return "N/A"
    return covariance / denominator


corr = correlation_coefficient


def test1():
    x = (1, 1, 1, 1)
    y = (2, 2, 2, 2)
    print("(a)")
    print("\tcos: ", u.cos(x, y))
    print("\tcorr: ", corr(x, y))
    print("\teuclidean: ", u.minkowski(x, y, 2))


def test2():
    x = (0, 1, 0, 1)
    y = (1, 0, 1, 0)
    print("(b)")
    print("\tcos: ", u.cos(x, y))
    print("\tcorr: ", corr(x, y))
    print("\teuclidean: ", u.minkowski(x, y, 2))
    print("\tjaccard: ", u.jaccard(x, y))


def test3():
    x = (0, -1, 0, 1)
    y = (1, 0, -1, 0)
    print("(c)")
    print("\tcos: ", u.cos(x, y))
    print("\tcorr: ", corr(x, y))
    print("\teuclidean: ", u.minkowski(x, y, 2))


def test4():
    x = (1, 1, 0, 1, 0, 1)
    y = (1, 1, 1, 0, 0, 1)
    print("(d)")
    print("\tcos: ", u.cos(x, y))
    print("\tcorr: ", corr(x, y))
    print("\tjaccard: ", u.jaccard(x, y))


def test5():
    x = (2, -1, 0, 2, 0, -3)
    y = (-1, 1, -1, 0, 0, -1)
    print("(e)")
    print("\tcos: ", u.cos(x, y))
    print("\tcorr: ", corr(x, y))
    # 此处corr很小, 做一个手动计算:
    # x_mean = 0
    # y_mean = -1/3
    # covariance = 2 * (-1-(-1/3)) + (-1) * (1-(-1/3)) + 0 + 2 * (0-(-1/3)) + 0  + (-3) * (-1-(-1/3)) = 0


test1()
test2()
test3()
test4()
test5()
