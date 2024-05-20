import numpy as np
from scipy.optimize import minimize
from numpy import sqrt


# 定义函数来计算 m1, m2, m3 基于 k1, k2, k4, k5
def compute_m(k1, k2, k4, k5):
    m1 = -(-k1 + 2 * (sqrt(k2) + k5) + sqrt(k4))
    m2 = k1
    m3 = -k1 + (sqrt(k2) + k5)
    return m1, m2, m3


# 定义函数计算 k
def compute_k(kvec):
    k1, k2, k4, k5 = kvec
    k = sqrt(k1 ** 2 + 2 * k2 ** 2 + k4 + 2 * k5 ** 2)  # 根据需要调整计算方式
    return k


# 最大化最小的 di 的目标函数
def objective(kvec):
    k1, k2, k4, k5 = kvec  # 解包 kvec
    m1, m2, m3 = compute_m(k1, k2, k4, k5)
    k = compute_k(kvec)
    d1, d2, d3 = m1 / k, m2 / k, m3 / k
    return -np.min([d1, d2, d3])  # 最小化负值以找到最大的最小值


# 初始猜测值
initial_guess = np.array([1, 1, 1, 1])

# k1, k2, k4, k5 的界限，假设它们为正
bounds = [(0.01, 10), (0.01, 10), (0.01, 10), (0.01, 10)]


def minimize_objective(objective, initial_guess, bounds):
    # 执行优化
    result = minimize(objective, initial_guess, bounds=bounds, method='L-BFGS-B')
    # 打印结果
    if result.success:
        optimized_kvec = result.x
        max_min_di = -result.fun  # 转换回正值因为我们最小化了负值
        print("Optimized kvec:", optimized_kvec)
        print("Maximum of minimum di:", max_min_di)
    else:
        print("Optimization failed:", result.message)
    return optimized_kvec
