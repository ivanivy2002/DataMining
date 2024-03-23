# hw3_7.py
import utils as u  # 从utils.py导入辅助函数

age = [13, 15, 16, 16, 19, 20,
       20, 21, 22, 22, 25, 25,
       25, 25, 30, 33, 33, 35,
       35, 35, 35, 36, 40, 45,
       46, 52, 70]

# 计算均值
mean = u.np.mean(age)
print(f"均值：{u.format_2f(mean)}")
# 计算标准差
std = u.np.std(age)
# 计算均值绝对偏差
mad = u.np.mean(u.np.abs(age - mean))
# 使用最小-最大规范化
min = u.np.min(age)
max = u.np.max(age)
new_min = 0
new_max = 1
new_age = (age - min) / (max - min) * (new_max - new_min) + new_min
print(u.format_nf_list(new_age, 2))
