# hw3_3.py
import numpy as np
import utils as u  # 从utils.py导入辅助函数

age = [13, 15, 16, 16, 19, 20,
       20, 21, 22, 22, 25, 25,
       25, 25, 30, 33, 33, 35,
       35, 35, 35, 36, 40, 45,
       46, 52, 70]

# 分成3个一组:
div_age = [age[i:i + 3] for i in range(0, len(age), 3)]
print(div_age)
# 计算均值
mean = [np.mean(item) for item in div_age]
# 替换每组的每个数据为均值
replace_mean = [[u.format_2f(mean[i])] * 3 for i in range(len(mean))]
smoothed_data = [[float(val) for val in item] for item in replace_mean]
print(smoothed_data)
# # 给出一个方法寻找离群点
# outliers = []
# for i in range(len(age)):
#     if age[i] < Q1 - 1.5 * (Q3 - Q1) or age[i] > Q3 + 1.5 * (Q3 - Q1):
#         outliers.append(age[i])