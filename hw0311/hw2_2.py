# hw2_2.py
import numpy as np
import matplotlib.pyplot as plt

# 解决中文字体问题
plt.rcParams['font.sans-serif'] = ['SimHei']
import utils

# 数据
data = [13, 15, 16, 16, 19, 20, 20, 21, 22, 22, 25, 25, 25, 25, 30, 33, 33, 35, 35, 35, 35, 36, 40, 45, 46, 52, 70]

# 计算数据的长度
n = len(data)
print("数据长度：", n)
# 计算均值
mean = np.mean(data)
print("均值：", utils.format_2f(mean))

# 计算中位数
median = np.median(data)
print("中位数：", median)

# 计算众数
# 计算 bincount(data) 数组
bin_count_array = np.bincount(data)
# 找出出现次数最多的值的索引（众数的索引）
max_count_index = np.argmax(bin_count_array)
# 找出所有的众数
modes = np.where(bin_count_array == np.max(bin_count_array))[0]
print("所有的众数为：", modes)
# 判断是几模
num_modes = len(modes)
print("数据集是", num_modes, "模")

# 计算中列数
midrange = (data[0] + data[-1]) / 2
print("中列数：", midrange)

# 计算四分位数
Q1 = np.percentile(data, 25)
Q3 = np.percentile(data, 75)
print("第一个四分位数(Q1)：", Q1)
print("第三个四分位数(Q3)：", Q3)

# 计算五数概括
minimum = np.min(data)
maximum = np.max(data)
print("五数概括：min Q1 median Q3 max")
print(minimum, Q1, median, Q3, maximum)
# print("最小值：", minimum)
# print("第一个四分位数(Q1)：", Q1)
# print("中位数：", median)
# print("第三个四分位数(Q3)：", Q3)
# print("最大值：", maximum)

# 绘制盒图
plt.boxplot(data)
plt.title('2.2(f) 盒图')
plt.show()
