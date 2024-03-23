# hw3_11.py
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei']

age = [13, 15, 16, 16, 19, 20,
       20, 21, 22, 22, 25, 25,
       25, 25, 30, 33, 33, 35,
       35, 35, 35, 36, 40, 45,
       46, 52, 70]


# 画一个宽度为10等宽的直方图
plt.hist(age, bins=10, edgecolor='black', alpha=0.7)
plt.title('年龄直方图')
plt.xlabel('年龄')
plt.ylabel('频数')
plt.show()


# 计算数据范围
min_age = min(age)
max_age = max(age)
# 计算 bins 的数量
num_bins = int((max_age - min_age) / 10) + 1
# 画直方图
plt.hist(age, bins=num_bins, range=(min_age, min_age + num_bins * 10), edgecolor='black', alpha=0.7)
# 添加标题和标签
plt.title('年龄直方图')
plt.xlabel('年龄')
plt.ylabel('频数')
# 显示图形
plt.show()


# 表格


# 簇抽样:
# 分成6簇
# 编号 1-5, 6-10, 11-15, 16-20, 21-25, 26-27