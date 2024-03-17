import numpy as np
import utils as u

fat_age = {
    9.5: 23, 26.5: 23, 7.8: 27, 17.8: 27, 31.4: 39, 25.9: 41, 27.4: 47, 27.2: 49, 31.2: 50, 34.6: 52, 42.5: 54,
    28.8: 54, 33.4: 56, 30.2: 57, 34.1: 58, 32.9: 58, 41.2: 60, 35.7: 61
}
# 计算均值
fat_list = list(fat_age.keys())
age_list = list(fat_age.values())
# print("年龄: ", age_list)
# print("脂肪: ", fat_list)
fat_mean = u.format_2f(np.mean(fat_list))
age_mean = u.format_2f(np.mean(age_list))
fat_median = np.median(fat_list)
age_median = np.median(age_list)
fat_std = u.format_2f(np.std(fat_list))
age_std = u.format_2f(np.std(age_list))
# 打印表格
print(" \t\t 均值  中位数  标准差 ")
print("年龄    ", age_mean, "", age_median, "", age_std)
print("脂肪含量 ", fat_mean, "", fat_median, "", fat_std)

# 分开画盒图
u.plot_box([fat_list], ['脂肪含量'], '脂肪含量 盒图')  # 导入的是boxplot函数，我封装在utils中
u.plot_box([age_list], ['年龄'], '年龄 盒图')

# 画散点图
u.plot_scatter(age_list, fat_list, '年龄与脂肪含量散点图', '年龄', '脂肪含量')
# 画q-q图, 横坐标是年龄的分位数，纵坐标是脂肪含量的分位数
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei']
age_sorted = np.sort(age_list)
fat_sorted = np.sort(fat_list)
# 计算百分位数
percentiles = np.linspace(0, 100, len(age_sorted))
# 绘制 Q-Q 图
plt.title('age-fat Q-Q 图')
plt.xlabel('Age 百分位数')
plt.ylabel('Fat 百分位数')
plt.grid(True)
qx = [0, 0, 0, 0, 0]
qy = [0, 0, 0, 0, 0]
for i in range(5):  # 0, 1, 2, 3, 4
    qx[i] = np.percentile(age_sorted, i * 25)
    qy[i] = np.percentile(fat_sorted, i * 25)
    plt.scatter(qx[i], qy[i], color='black', label="Q" + str(i))  # 标出四分位数
for i in [1, 2, 3]:  # 1,2,3
    plt.text(qx[i], qy[i], f'Q{i}', ha='right', va='bottom', color='black')
plt.scatter(np.percentile(age_sorted, percentiles), np.percentile(fat_sorted, percentiles))
# 绘制参考线，这里用Q1,Q3连线
plt.plot([qx[1], qx[3]], [qy[1], qy[3]], color='red', linestyle='-')  # 添加实线部分
slope = (qy[3] - qy[1]) / (qx[3] - qx[1])
dx01 = qx[1] - qx[0]
dx34 = qx[3] - qx[4]
dy01 = slope * dx01
dy34 = slope * dx34
plt.plot([qx[0], qx[4]], [qy[1] - dy01, qy[3] - dy34], color='red', linestyle='--')  # 添加左侧虚线部分
plt.show()
