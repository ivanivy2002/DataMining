# hw3.6.py
import utils as u
data = [200, 300, 400, 600, 1000]
mean = u.np.mean(data)
# 计算标准差
std = u.np.std(data)
print(f"均值：{u.format_2f(mean)}")
print(f"标准差：{u.format_nf(std,3)}")
# 标准化数据
normalized_data = [u.format_nf((item - mean) / std, 3) for item in data]
normalized_data = [float(item) for item in normalized_data]
print("标准化数据：", normalized_data)
# 均值绝对偏差
mad = u.np.mean([abs(item - mean) for item in data])
print(f"均值绝对偏差：{u.format_nf(mad, 3)}")
# 使用均值绝对偏差标准化数据
normalized_data = [u.format_nf((item - mean) / mad, 3) for item in data]
normalized_data = [float(item) for item in normalized_data]
print("使用均值绝对偏差标准化数据：", normalized_data)