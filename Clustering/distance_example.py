import numpy as np

# 创建示例数组
a = np.random.rand(1, 5, 3)
# a = np.random.rand(5, 1, 3)
b = np.random.rand(2, 3)

# 广播运算
# result = a + b[:, np.newaxis, :]
result = a + b

print("Array a shape:", a.shape)
print("Array b shape:", b.shape)
print("Result shape:", result.shape)
