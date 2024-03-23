# hw3_8.py
import utils as u
age = [23, 23, 27, 27, 39, 41, 47, 49, 50, 52, 54, 54, 56, 57, 58, 58, 60, 61]
fat = [9.5, 26.5, 7.8, 17.8, 31.4, 25.9, 27.4, 27.2, 31.2, 34.6, 42.5, 28.8, 33.4, 30.2, 34.1, 32.9, 41.2, 35.7]
# z-score标准化
z_age = u.z_score_normalize(age)
z_fat = u.z_score_normalize(fat)
print("年龄标准化：", u.format_nf_list(z_age, 2))
print("脂肪标准化：", u.format_nf_list(z_fat, 2))

# 计算相关系数
corr = u.np.corrcoef(age, fat)
print("相关系数：", u.format_2f(corr[0][1]))
# 计算协方差
cov = u.np.cov(age, fat)
print("协方差：",  u.format_2f(cov[0][1]))