# hw2_3.py
# 示例数据
data = {
    '1-5': 200,
    '6-15': 450,
    '16-20': 300,
    '21-50': 1500,
    '51-80': 700,
    '81-110': 44
}
N = sum(data.values())  # 3194
cumulative_frequency = 0
lower_bound, upper_bound = 0, 0
freq_median = 0
# 确定中位数区间
for interval, frequency in data.items():
    cumulative_frequency += frequency
    if cumulative_frequency >= N / 2:
        lower_bound, upper_bound = map(int, interval.split('-'))
        freq_median = frequency
        break
print("中位数区间：", lower_bound, "-", upper_bound)
L1 = lower_bound - 1  # 20
freq_l = 0
for interval, frequency in data.items():
    if int(interval.split('-')[1]) <= lower_bound:
        freq_l += frequency
width = upper_bound - lower_bound + 1  # 30
median_hand = L1 + ((N / 2) - freq_l) / freq_median * width
# 20 + (3194 / 2 - 950) / 1500 * 30 = 32.94
print("近似中位数：", median_hand)
