# hw0311_7.py
def hamming_distance(x, y):
    return bin(x ^ y).count("1")


def jaccard_similarity(x, y):
    return bin(x & y).count("1") / bin(x | y).count("1")


x = 0b0101010001
y = 0b0100011000
# 统计不同的位数
print(hamming_distance(x, y))
# Jaccard 相似度: 交集元素个数 / 并集元素个数
print(jaccard_similarity(x, y))
