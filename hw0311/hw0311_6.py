# hw0311_6.py
import math
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei']


def prob_draw(n, M):
    # sum (-1)^(n-k) * C(n, k) (k/n)^M
    prob = 0
    for k in range(0, n + 1):
        prob += math.pow(-1, n - k) * math.comb(n, k) * math.pow(k / n, M)
    return prob


n = 10
M = 0
prob_dic = {}
for M in range(10, 60):
    prob_dic[M] = prob_draw(n, M)

plt.plot(prob_dic.keys(), prob_dic.values())
plt.title('样本包含所有' + str(n) + '组的点的概率')
plt.xlabel('样本大小M')
plt.ylabel('概率')
plt.xlim(0, 60)
plt.ylim(0, 1)
plt.grid(True)
plt.show()
