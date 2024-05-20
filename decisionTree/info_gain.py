from numpy import log2


def entropy(p):
    if p == 0 or p == 1:
        return 0
    return -p * log2(p) - (1 - p) * log2(1 - p)


def part_info(posi, tot):
    p = posi / tot
    # print(f'{posi}/{tot} ={p}')
    return entropy(p)


def Info(posi1, tot1, posi2, tot2):
    tot = tot1 + tot2
    p1 = tot1 / tot
    p2 = tot2 / tot
    ret = p1 * part_info(posi1, tot1) + p2 * part_info(posi2, tot2)
    return ret


ALL = 0.99108
arr = [1, 0, 1, 0, 0, 1, 0, 1, 0]
# Info(1, 1, 3, 8)
for i in range(1, 9):
    # print(f'Feature {i}')
    posi1 = sum(arr[:i])
    posi2 = sum(arr[i:])
    ans = Info(posi1, i, posi2, 9 - i)
    print(f'{i}/9 * - e({posi1}/{i}).. + {9 - i}/9 * - e({posi2}/{9-i}).. = {ans:.4f}, Gain = {ALL - ans:.4f}')

