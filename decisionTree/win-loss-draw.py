from numpy import std, mean, sqrt

support_vector_machine = [0.8719, 0.8478, 0.7073, 0.9642, 0.8449, 0.8507, 0.7682, 0.744, 0.5981, 0.837, 0.871, 0.8261,
                          0.8889,
                          0.96, 0.9298, 0.7356, 0.8649, 0.7695, 0.7692, 0.9833, 0.7494, 0.9888, 0.9604]
naive_bayes = [0.7962, 0.7681, 0.5805, 0.9599, 0.835, 0.7754, 0.7591, 0.747, 0.4859, 0.8407, 0.8323, 0.788,
               0.8234, 0.9533, 0.9474, 0.7316, 0.8311, 0.7604, 0.6971, 0.7004, 0.4504, 0.9663, 0.9307]
decision_tree = [0.9209, 0.8551, 0.8195, 0.9514, 0.7624, 0.858, 0.724, 0.709, 0.6729, 0.8, 0.8194, 0.8533, 0.8917,
                 0.9467, 0.7895, 0.7334, 0.7703, 0.7435, 0.7885, 0.8372, 0.7104, 0.9438, 0.9307]
sizes = [898, 690, 205, 699, 303, 690, 768, 1000, 214, 270, 155, 368, 351, 150, 57, 3200, 148, 768, 208, 958, 846, 178,
         101]


def z_score(p1, p2, n):
    p = (p1 + p2) / 2
    return (p1 - p2) / sqrt(2 * p * (1 - p) / n)


def cal_win_loss_draw(data1: list, data2: list) -> tuple:
    win = loss = draw = 0
    for i in range(len(data1)):
        z_score_p = z_score(data1[i], data2[i], sizes[i])
        if z_score_p > 1.96:
            win += 1
        elif z_score_p < -1.96:
            loss += 1
        else:
            draw += 1
    return win, loss, draw


print("decision_tree vs naive_bayes:", cal_win_loss_draw(decision_tree, naive_bayes))
print("decision_tree vs support_vector_machine:", cal_win_loss_draw(decision_tree, support_vector_machine))
print("naive_bayes vs support_vector_machine:", cal_win_loss_draw(naive_bayes, support_vector_machine))
print("decision_tree vs decision_tree:", cal_win_loss_draw(decision_tree, decision_tree))
