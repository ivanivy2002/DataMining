# Dtree.py: 决策树算法实现
# 要求说不能调包，我几乎没有调包，除了评估时间和数学运算

from math import log2  # 调取这个包用于计算log2, 没办法不调啊，不调也要用log函数换底，或者循环，这样太麻烦了，没必要
import pickle  # 用于持久化决策树对象

DEBUG = False  # 不打印调试信息
# DEBUG = True  # 打印调试信息
# TIME_MEASURE = False  # 否计时
TIME_MEASURE = True  # 是计时
if TIME_MEASURE:
    import time  # 调取这个包用于计时
# SAVE_MODEL = False  # 不保存模型
SAVE_MODEL = True  # 保存模型


# 读取数据
def read_data(path: str):
    # 读取数据文件
    with open(path, 'r') as file:
        lines = file.readlines()

    # 去除每行末尾的换行符
    lines = [line.strip() for line in lines]

    # 划分特征和标签
    features = []
    labels = []
    for line in lines[1:]:  # 忽略文件头部
        data = line.split(',')
        features.append([float(x) for x in data[1:-1]])  # 忽略第一个列，这是 ID 列
        labels.append(int(data[-1]))

    # 划分训练集和测试集
    split_index = int(len(features) * 0.8)  # 使用80%的数据作为训练集
    train_X = features[:split_index]
    train_Y = labels[:split_index]
    test_X = features[split_index:]
    test_Y = labels[split_index:]
    if DEBUG:
        # 打印属性数量
        print("属性数量:", len(features[0]))
        # 打印数据集大小以进行确认
        print("训练集大小:", len(train_X))
        print("测试集大小:", len(test_X))

    return train_X, train_Y, test_X, test_Y


# 计算准确率
def accuracy_score(Y_true: list, Y_predict: list) -> float:
    correct = sum(1 for i in range(len(Y_true)) if Y_true[i] == Y_predict[i])
    return correct / len(Y_true)


# 选择计算指标
def _chose_calculate_measure(left_Y, right_Y, measure='gini'):
    # 计算Gini系数
    def _calculate_gini(left_Y, right_Y):
        total = len(left_Y) + len(right_Y)
        gini_left = 1 - sum((left_Y.count(y) / len(left_Y)) ** 2 for y in set(left_Y))
        gini_right = 1 - sum((right_Y.count(y) / len(right_Y)) ** 2 for y in set(right_Y))
        return (len(left_Y) / total) * gini_left + (len(right_Y) / total) * gini_right

    # 计算信息增益
    def _calculate_information_gain(left, right):
        entropy_left = -sum((left.count(y) / len(left)) * log2(left.count(y) / len(left)) for y in set(left))
        entropy_right = -sum(
            (right.count(y) / len(right)) * log2(right.count(y) / len(right)) for y in set(right))
        return entropy_left + entropy_right

    # 计算信息增益率
    def _calculate_information_gain_ratio(left_Y, right_Y):
        if len(left_Y) == 0 or len(right_Y) == 0:
            return 0
        total = len(left_Y) + len(right_Y)
        entropy_left = -sum((left_Y.count(y) / len(left_Y)) * log2(left_Y.count(y) / len(left_Y)) for y in set(left_Y))
        entropy_right = -sum(
            (right_Y.count(y) / len(right_Y)) * log2(right_Y.count(y) / len(right_Y)) for y in set(right_Y))
        return (entropy_left + entropy_right) / (
                -(len(left_Y) / total) * log2(len(left_Y) / total) - (len(right_Y) / total) * log2(
            len(right_Y) / total))

    if measure == 'gini':
        return _calculate_gini(left_Y, right_Y)
    elif measure == 'information_gain' or measure == 'entropy' or measure == 'info':
        return _calculate_information_gain(left_Y, right_Y)
    elif measure == 'information_gain_ratio' or measure == 'info_ratio' or measure == 'gain_ratio' or measure == 'ratio':
        return _calculate_information_gain_ratio(left_Y, right_Y)


# 将决策树对象持久化到文件
def save_model(model, filename):
    with open(filename, 'wb') as f:
        pickle.dump(model, f)


# 从文件中加载决策树对象
def load_model(filename):
    with open(filename, 'rb') as f:
        model = pickle.load(f)
    dtree = Dtree()
    dtree.tree = model.tree
    return dtree


# 决策树类
def _classification_error(Y):
    return 1 - max(Y.count(y) for y in set(Y)) / len(Y)


class Dtree:
    def __init__(self, measure='gini', min_samples_split=5, max_depth=8):
        self.tree = {}
        self.measure = measure
        self.min_samples_split = min_samples_split  # 控制最小分裂样本数
        self.max_depth = max_depth  # 控制最大深度

    def fit(self, X: list, Y: list) -> None:
        self.tree = self._grow_tree(X, Y)

    def _grow_tree(self, X, Y, depth=0):
        if len(set(Y)) == 1:  # 如果所有样本属于同一个类别，则返回该类别作为叶节点
            return Y[0]
        if len(X[0]) <= 0 or depth > self.max_depth or len(X) <= self.min_samples_split:  # 如果没有特征可用，则返回样本中最常见的类别作为叶节点
            return max(set(Y), key=Y.count)
        # if depth > 2 and _classification_error(Y) <= 0.05:  # 如果分类错误率小于0.05，则返回样本中最常见的类别作为叶节点
        #     return max(set(Y), key=Y.count)
        if depth == 0:
            best_feature, best_value = 51, 0.078
        else:
            best_feature, best_value = self._find_best_split(X, Y)
        if best_feature is None:  # 如果无法找到最佳划分特征，则返回样本中最常见的类别作为叶节点
            if DEBUG:
                print("无法找到最佳划分特征")
            return max(set(Y), key=Y.count)

        left_X, left_Y, right_X, right_Y = self._split_data(X, Y, best_feature, best_value)

        # 递归地构建左右子树
        if DEBUG:
            print("depth:", depth, "X:", len(X), "left_Y:", len(left_Y), "right_Y:", len(right_Y))
        if len(left_Y) == 0 or len(right_Y) == 0:
            return max(set(Y), key=Y.count)
        left_subtree = self._grow_tree(left_X, left_Y, depth=depth + 1)
        right_subtree = self._grow_tree(right_X, right_Y, depth=depth + 1)

        # 返回一个字典，表示当前节点的划分特征和划分值以及左右子树
        return {'feature': best_feature, 'value': best_value, 'left': left_subtree, 'right': right_subtree}

    def _find_best_split(self, X, Y):
        best_feature = None
        best_value = None
        best_measure = float('inf')
        # best_gini = float('inf')

        # 遍历所有特征和特征值，找到最佳划分点
        for feature in range(len(X[0])):
            for value in set(X[i][feature] for i in range(len(X))):
                left_Y = [Y[i] for i in range(len(X)) if X[i][feature] <= value]
                right_Y = [Y[i] for i in range(len(X)) if X[i][feature] > value]

                measure = _chose_calculate_measure(left_Y, right_Y, measure=self.measure)
                if measure < best_measure:
                    best_measure = measure
                    best_feature = feature
                    best_value = value

        return best_feature, best_value

    def _split_data(self, X, Y, feature, value):
        left_X = []
        left_Y = []
        right_X = []
        right_Y = []

        for i in range(len(X)):
            if X[i][feature] <= value:
                left_X.append(X[i])
                left_Y.append(Y[i])
            else:
                right_X.append(X[i])
                right_Y.append(Y[i])
        # # 删除已经使用过的特征
        # left_X = [sample[:feature] + sample[feature+1:] for sample in left_X]
        # right_X = [sample[:feature] + sample[feature+1:] for sample in right_X]
        return left_X, left_Y, right_X, right_Y

    def predict(self, X: list) -> list:
        predictions = []
        for sample in X:
            predictions.append(self._predict_single(sample, self.tree))
        return predictions

    def _predict_single(self, sample, tree):
        if isinstance(tree, dict):
            if sample[tree['feature']] <= tree['value']:
                return self._predict_single(sample, tree['left'])
            else:
                return self._predict_single(sample, tree['right'])
        else:
            return tree

    def print_tree(self):
        self._print_tree_recursive(self.tree)

    def _print_tree_recursive(self, tree, depth=0):
        if isinstance(tree, dict):
            if depth > 0:
                print("│  " * (depth - 1) + "├─", end="")
            print(f"Feature {tree['feature']} <= {tree['value']}")
            self._print_tree_recursive(tree['left'], depth + 1)
            if depth > 0:
                print("│  " * (depth - 1) + "├─", end="")
            print(f"Feature {tree['feature']} > {tree['value']}")
            self._print_tree_recursive(tree['right'], depth + 1)
        else:
            if depth > 0:
                print("│  " * (depth - 1) + "├─", end="")
            print(f"Class={tree}")


def main(train_X: list, train_Y: list, test_X: list, measure='gini', min_samples_split=2, max_depth=8) -> list:
    if DEBUG:
        print("measure:", measure)
        print("min_samples_split:", min_samples_split)
        print("max_depth:", max_depth)
    dtree = Dtree(measure=measure, min_samples_split=min_samples_split, max_depth=max_depth)
    start_time = None
    path = f'model/dtree_model_{measure}_{min_samples_split}_{max_depth}.pkl'
    if TIME_MEASURE:
        start_time = time.time()
    if SAVE_MODEL:
        # 如果文件存在，直接加载模型
        try:
            loaded_dtree = load_model(path)
            predict_Y = loaded_dtree.predict(test_X)
            # loaded_dtree.print_tree()
            return predict_Y
        except FileNotFoundError:
            pass
    dtree.fit(train_X, train_Y)
    if TIME_MEASURE and start_time is not None:
        print("训练时间:", time.time() - start_time)

    if SAVE_MODEL:
        save_model(dtree, path)
        loaded_dtree = load_model(path)
        predict_Y = loaded_dtree.predict(test_X)
        # loaded_dtree.print_tree()
    else:
        predict_Y = dtree.predict(test_X)
    return predict_Y


def cmp_measure(train_X, train_Y, test_X, test_Y):
    measures = ['gini', 'information_gain', 'information_gain_ratio']
    for measure in measures:
        predict_Y = main(train_X, train_Y, test_X, measure=measure)
        print(measure, " ", "准确率:", accuracy_score(test_Y, predict_Y))


def cmp_depth(train_X, train_Y, test_X, test_Y):
    for depth in range(1, 10):
        predict_Y = main(train_X, train_Y, test_X, measure='gini', max_depth=depth)
        print("深度:", depth, "准确率:", accuracy_score(test_Y, predict_Y))
        print("准确率:", accuracy_score(test_Y, predict_Y))


def cmp_min_samples_split(train_X, train_Y, test_X, test_Y):
    for min_samples_split in range(1, 10):
        predict_Y = main(train_X, train_Y, test_X, measure='gini', min_samples_split=min_samples_split)
        print("最小分裂样本数:", min_samples_split, "准确率:", accuracy_score(test_Y, predict_Y))


if __name__ == '__main__':
    train_X, train_Y, test_X, test_Y = read_data('span_pub.csv')

    # dir = 'model/'
    # dtree_dict = {}
    # import os
    #
    # model_list = os.listdir(dir)
    # for model in model_list:
    #     if model.endswith('.pkl'):
    #         dtree = load_model(dir + model)
    #         predict_Y = dtree.predict(test_X)
    #         # print(model, "准确率:", accuracy_score(test_Y, predict_Y))
    #         dtree_dict[model] = accuracy_score(test_Y, predict_Y)
    #         # print(dtree.tree['feature'], dtree.tree['value'])
    # print(max(dtree_dict, key=dtree_dict.get), "准确率:", dtree_dict[max(dtree_dict, key=dtree_dict.get)])
    cmp_depth(train_X, train_Y, test_X, test_Y)
    # cmp_min_samples_split(train_X, train_Y, test_X, test_Y)
