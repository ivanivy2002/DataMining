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

    return features, labels


def main():
    # 读取数据
    features, labels = read_data('span_pub.csv')

    # 统计数据
    num_samples = len(features)
    num_features = len(features[0])
    num_pos = sum(labels)
    num_neg = num_samples - num_pos

    # 输出统计结果
    print('Number of samples:', num_samples)
    print('Number of features:', num_features)
    print('Number of positive samples:', num_pos)
    print('Number of negative samples:', num_neg)
    # 输出每个特征的均值和标准差，四分位数
    for i in range(num_features):
        values = [x[i] for x in features]
        mean = sum(values) / num_samples
        std = (sum((x - mean) ** 2 for x in values) / num_samples) ** 0.5
        values.sort()
        q1 = values[num_samples // 4]
        q2 = values[num_samples // 2]
        q3 = values[num_samples * 3 // 4]
        print('Feature %d: mean=%.2f, std=%.2f, 25%%=%.2f, 50%%=%.2f, 75%%=%.2f' % (i, mean, std, q1, q2, q3))


if __name__ == '__main__':
    main()
