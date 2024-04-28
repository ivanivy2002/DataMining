import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import NearestNeighbors
import warnings
import math
import numpy as np

warnings.filterwarnings("ignore", category=UserWarning)

DEBUG = False


# DEBUG = True


def minkowski(a, b, h):
    return np.power(np.sum(np.power(np.abs(np.array(a) - np.array(b)), h)), 1 / h)


def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def categorical_attribute_select():
    # 加载数据集
    data_path = "../data/hw11/kddcup.data_10_percent_corrected_10000"
    test_path = "../data/hw11/testdata_labeled_500"
    column_names = [
        "duration", "protocol_type", "service", "flag", "src_bytes", "dst_bytes",
        "land", "wrong_fragment", "urgent", "hot", "num_failed_logins", "logged_in",
        "num_compromised", "root_shell", "su_attempted", "num_root", "num_file_creations",
        "num_shells", "num_access_files", "num_outbound_cmds", "is_host_login",
        "is_guest_login", "count", "srv_count", "serror_rate", "srv_serror_rate",
        "rerror_rate", "srv_rerror_rate", "same_srv_rate", "diff_srv_rate",
        "srv_diff_host_rate", "dst_host_count", "dst_host_srv_count",
        "dst_host_same_srv_rate", "dst_host_diff_srv_rate", "dst_host_same_src_port_rate",
        "dst_host_srv_diff_host_rate", "dst_host_serror_rate", "dst_host_srv_serror_rate",
        "dst_host_rerror_rate", "dst_host_srv_rerror_rate", "class"
    ]
    data = pd.read_csv(data_path, header=None, names=column_names)
    test_data = pd.read_csv(test_path, header=None, names=column_names)
    # 仅保留分类属性
    categorical_attributes = data.select_dtypes(include=['object']).copy()
    test_categorical_attributes = test_data.select_dtypes(include=['object']).copy()
    if DEBUG:  # 输出分类属性: protocol_type, service, flag, class
        print(categorical_attributes)
        print(test_categorical_attributes)
    # 通过 check_data.py的输出，我们可以看到：
    # protocol_type 有3个不同的值: icmp, tcp, udp
    # service 有58个不同的值
    # flag 有7个不同的值：{'OTH', 'RSTO', 'REJ', 'S1', 'SF', 'RSTR', 'S0'}
    # class 有6个不同的值：{'buffer_overflow.', 'normal.', 'smurf.', 'portsweep.', 'pod.', 'neptune.'}

    # 对分类属性进行编码
    le = LabelEncoder()
    # 合并训练数据集和测试数据集
    combined_data = pd.concat([categorical_attributes, test_categorical_attributes])
    # 对整个数据集进行编码
    combined_encoded = combined_data.apply(LabelEncoder().fit_transform)
    # 将编码后的数据集拆分回训练数据集和测试数据集
    cat_attr = combined_encoded[:len(categorical_attributes)]
    test_cat_attr = combined_encoded[len(categorical_attributes):]
    # cat_attr = categorical_attributes.apply(le.fit_transform)
    # test_cat_attr = test_categorical_attributes.apply(le.fit_transform)
    if DEBUG:
        print(cat_attr)  # 输出编码后的分类属性
        print(test_cat_attr)
    # 写入到文件
    cat_attr.to_csv("../data/hw11/encoded_cat_attr.csv", index=False)
    test_cat_attr.to_csv("../data/hw11/test_encoded_cat_attr.csv", index=False)
    print("completed")
    # 检验写入成功


def neighbour_select():
    # 加载数据集
    data_path = "../data/hw11/encoded_cat_attr.csv"
    test_path = "../data/hw11/test_encoded_cat_attr.csv"
    out_path = "../out/hw0311/match_measure_minkowski_cos.txt"
    data = pd.read_csv(data_path)
    test_data = pd.read_csv(test_path)
    # 去除class列
    data_X = data.drop(columns=['class'])
    test_data_X = test_data.drop(columns=['class'])
    if DEBUG:
        # 打印每个样本的特征数量
        for i in range(len(test_data_X)):
            print(f"i: {(test_data_X.iloc[i])}")
    min_arr = []
    for i in range(0, len(test_data_X)):
        t = test_data_X.iloc[i]
        found = False
        if i > 0:
            # 遍历前面的样本, 如果有一样的，直接取前一个样本的class
            for j in range(0, i):
                if t.equals(test_data_X.iloc[j]):
                    found = True
                    min_arr.append(min_arr[j])
                    class_predict = min_arr[j]
                    with open(out_path, 'a') as f:
                        f.write(f"{i}: cop {j} {class_predict}\n")
                    print(f"{i}: copy={j} class= {class_predict}")
                    break
        if found:
            continue
        # min_idx = match_measure(test_data_X.iloc[i], data_X)
        min_idx = match_measure(t, data_X)
        class_predict = data.loc[min_idx, 'class']
        with open(out_path, 'a') as f:
            f.write(f"{i}: {min_idx} {class_predict}\n")
        print(f"{i}: min_idx= {min_idx} {class_predict}")
        # 添加测试集的class列, 仅仅把min_idx那行的class值给test_data的class列
        min_arr.append(class_predict)

    for i in range(0, len(test_data_X)):
        test_data_X.loc[i, 'class'] = min_arr[i]
    # 校对结果
    correct = 0
    for i in range(0, len(test_data_X)):
        if test_data_X.loc[i, 'class'] == test_data.loc[i, 'class']:
            correct = correct + 1
    accuracy = correct / len(test_data_X)
    with open(out_path, 'a') as f:
        f.write(f"accuracy: {accuracy}\n")
    print(f"accuracy: {accuracy}")

    # 计算k近邻
    # n = 5
    # nbrs = NearestNeighbors(n_neighbors=n, algorithm='ball_tree').fit(data)
    # distances, indices = nbrs.kneighbors(test_data)
    # print(indices)
    # print(distances)


def match_measure(i, X):
    # 使用闵可夫斯基距离计算相似度选出最近的邻居
    h = 2
    min = math.inf
    min_index = 0
    for j in range(0, len(X)):
        distance = minkowski(X.iloc[j], i, h)
        if distance < min or (
                distance == min and cosine_similarity(X.iloc[j], i) < cosine_similarity(X.iloc[min_index], i)):
            min = distance
            min_index = j
    return min_index


def match_measure_manhattan(i, X):
    # 使用曼哈顿距离计算相似度选出最近的邻居
    h = 1
    min = math.inf
    min_index = 0
    for j in range(0, len(X)):
        distance = minkowski(X.iloc[j], i, h)
        if distance < min:
            min = distance
            min_index = j
    return min_index


def match_measure_cos(i, X):
    # 使用余弦相似度计算相似度选出最近的邻居
    min = math.inf
    min_index = 0
    for j in range(0, len(X)):
        distance = cosine_similarity(X.iloc[j], i)
        if distance < min or (distance == min and minkowski(X.iloc[j], i, 2) < minkowski(X.iloc[min_index], i, 2)):
            min = distance
            min_index = j
    return min_index


# categorical_attribute_select()
neighbour_select()
