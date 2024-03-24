import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import NearestNeighbors
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

# 加载数据集
data_path = "../data/hw11/kddcup.data_10_percent_corrected_10000"
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

# 仅保留分类属性
categorical_attributes = data.select_dtypes(include=['object']).copy()

# 对分类属性进行编码
label_encoders = {}
for column in categorical_attributes.columns:
    label_encoders[column] = LabelEncoder()
    categorical_attributes[column] = label_encoders[column].fit_transform(categorical_attributes[column])

# 使用匹配和逆发生频率度量计算最近邻居
matching_neighbors = NearestNeighbors(n_neighbors=2, metric='hamming')
matching_neighbors.fit(categorical_attributes, categorical_attributes.index)

inv_freq_neighbors = NearestNeighbors(n_neighbors=2, metric='jaccard')
inv_freq_neighbors.fit(categorical_attributes, categorical_attributes.index)

# 计算具有匹配类标签的实例数量
matching_count = 0
inv_freq_count = 0


for i, instance in enumerate(categorical_attributes.values):
    neighbors_idx_match = matching_neighbors.kneighbors([instance], return_distance=False)[0][1]
    neighbors_idx_inv_freq = inv_freq_neighbors.kneighbors([instance], return_distance=False)[0][1]

    if data.iloc[i]['class'] == data.iloc[neighbors_idx_match]['class']:
        matching_count += 1

    if data.iloc[i]['class'] == data.iloc[neighbors_idx_inv_freq]['class']:
        inv_freq_count += 1

print(f"具有匹配类标签的实例数量（匹配度量）：{matching_count}")
print(f"具有匹配类标签的实例数量（逆发生频率度量）：{inv_freq_count}")
