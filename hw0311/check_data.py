import csv
import pandas as pd


def count_unique_values(csv_data):
    unique_values = {}
    for row in csv_data:
        for column_name, value in row.items():
            if column_name not in unique_values:
                unique_values[column_name] = set()
            unique_values[column_name].add(value)  # 去除空白字符

    for column_name, values in unique_values.items():
        print(f"Column '{column_name}' has {len(values)} unique values:")
        print(values)


data_path = "../data/hw11/kddcup.data_10_percent_corrected_10000"


def load_data(data_path):
    # 示例数据
    with open(data_path, newline="") as csvfile:
        csv_data = csv.DictReader(csvfile)
        # count_unique_values(csv_data)

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
        # print(data)
        count_unique_values(data.to_dict(orient="records"))


# load_data(data_path)

def load_data_encoded(data_path):
    # 示例数据
    with open(data_path, newline="") as csvfile:
        csv_data = csv.DictReader(csvfile)
        data = pd.read_csv(data_path)
        # print(data)
        count_unique_values(data.to_dict(orient="records"))


load_data_encoded("../data/hw11/test_encoded_cat_attr.csv")
