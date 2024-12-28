import numpy as np
from sklearn.datasets import make_blobs

def generate_data(samples=70000, centers=5, features=2, random_state=1214):
    # 生成带有簇结构的样本数据
    data, labels = make_blobs(n_samples=samples, centers=centers, n_features=features, random_state=random_state)
    return data, labels

def save_data(data, file_path="data/dataset.csv"):
    # 将数据保存为 CSV 文件
    np.savetxt(file_path, data, delimiter=",")
    print(f"Data saved to {file_path}")

def load_data(file_path="data/dataset.csv"):
    # 从 CSV 文件加载数据
    data = np.loadtxt(file_path, delimiter=",")
    print(f"Load data from {file_path}, data shape:{data.shape}")
    return data

if __name__ == "__main__":
    from sklearn.datasets import make_blobs
    # 生成数据
    data, labels = generate_data(samples=70000, centers=5, features=2)
    # 保存数据
    save_data(data, file_path="data/dataset.csv")
