import numpy as np
from data_utils import load_data

def initialize_centroids(data, k):
    # 随机初始化 k 个质心
    return data[np.random.choice(data.shape[0], k, replace=False)]

def assign_clusters(data, centroids):
    # 分配数据点到最近的质心
    distances = np.linalg.norm(data[:, np.newaxis] - centroids, axis=2)
    return np.argmin(distances, axis=1)

def update_centroids(data, labels, k):
    # 更新质心为分配点的平均值
    return np.array([data[labels == i].mean(axis=0) for i in range(k)])

def kmeans(data, k, max_iter=100, tolerance=1e-4):
    # 非并行版本的 k-means 算法
    # 初始化质心
    centroids = initialize_centroids(data, k)
    for _ in range(max_iter):
        # 分配数据点到最近的质心
        labels = assign_clusters(data, centroids)
        # 更新质心
        new_centroids = update_centroids(data, labels, k)
        # 检查质心变化是否小于容差
        if np.linalg.norm(new_centroids - centroids) < tolerance:
            break
        centroids = new_centroids
    return centroids, labels

if __name__ == "__main__":
    # 加载数据
    data = load_data("data/dataset.csv")
    k = 5  # 定义簇的数量

    # 运行 k-means 算法
    centroids, labels = kmeans(data, k)

    # 输出结果
    print("Calculation complete!")
    print(f"Center of mass location:\n{centroids}")
    print(f"The cluster label for each point:\n{labels}")
