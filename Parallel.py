import numpy as np
from multiprocessing import Pool
from data_utils import load_data

def initialize_centroids(data, k):
    # 随机初始化 k 个质心
    return data[np.random.choice(data.shape[0], k, replace=False)]

def calculate_distances(data_chunk, centroids):
    # 计算一个数据块中每个点到所有质心的距离
    distances = np.linalg.norm(data_chunk[:, np.newaxis] - centroids, axis=2)
    return np.argmin(distances, axis=1)  # 返回最近质心的索引

def assign_clusters_parallel(data, centroids, num_workers=4):
    # 并行分配数据点到最近的质心
    # 将数据分块
    chunk_size = len(data) // num_workers
    data_chunks = [data[i:i + chunk_size] for i in range(0, len(data), chunk_size)]

    # 使用进程池并行计算
    with Pool(num_workers) as pool:
        results = pool.starmap(calculate_distances, [(chunk, centroids) for chunk in data_chunks])

    # 合并结果
    return np.concatenate(results)

def update_centroids(data, labels, k):
    # 更新质心为分配点的平均值
    return np.array([data[labels == i].mean(axis=0) for i in range(k)])

def kmeans_parallel(data, k, max_iter=100, tolerance=1e-4, num_workers=4):
    # 并行版本的 k-means 算法
    # 初始化质心
    centroids = initialize_centroids(data, k)
    for _ in range(max_iter):
        # 并行分配数据点到最近的质心
        labels = assign_clusters_parallel(data, centroids, num_workers)
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

    # 运行并行版本的 k-means 算法
    centroids, labels = kmeans_parallel(data, k, num_workers=4)

    # 输出结果
    print("Calculation complete!")
    print(f"Center of mass location:\n{centroids}")
    print(f"The cluster label for each point:\n{labels}")
