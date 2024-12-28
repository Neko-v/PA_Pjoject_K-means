import time
import matplotlib.pyplot as plt
from data_utils import load_data
from Non_Parallel import kmeans as kmeans_sequential
from Parallel import kmeans_parallel

def benchmark():
    # 加载数据
    data = load_data("data/dataset.csv")
    k = 5  # 定义簇的数量

    # 测试非并行版本
    start = time.time()
    centroids_seq, labels_seq = kmeans_sequential(data, k)
    end = time.time()
    sequential_time = end - start
    print(f"Non-parallel version running time: {sequential_time:.2f} seconds")

    # 测试并行版本
    num_workers_list = [2, 4, 8]  # 测试不同线程数
    parallel_times = []
    for num_workers in num_workers_list:
        start = time.time()
        centroids_par, labels_par = kmeans_parallel(data, k, num_workers=num_workers)
        end = time.time()
        parallel_time = end - start
        parallel_times.append(parallel_time)
        print(f"Parallel version running time (number of threads = {num_workers}): {parallel_time:.2f} seconds")

    # 输出加速比
    print("\nSpeedup:")
    speedups = []
    for num_workers, parallel_time in zip(num_workers_list, parallel_times):
        speedup = sequential_time / parallel_time
        speedups.append(speedup)
        print(f"Number of threads = {num_workers}: Speedup = {speedup:.2f}")

    # 绘制加速比图
    plt.figure()
    plt.plot(num_workers_list, speedups, marker='o')
    plt.xlabel("Number of threads")
    plt.ylabel("Speedup")
    plt.title("Relationship between number of threads and speedup ratio")
    plt.grid()
    plt.savefig("benchmarks/speedup_plot.png")
    plt.show()

    # 保存测试结果
    with open("benchmarks/results.txt", "w") as f:
        f.write(f"Non-parallel version running time: {sequential_time:.2f} seconds\n")
        for num_workers, parallel_time, speedup in zip(num_workers_list, parallel_times, speedups):
            f.write(f"Number of threads = {num_workers}: Speedup = {speedup:.2f}, Parallel version running time = {parallel_time:.2f} seconds\n")

if __name__ == "__main__":
    benchmark()
