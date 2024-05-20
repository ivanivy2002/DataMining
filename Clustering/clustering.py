from sklearn.datasets import make_blobs
import numpy as np

np.random.seed(0)
data, labels = make_blobs(n_samples=1000, n_features=20, centers=5)


# 绘图
def draw_dim_reduction(data, labels, method='PCA'):
    from matplotlib import pyplot as plt
    if method == 'PCA':
        from sklearn.decomposition import PCA
        # 使用PCA将数据降到2维
        pca = PCA(n_components=2)
        data_2d = pca.fit_transform(data)
    elif method == 't-SNE':
        from sklearn.manifold import TSNE
        # 使用t-SNE将数据降到2维
        tsne = TSNE(n_components=2, random_state=0)
        data_2d = tsne.fit_transform(data)
    else:
        raise ValueError('Invalid method')
    # 可视化
    plt.figure(figsize=(10, 7))
    plt.scatter(data_2d[:, 0], data_2d[:, 1], c=labels, cmap='viridis', s=50)
    plt.colorbar()
    plt.title(f'Data Visualization using {method}')
    plt.xlabel('Component 1')
    plt.ylabel('Component 2')
    plt.show()


draw_dim_reduction(data, labels, method='PCA')


def init_centers(X, K, method='random'):
    n_samples, n_features = X.shape
    if method == 'random':
        return X[np.random.choice(n_samples, K, replace=False)]
    elif method == 'cautious':
        centers = np.empty((K, n_features), dtype=X.dtype)
        # 选择第一个中心
        centers[0] = X[np.random.randint(n_samples)]
        # 选择剩余的中心
        closest_dist_sq = np.full(n_samples, np.inf)
        for i in range(1, K):
            dist_sq = np.sum((X - centers[i - 1]) ** 2, axis=1)
            closest_dist_sq = np.minimum(closest_dist_sq, dist_sq)
            probabilities = closest_dist_sq / closest_dist_sq.sum()
            cumulative_probabilities = np.cumsum(probabilities)
            r = np.random.rand()
            index = np.searchsorted(cumulative_probabilities, r)
            centers[i] = X[index]
        return centers


# 实现算法
def kmeanspp(X, K, max_iters=100, threshold=1e-4):
    # TODO
    centers = init_centers(X, K, method='cautious')
    tmp_labels = np.zeros(X.shape[0])
    for _ in range(max_iters):
        # 计算距离
        distances = np.linalg.norm(X[:, np.newaxis] - centers, axis=2)
        # 分配簇：更新标签
        tmp_labels = np.argmin(distances, axis=1)

        # 计算新的中心
        new_centers = np.array([X[tmp_labels == k].mean(axis=0) for k in range(K)])

        # 检查收敛
        if np.linalg.norm(new_centers - centers) < threshold:
            break
        centers = new_centers

    return tmp_labels


def em(X, K):
    # TODO
    # 定义一个很小的正数epsilon以避免除以零
    epsilon = 1e-10
    threshold = 30

    def e_step(X, centers):
        K = centers.shape[0]
        w = np.zeros((K, X.shape[0]))
        # 计算权重
        for i in range(K):
            distances = np.linalg.norm(X - centers[i], axis=1)
            w[i] = 1 / ((distances + epsilon) ** 2)
        w = w / w.sum(axis=0)
        return w

    def m_step(X, w):
        K = w.shape[0]
        centers = np.zeros((K, X.shape[1]))
        # 更新中心
        for i in range(K):
            centers[i] = (w[i][:, np.newaxis] * X).sum(axis=0) / w[i].sum()
        return centers

    centers = init_centers(X, K, method='random')
    w = np.zeros((K, X.shape[0]))
    # EM 的迭代
    for _ in range(100):
        w = e_step(X, centers)
        new_centers = m_step(X, w)
        if np.linalg.norm(new_centers - centers) < threshold:
            break
        centers = new_centers
    return np.argmax(w, axis=0)


# 尝试选择最佳的K
def n_cluster_choice(X, max_K=10):
    # TODO
    # 选择最佳的K
    ari = []
    for K in range(1, max_K + 1):
        ari.append(adjusted_rand_score(labels, kmeanspp(X, K)))
        print(f'K={K}, ARI={ari[-1]}')
    return np.argmax(ari) + 1


from sklearn.metrics import adjusted_rand_score

n_cluster_choice(data, 10)
n_clusters = 5

kmeanspp_labels = kmeanspp(data, n_clusters)
ari = adjusted_rand_score(labels, kmeanspp_labels)
print(ari)
assert ari > 0.999

em_labels = em(data, n_clusters)
ari = adjusted_rand_score(labels, em_labels)
print(ari)
assert ari > 0.45

print("PASS")
