import numpy as np
from scipy.stats import multivariate_normal


def initialize_parameters(X, K):
    n_samples, n_features = X.shape
    pi = np.ones(K) / K
    mu = X[np.random.choice(n_samples, K, False), :]
    sigma = np.array([np.eye(n_features)] * K)
    return pi, mu, sigma


def e_step(X, pi, mu, sigma):
    n_samples, K = X.shape[0], pi.shape[0]
    gamma = np.zeros((n_samples, K))

    for k in range(K):
        gamma[:, k] = pi[k] * multivariate_normal(mean=mu[k], cov=sigma[k]).pdf(X)

    gamma /= gamma.sum(axis=1, keepdims=True)
    return gamma


def m_step(X, gamma):
    n_samples, n_features = X.shape
    K = gamma.shape[1]

    N_k = gamma.sum(axis=0)
    pi = N_k / n_samples
    mu = np.dot(gamma.T, X) / N_k[:, np.newaxis]
    sigma = np.zeros((K, n_features, n_features))

    for k in range(K):
        X_centered = X - mu[k]
        sigma[k] = np.dot(gamma[:, k] * X_centered.T, X_centered) / N_k[k]

    return pi, mu, sigma


def gmm_em(X, K, max_iters=100, tol=1e-4):
    pi, mu, sigma = initialize_parameters(X, K)
    log_likelihoods = []

    for _ in range(max_iters):
        gamma = e_step(X, pi, mu, sigma)
        pi, mu, sigma = m_step(X, gamma)

        log_likelihood = np.sum(
            np.log(np.sum([pi[k] * multivariate_normal(mu[k], sigma[k]).pdf(X) for k in range(K)], axis=0)))
        log_likelihoods.append(log_likelihood)

        if len(log_likelihoods) > 1 and np.abs(log_likelihoods[-1] - log_likelihoods[-2]) < tol:
            break

    return pi, mu, sigma, gamma


# 创建数据
np.random.seed(0)
X = np.vstack(
    [np.random.multivariate_normal(mean, cov, 200) for mean, cov in [([0, 0], np.eye(2)), ([3, 3], np.eye(2))]])

# 使用EM算法进行聚类
K = 2
pi, mu, sigma, gamma = gmm_em(X, K)

print("混合系数 (pi):", pi)
print("均值向量 (mu):", mu)
print("协方差矩阵 (sigma):", sigma)
