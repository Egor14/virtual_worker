import numpy as np


def calc_eps(X, k=0.01):
    #print('ищу')
    min_dist = np.linalg.norm(X[1] - X[0], 1)
    for i in range(X.shape[0] - 1):
        #print('ищу')
        if np.amin(np.linalg.norm(X[i] - X[i + 1])) < min_dist:
            min_dist = np.amin(np.linalg.norm((X[i] - X)))

    return k * min_dist


def assign_clusters(X, clusters):
    cluster_labels = np.zeros(X.shape[0])

    for i in range(cluster_labels.shape[0]):
        cluster_labels[i] = np.argmin(np.linalg.norm(X[i] - clusters, axis=1))

    return cluster_labels


def update_clusters(X, cluster_labels):
    updated_clusters = np.zeros(((k, X.shape[1])))

    for i in range(k):
        updated_clusters[i] = np.mean(X[cluster_labels == i], axis=0)

    return updated_clusters


def is_converged(clusters, updated_clusters, eps):
    return (np.max(np.abs(clusters - updated_clusters)) < eps)


def kmeans(X, k, eps=None):
    if not eps:
        eps = calc_eps(X)

    # random selection of k different initial points as clusters' centers
    clusters = X[np.random.choice(X.shape[0], k, replace=False), :]
    #print('Изначальные центры кластеров')
    # print(clusters)
    #print('ищу')
    while True:
        #print('продолжаю')
        cluster_labels = assign_clusters(X, clusters)
        updated_clusters = update_clusters(X, cluster_labels)
        if is_converged(clusters, updated_clusters, eps):
            break
        else:
            clusters = updated_clusters

    # print(cluster_labels)


    # print('\nКонечные центры кластеров')
    return cluster_labels


k = 3
#
# X1 = np.array([[2, 1, 1],
#                [8, 3, 5],
#                [9, 4, 6],
#                [3, 6, 1],
#                [4, 7, 2],
#                [2, 2, 1],
#                [9, 1, 6],
#                [3, 1, 2]])
#
# print(kmeans(X1, 3))

