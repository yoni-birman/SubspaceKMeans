from sklearn.base import BaseEstimator, ClusterMixin, TransformerMixin
from sklearn.cluster import KMeans, _k_means
from sklearn.cluster.k_means_ import _tolerance, _labels_inertia, _init_centroids, _check_sample_weight
from sklearn.metrics import normalized_mutual_info_score, pairwise_distances_argmin
from sklearn.preprocessing import scale
from sklearn.utils import check_random_state, as_float_array
from sklearn.utils.extmath import row_norms, squared_norm
from datetime import datetime
from scipy import sparse
import pandas as pd
import numpy as np
import os


class SubKMeans(BaseEstimator, ClusterMixin, TransformerMixin):
    def __init__(self, n_clusters, n_init=10, max_iter=300, tol=1e-4, tol_eig=-1e-10, random_state=None):
        # validate parameters
        if n_init <= 0:
            raise ValueError('n_init must be bigger than zero.')
        if max_iter <= 0:
            raise ValueError('max_iter must be bigger than zero.')

        self.cluster_centers_, self.labels_, self.inertia_ = None, None, None
        self.V_, self.m_ = None, None
        self.n_clusters = n_clusters      # The number of clusters
        self.n_init = n_init              # Number of time the algorithm will be run with different centroid seeds
        self.max_iter = max_iter          # Maximum number of iterations of the k-means algorithm for a single run
        self.tol = tol                    # Relative tolerance with regards to inertia to declare convergence
        self.tol_eig = tol_eig            # Relative tolerance with regards to eigenvalues
        self.random_state = random_state

    def fit(self, X, sample_weight=None):
        if sparse.issparse(X):
            raise ValueError('Data cannot be sparse.')

        random_state = check_random_state(self.random_state)
        self.cluster_centers_, self.labels_, self.inertia_ = None, None, None

        # Scale the data to zero mean and unit varience
        X = scale(as_float_array(X, copy=True)) #TODO: BOM
        # X = as_float_array(X, copy=True)
        # X_mean = X.mean()
        # X -= X_mean

        # precompute squared norms of data points
        x_squared_norms = row_norms(X, squared=True)

        # precomputes a tolerance which is independent of the dataset
        tol = _tolerance(X, self.tol)

        for seed in random_state.randint(np.iinfo(np.int32).max, size=self.n_init):
            centers, labels, inertia = self.sub_kmeans_single_(X, sample_weight, x_squared_norms, tol, seed)

            # determine if these results are the best so far
            if self.inertia_ is None or inertia < self.inertia_:
                self.labels_ = labels.copy()
                self.cluster_centers_ = centers.copy()
                self.inertia_ = inertia

        # self.cluster_centers_ += X_mean

        SD = np.dot(X.T, X)
        S = self.update_step_(X, self.cluster_centers_, self.labels_)
        self.V_, self.m_ = self.eigen_decomposition_(S - SD)
        return self

    def fit_predict(self, X, y=None):
        return self.fit(X).labels_

    def sub_kmeans_single_(self, X, sample_weight, x_squared_norms, tol, random_state):
        random_state = check_random_state(random_state)
        sample_weight = _check_sample_weight(X, sample_weight)
        best_labels, best_inertia, best_centers = None, None, None

        distances = np.zeros(shape=(X.shape[0],), dtype=X.dtype)
        centers = _init_centroids(X, self.n_clusters, init='k-means++', random_state=random_state,
                                  x_squared_norms=x_squared_norms)

        d = X.shape[1]        # dimentionality of original space
        m = d // 2            # dimentionality of clustered space
        SD = np.dot(X.T, X)   # scatter matrix of the dataset in the original space

        # orthonormal matrix of a rigid transformation
        V, _ = np.linalg.qr(random_state.random_sample(d ** 2).reshape(d, d), mode='complete')
        for i in range(self.max_iter):
            centers_old = centers.copy()

            # get the clusters' labels
            labels = self.assignment_step_(X=X, V=V, centers=centers, m=m)

            # compute new centers and sum the clusters' scatter matrices
            centers = _k_means._centers_dense(X, sample_weight, labels, self.n_clusters, distances)
            S = self.update_step_(X, centers, labels)

            # sorted eigenvalues and eigenvectors of SIGMA=S-SD
            V, m = self.eigen_decomposition_(S - SD)
            if m == 0:
                raise ValueError('Might be a single cluster (m = 0).')

            # inertia - sum of squared distances of samples to their closest cluster center
            inertia = sum([row_norms(X[labels == j] - centers[j], squared=True).sum() for j in range(self.n_clusters)])

            # print("Iteration %2d, inertia %.3f" % (i, inertia))
            if best_inertia is None or inertia < best_inertia:
                best_labels = labels.copy()
                best_centers = centers.copy()
                best_inertia = inertia

            center_shift_total = squared_norm(centers_old - centers)
            if center_shift_total <= tol:
                # print("Converged at iteration %d: center shift %e within tolerance %e" % (i, center_shift_total, tol))
                break

        if center_shift_total > 0:
            # rerun E-step in case of non-convergence so that predicted labels match cluster centers
            best_labels, best_inertia = _labels_inertia(X, sample_weight, x_squared_norms, best_centers,
                                                        precompute_distances=False, distances=distances)

        return best_centers, best_labels, best_inertia

    def assignment_step_(self, X, V, centers, m):
        Pc = np.eye(X.shape[1], m)
        PcV = np.dot(Pc.T, V.T)
        PcVX = np.dot(PcV, X.T)
        PcVmu = np.dot(PcV, centers.T)

        labels = pairwise_distances_argmin(PcVX.T, PcVmu.T, metric_kwargs={'squared': True}).astype(np.int32)
        return labels

    def update_step_(self, X, centers, labels):
        d = X.shape[1]
        S = np.zeros((d, d))
        for i in range(self.n_clusters):
            SiX = X[labels == i] - centers[i]
            S += np.dot(SiX.T, SiX)  # S += Si
        return S

    def eigen_decomposition_(self, sigma):
        eig_vals, eig_vecs = np.linalg.eigh(sigma)
        V = eig_vecs[np.argsort(eig_vals)]
        m = len(np.where(eig_vals < self.tol_eig)[0])
        return V, m


def nmi_with_time(model, X, y, k):
    s = datetime.now()
    m = model(n_clusters=k, random_state=1000).fit(X)
    train_t = (datetime.now() - s).total_seconds() * 1000  # convert to milisec
    nmi = normalized_mutual_info_score(y, m.labels_, average_method='arithmetic')
    return nmi, train_t, m


def evaluate(name):
    # Load the data
    df = pd.read_csv('data/%s' % name, header=None)

    # Factorize nominal columns
    for col in df.columns:
        if df[col].dtype == 'object':
            df[col] = pd.factorize(df[col])[0]

    # Split data and labels, and get k (number of clusters)
    X, y = df.iloc[:, :-1], df.iloc[:, -1]
    k = y.unique().shape[0]

    # Get results for KMeans and SubKMeans
    km_nmi, km_t, _ = nmi_with_time(KMeans, X, y, k)
    skm_nmi, skm_t, skm = nmi_with_time(SubKMeans, X, y, k)

    # Print results
    print("------ %s ------" % name)
    print("|D|=%d, k=%d, d=%d, m=%d" % (X.shape[0], k, X.shape[1], skm.m_))
    print("\t\t\tNMI\t\tTIME")
    print("SubKMeans\t%.3f\t%.3f" % (skm_nmi, skm_t))
    print("   KMeans\t%.3f\t%.3f" % (km_nmi, km_t))

if __name__ == "__main__":
    for f in os.listdir('data'):
        evaluate(f)
