from sklearn.base import BaseEstimator, ClusterMixin, TransformerMixin
from sklearn.cluster import _k_means
from sklearn.cluster.k_means_ import _tolerance, _labels_inertia, _init_centroids, _check_sample_weight
from sklearn.metrics.pairwise import pairwise_distances_argmin
from sklearn.utils import check_random_state, as_float_array
from sklearn.utils.extmath import row_norms, squared_norm
from sklearn.utils.validation import check_is_fitted
import scipy.sparse as sp
import numpy as np
import pandas as pd


class SubspaceKMeans(BaseEstimator, ClusterMixin, TransformerMixin):
    def __init__(self, k=8, n_init=10, max_iter=300, tol=1e-4, tol_eig=-1e-10, random_state=None):
        self.k = k
        self.n_init = n_init
        self.max_iter = max_iter
        self.tol = tol
        self.tol_eig = tol_eig
        self.random_state = random_state

        # validate parameters
        if self.n_init <= 0:
            raise ValueError('n_init must be bigger than zero.')
        if self.max_iter <= 0:
            raise ValueError('max_iter must be bigger than zero.')

        self.cluster_centers_ = None
        self.labels_ = None
        self.inertia_ = None
        self.feature_importances_ = None
        self.V_ = None
        self.m_ = None

    def subspace_kmeans_single_(self, X, sample_weight, x_squared_norms, tol, random_state):
        random_state = check_random_state(random_state)
        sample_weight = _check_sample_weight(X, sample_weight)
        best_labels, best_inertia, best_centers = None, None, None

        centers = _init_centroids(X, self.k, init='k-means++', random_state=random_state, x_squared_norms=x_squared_norms)
        distances = np.zeros(shape=(X.shape[0],), dtype=X.dtype)

        print("Initialization complete")

        d = X.shape[1]        # dimentionality of original space
        m = d // 2            # dimentionality of clustered space
        SD = np.dot(X.T, X)  # scatter matrix of the dataset in the original space

        # orthonormal matrix of a rigid transformation
        V, _ = np.linalg.qr(random_state.random_sample(d ** 2).reshape(d, d), mode='complete')
        for i in range(self.max_iter):
            # store centers for shift computation
            centers_old = centers.copy()

            PC = np.eye(m, d)
            XC = np.dot(np.dot(X, V), PC.T)
            muC = np.dot(np.dot(centers, V), PC.T)
            labels = pairwise_distances_argmin(XC, muC, metric_kwargs={'squared': True}).astype(np.int32)

            # computation of the means is also called the M-step of EM
            centers = _k_means._centers_dense(X, sample_weight, labels, self.k, distances)

            S = np.zeros((d, d))
            for j in range(self.k):
                Xj = X[:][labels == j] - centers[:][j]  # Xj - muj, for each j
                S += np.dot(Xj.T, Xj)  # S += Sj

            # sorted eigenvalues and eigenvectors of SIGMA=S-S_D (which is symmetric, hence used eigh)
            eig_vals, eig_vecs = np.linalg.eigh(S - SD)
            V = eig_vecs[np.argsort(eig_vals)]
            m = len(np.where(eig_vals < self.tol_eig)[0])
            if m == 0:
                raise ValueError('Dimensionality is 0. The dataset is better explained by a single cluster.')

            inertia = sum([row_norms(X[:][labels == j] - centers[:][j], squared=True).sum() for j in range(self.k)])
            print("Iteration %2d, inertia %.3f" % (i, inertia))

            if best_inertia is None or inertia < best_inertia:
                best_labels = labels.copy()
                best_centers = centers.copy()
                best_inertia = inertia

            center_shift_total = squared_norm(centers_old - centers)
            if center_shift_total <= tol:
                print("Converged at iteration %d: center shift %e within tolerance %e" % (i, center_shift_total, tol))
                break

        if center_shift_total > 0:
            # rerun E-step in case of non-convergence so that predicted labels match cluster centers
            best_labels, best_inertia = _labels_inertia(X, sample_weight, x_squared_norms, best_centers,
                                                        precompute_distances=False, distances=distances)

        return best_centers, best_labels, best_inertia

    def fit(self, X, sample_weight=None):
        if sp.issparse(X):
            raise ValueError('SubspaceKMeans does not support sparse matrix')

        random_state = check_random_state(self.random_state)
        self.cluster_centers_, self.labels_, self.inertia_ = None, None, None

        # subtract of mean of x for more accurate distance computations
        X = as_float_array(X, copy=True)
        X_mean = X.mean(axis=0)
        X -= X_mean

        # precompute squared norms of data points
        x_squared_norms = row_norms(X, squared=True)

        # computes a tolerance which is independent of the dataset
        tol = _tolerance(X, self.tol)

        for seed in random_state.randint(np.iinfo(np.int32).max, size=self.n_init):
            centers, labels, inertia = self.subspace_kmeans_single_(X, sample_weight, x_squared_norms, tol, seed)

            # determine if these results are the best so far
            if self.inertia_ is None or inertia < self.inertia_:
                self.labels_ = labels.copy()
                self.cluster_centers_ = centers.copy()
                self.inertia_ = inertia

        self.cluster_centers_ += X_mean

        # TODO: export to function ===============================================
        d = X.shape[1]
        S_D = np.dot(X.T, X)
        S = np.zeros((d, d))
        for i in range(self.k):
            X_i = X[:][self.labels_ == i] - self.cluster_centers_[:][i]
            S += np.dot(X_i.T, X_i)
        sigma = S - S_D
        self.feature_importances_, self.V_ = np.linalg.eigh(sigma)
        self.m_ = len(np.where(self.feature_importances_ < self.tol_eig)[0])
        # TODO: export to function  ===============================================

        return self

# df = pd.read_csv('wine.csv')
# lab = df.iloc[:, -1].tolist()
# print(list(zip(SubspaceKMeans(k=3).fit(df.iloc[:, :-1]).labels_, lab)))