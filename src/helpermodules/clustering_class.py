'''
more info on the theory behind DTW Clustering: https://rtavenar.github.io/blog/dtw.html
'''
import time
import numpy as np
import pandas as pd
from typing import Union, Dict, Optional
from fastdtw import fastdtw
from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score
from sklearn.utils import check_random_state
import matplotlib.pyplot as plt
from joblib import Parallel, delayed  # For parallel distance computations


class TimeSeriesClustering:
    def __init__(self, df: pd.DataFrame, random_state: Optional[int] = 42):
        """
        Initialize clustering object with validation

        Args:
            df: DataFrame with rows as time series (tickers) and columns as timestamps
            random_state: Seed for reproducibility
        """
        self.random_state = random_state
        self.rng = check_random_state(random_state)

        # Validate and store data
        self._validate_data(df)
        self.raw_data = df.copy()
        self.processed_data = None
        self.distance_matrices = {}  # Cache for distance matrices

    def _validate_data(self, df: pd.DataFrame) -> None:
        """Comprehensive data validation"""
        if not isinstance(df, pd.DataFrame):
            raise TypeError("Input must be a pandas DataFrame")

        if df.empty:
            raise ValueError("Input DataFrame is empty")

        if not df.apply(lambda s: pd.api.types.is_numeric_dtype(s)).all():
            raise ValueError("All DataFrame columns must be numeric")

        if df.isna().any().any():
            raise ValueError(
                "Data contains missing values. Handle missing values first.")

        if np.isinf(df.values).any():
            raise ValueError("Data contains infinite values")

    def preprocess(self, method: str = 'pct_change', **kwargs) -> None:
        """
        Preprocess time series data with proper look-forward protection

        Args:
            method: Preprocessing method ('pct_change' or 'standardize')
            kwargs: Additional arguments for preprocessing
        """
        if method == 'pct_change':
            # Compute percentage change across time (axis=1)
            processed = self.raw_data.pct_change(axis=1, **kwargs)

            # Handle NaNs by forward filling (no look-ahead bias)
            processed = processed.ffill(axis=1).bfill(
                axis=1)  # Handle initial NaN

        elif method == 'standardize':
            # Z-score standardization per time series
            means = self.raw_data.mean(axis=1)
            stds = self.raw_data.std(axis=1)
            processed = (self.raw_data.sub(means, axis=0)).div(stds, axis=0)

        elif method == 'returns_standardize':
            # First compute returns
            returns = self.raw_data.pct_change(axis=1, **kwargs)
            # Then standardize the returns
            means = returns.mean(axis=1)
            stds = returns.std(axis=1)
            processed = (returns.sub(means, axis=0)).div(stds, axis=0)
            processed = processed.ffill(axis=1).bfill(axis=1)

        else:
            raise ValueError(f"Unknown preprocessing method: {method}")

        # Remove columns with remaining NaNs
        self.processed_data = processed.dropna(axis=1)

    def _compute_distance_matrix(self, metric: str = 'euclidean', **dtw_args) -> np.ndarray:
        """
        Compute distance matrix with caching and parallel processing

        Args:
            metric: Distance metric ('euclidean' or 'fastdtw')
            dtw_args: Additional arguments for FastDTW
        """
        if metric not in ['euclidean', 'fastdtw']:
            raise ValueError(f"Unsupported distance metric: {metric}")

        # Check cache first
        cache_key = f"{metric}_{hash(frozenset(dtw_args.items()))}"
        if cache_key in self.distance_matrices:
            return self.distance_matrices[cache_key]

        data = self.processed_data.values

        if metric == 'euclidean':
            distances = pdist(data, metric='euclidean')
        else:
            # Parallel FastDTW computation
            n_samples = data.shape[0]
            indices = np.triu_indices(n_samples, k=1)

            def _parallel_dtw(i, j):
                return fastdtw(data[i], data[j], **dtw_args)[0]

            results = Parallel(n_jobs=-1)(
                delayed(_parallel_dtw)(i, j)
                for i, j in zip(indices[0], indices[1])
            )

            distances = np.array(results)

        # Cache and return squareform matrix
        self.distance_matrices[cache_key] = squareform(distances)
        return self.distance_matrices[cache_key]

    def hierarchical_clustering(
        self,
        n_clusters: int,
        metric: str = 'euclidean',
        linkage_method: str = 'ward',
        **dtw_args
    ) -> Dict:
        """Hierarchical clustering with automatic method selection"""
        if metric == 'fastdtw' and linkage_method == 'ward':
            raise ValueError(
                "Ward linkage can't be used with non-Euclidean distances")

        distance_matrix = self._compute_distance_matrix(metric, **dtw_args)
        Z = linkage(distance_matrix, method=linkage_method)
        clusters = fcluster(Z, n_clusters, criterion='maxclust')

        # Silhouette score calculation
        sil_score = self._calculate_silhouette(
            clusters, distance_matrix, metric)

        return {
            'linkage_matrix': Z,
            'clusters': clusters,
            'silhouette_score': sil_score,
            'n_clusters': len(np.unique(clusters))
        }

    def kmeans_clustering(
        self,
        n_clusters: int,
        init: str = 'k-means++',
        n_init: int = 10,
        **kwargs
    ) -> Dict:
        """KMeans clustering with proper initialization"""
        model = KMeans(
            n_clusters=n_clusters,
            init=init,
            n_init=n_init,
            random_state=self.random_state,
            **kwargs
        )

        clusters = model.fit_predict(self.processed_data.values)

        return {
            'clusters': clusters,
            'silhouette_score': silhouette_score(
                self.processed_data.values,
                clusters,
                metric='euclidean'
            ),
            'inertia': model.inertia_,
            'n_clusters': n_clusters
        }

    def dbscan_clustering(
        self,
        eps: float,
        min_samples: int = 5,
        metric: str = 'euclidean',
        **dtw_args
    ) -> Dict:
        """DBSCAN clustering with proper distance matrix handling"""
        distance_matrix = self._compute_distance_matrix(metric, **dtw_args)
        model = DBSCAN(
            eps=eps,
            min_samples=min_samples,
            metric='precomputed',
            n_jobs=-1
        )

        clusters = model.fit_predict(distance_matrix)
        n_clusters = len(np.unique(clusters)) - (1 if -1 in clusters else 0)

        return {
            'clusters': clusters,
            'silhouette_score': self._calculate_silhouette(clusters, distance_matrix, metric),
            'n_clusters': n_clusters,
            'core_sample_indices': model.core_sample_indices_
        }

    def _calculate_silhouette(
        self,
        clusters: np.ndarray,
        distance_matrix: np.ndarray,
        metric: str
    ) -> float:
        """Safe silhouette score calculation"""
        unique_clusters = np.unique(clusters)
        if len(unique_clusters) < 2:
            return np.nan

        # Handle noise points in DBSCAN
        mask = clusters != -1 if -1 in clusters else slice(None)

        if metric == 'euclidean':
            return silhouette_score(
                self.processed_data.values[mask],
                clusters[mask],
                metric='euclidean'
            )
        else:
            return silhouette_score(
                distance_matrix[mask][:, mask],
                clusters[mask],
                metric='precomputed'
            )

    def plot_elbow(self, max_clusters: int = 15) -> None:
        """Improved elbow method with automatic knee detection"""
        inertias = []
        for k in range(1, max_clusters + 1):
            model = KMeans(
                n_clusters=k,
                random_state=self.random_state,
                n_init='auto'
            ).fit(self.processed_data.values)
            inertias.append(model.inertia_)

        plt.figure(figsize=(10, 6))
        plt.plot(range(1, max_clusters + 1), inertias, 'bo-')
        plt.xlabel('Number of Clusters')
        plt.ylabel('Inertia')
        plt.title('Elbow Method for Optimal K')
        plt.grid(True)
        plt.show()

    def plot_dendrogram(
        self,
        linkage_matrix: np.ndarray,
        title: str = "Hierarchical Clustering Dendrogram",
        **kwargs
    ) -> None:
        """Enhanced dendrogram visualization"""
        plt.figure(figsize=(15, 8))
        dendrogram(
            linkage_matrix,
            labels=self.processed_data.index.tolist(),
            orientation='top',
            **kwargs
        )
        plt.title(title)
        plt.xlabel('Sample Index')
        plt.ylabel('Distance')
        plt.xticks(rotation=90)
        plt.tight_layout()
        plt.show()

    def analyze_clusters(self, clusters: np.ndarray) -> pd.DataFrame:
        """Comprehensive cluster analysis"""
        analysis = []
        for cluster_id in np.unique(clusters):
            mask = clusters == cluster_id
            cluster_data = self.raw_data[mask]

            stats = {
                'cluster': cluster_id,
                'size': cluster_data.shape[0],
                'mean_return': cluster_data.pct_change(axis=1).mean().mean(),
                'volatility': cluster_data.pct_change(axis=1).std().mean(),
                'total_return': (cluster_data.iloc[:, -1] / cluster_data.iloc[:, 0] - 1).mean()
            }

            analysis.append(stats)

        return pd.DataFrame(analysis).set_index('cluster')


def plot_cluster_timeseries(df_actual, df_pct, clustering_columns):
    """
    Plots time series of tickers in each cluster (one image per cluster).

    Each image has two subplots:
    - Top: actual values
    - Bottom: percentage change (limited to [-1, 1] on y-axis)

    Args:
    - df_actual (pd.DataFrame): Actual values (tickers x timestamps + cluster column).
    - df_pct (pd.DataFrame): Percentage change values (tickers x timestamps + cluster column).
    - clustering_columns (list): Column names indicating clustering assignments.

    Returns:
    - None
    """

    for clustering_col in clustering_columns:
        clusters = df_actual[clustering_col].unique()

        for cluster in clusters:
            # Filter data for current cluster
            tickers_in_cluster = df_actual[df_actual[clustering_col]
                                           == cluster].index
            df_cluster_actual = df_actual.loc[tickers_in_cluster].drop(columns=[
                                                                       clustering_col])
            df_cluster_pct = df_pct.loc[tickers_in_cluster].drop(
                columns=[clustering_col])

            fig, axes = plt.subplots(
                nrows=2, ncols=1, figsize=(10, 6), sharex=True)
            fig.suptitle(
                f"{clustering_col} - Cluster {cluster + 1}", fontsize=16)

            # Actual values plot
            if not df_cluster_actual.empty:
                df_cluster_actual.transpose().plot(ax=axes[0], legend=False)
                axes[0].set_title("Actual Values")
                axes[0].set_ylabel('')
                axes[0].set_xticks([])
                axes[0].set_xticklabels([])

            # % Change plot
            if not df_cluster_pct.empty:
                df_cluster_pct.transpose().plot(ax=axes[1], legend=False)
                axes[1].set_title("% Change")
                axes[1].set_ylim(-0.5, 0.5)
                axes[1].set_ylabel('')
                axes[1].set_xticks([])
                axes[1].set_xticklabels([])

            plt.tight_layout(pad=2.0, rect=[0, 0, 1, 0.95])
            plt.show()
