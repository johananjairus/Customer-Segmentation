import numpy as np
import pandas as pd
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.decomposition import PCA
import logging
from typing import Dict, Any, Tuple, List

logger = logging.getLogger("CustomerSegmentation.clustering")

def calculate_kmeans_metrics(scaled_data: pd.DataFrame, max_k: int = 10) -> Tuple[List[int], List[float], List[float]]:
    """
    Calculates K-Means inertia and silhouette scores for range of K to support Elbow/Silhouette analysis.
    
    Parameters:
    -----------
    scaled_data : pd.DataFrame
        Scaled numerical features.
    max_k : int
        Maximum number of clusters to test. Default is 10.
        
    Returns:
    --------
    Tuple[List[int], List[float], List[float]]
        - k_range: list of tested cluster sizes (2 to max_k).
        - inertias: list of K-Means inertias.
        - silhouette_scores: list of silhouette scores.
    """
    logger.info("Computing K-Means inertia and silhouette scores for range of K...")
    k_range = list(range(2, max_k + 1))
    inertias = []
    silhouette_vals = []
    
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(scaled_data)
        inertias.append(kmeans.inertia_)
        score = silhouette_score(scaled_data, kmeans.labels_)
        silhouette_vals.append(score)
        logger.info(f" - K={k}: Inertia={kmeans.inertia_:.2f}, Silhouette={score:.4f}")
        
    return k_range, inertias, silhouette_vals


def fit_kmeans(scaled_data: pd.DataFrame, n_clusters: int) -> Tuple[KMeans, np.ndarray]:
    """
    Fits K-Means clustering algorithm on scaled data.
    
    Parameters:
    -----------
    scaled_data : pd.DataFrame
        Scaled features.
    n_clusters : int
        Number of clusters to create.
        
    Returns:
    --------
    Tuple[KMeans, np.ndarray]
        - Fitted KMeans model
        - Cluster label assignments
    """
    logger.info(f"Fitting K-Means with {n_clusters} clusters...")
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(scaled_data)
    logger.info("K-Means fit complete.")
    return kmeans, labels


def fit_dbscan(scaled_data: pd.DataFrame, eps: float = 0.5, min_samples: int = 5) -> Tuple[DBSCAN, np.ndarray]:
    """
    Fits DBSCAN clustering algorithm.
    
    Parameters:
    -----------
    scaled_data : pd.DataFrame
        Scaled features.
    eps : float
        The maximum distance between two samples for one to be considered as in the neighborhood of the other.
    min_samples : int
        The number of samples in a neighborhood for a point to be considered as a core point.
        
    Returns:
    --------
    Tuple[DBSCAN, np.ndarray]
        - Fitted DBSCAN model
        - Cluster label assignments
    """
    logger.info(f"Fitting DBSCAN with eps={eps}, min_samples={min_samples}...")
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    labels = dbscan.fit_predict(scaled_data)
    unique_labels = len(set(labels)) - (1 if -1 in labels else 0)
    logger.info(f"DBSCAN fit complete. Identified {unique_labels} clusters and {np.sum(labels == -1)} noise points.")
    return dbscan, labels


def fit_hierarchical(scaled_data: pd.DataFrame, n_clusters: int) -> Tuple[AgglomerativeClustering, np.ndarray]:
    """
    Fits Hierarchical (Agglomerative) clustering algorithm.
    
    Parameters:
    -----------
    scaled_data : pd.DataFrame
        Scaled features.
    n_clusters : int
        Number of clusters.
        
    Returns:
    --------
    Tuple[AgglomerativeClustering, np.ndarray]
        - Fitted AgglomerativeClustering model
        - Cluster label assignments
    """
    logger.info(f"Fitting Hierarchical clustering with {n_clusters} clusters...")
    hc = AgglomerativeClustering(n_clusters=n_clusters, linkage='ward')
    labels = hc.fit_predict(scaled_data)
    logger.info("Hierarchical clustering fit complete.")
    return hc, labels


def run_pca(scaled_data: pd.DataFrame, n_components: int = 2) -> Tuple[PCA, pd.DataFrame]:
    """
    Performs Principal Component Analysis (PCA) to reduce data dimensions.
    
    Parameters:
    -----------
    scaled_data : pd.DataFrame
        Scaled feature dataframe.
    n_components : int
        Number of dimensions. Default is 2.
        
    Returns:
    --------
    Tuple[PCA, pd.DataFrame]
        - Fitted PCA object
        - Dataframe containing principal components
    """
    logger.info(f"Applying PCA dimensionality reduction to {n_components}D...")
    pca = PCA(n_components=n_components, random_state=42)
    pca_result = pca.fit_transform(scaled_data)
    
    col_names = [f"PC{i+1}" for i in range(n_components)]
    pca_df = pd.DataFrame(pca_result, columns=col_names)
    
    explained_var = pca.explained_variance_ratio_
    logger.info(f"PCA complete. Explained variance ratio: {explained_var} (Total: {sum(explained_var):.4f})")
    
    return pca, pca_df


def evaluate_model(scaled_data: pd.DataFrame, labels: np.ndarray) -> Dict[str, float]:
    """
    Calculates clustering performance evaluation metrics.
    
    Note: Metric evaluations are skipped if only 1 cluster (or noise only) is found.
    
    Parameters:
    -----------
    scaled_data : pd.DataFrame
        Scaled features.
    labels : np.ndarray
        Predicted cluster labels.
        
    Returns:
    --------
    Dict[str, float]
        Metrics: Silhouette Score, Davies-Bouldin Index, Calinski-Harabasz Score.
    """
    unique_labels = np.unique(labels)
    # Exclude noise label (-1) from evaluations if DBSCAN is used
    eval_labels = labels[labels != -1]
    eval_data = scaled_data[labels != -1]
    
    n_clusters = len(np.unique(eval_labels))
    
    if n_clusters < 2:
        logger.warning("Clustering has less than 2 distinct non-noise clusters. Cannot evaluate metrics.")
        return {
            "silhouette": -1.0,
            "davies_bouldin": np.nan,
            "calinski_harabasz": np.nan
        }
        
    silhouette = silhouette_score(eval_data, eval_labels)
    db_index = davies_bouldin_score(eval_data, eval_labels)
    ch_score = calinski_harabasz_score(eval_data, eval_labels)
    
    metrics = {
        "silhouette": silhouette,
        "davies_bouldin": db_index,
        "calinski_harabasz": ch_score
    }
    
    logger.info(f"Evaluation Metrics:")
    logger.info(f" - Silhouette Score: {silhouette:.4f} (Higher is better, range [-1, 1])")
    logger.info(f" - Davies-Bouldin Index: {db_index:.4f} (Lower is better, minimum 0)")
    logger.info(f" - Calinski-Harabasz Score: {ch_score:.4f} (Higher is better)")
    
    return metrics


def analyze_clusters(df: pd.DataFrame, labels: np.ndarray) -> pd.DataFrame:
    """
    Aggregates profile statistics (mean Age, Income, Spending, count, and percentage) for each cluster.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Original dataframe containing features ('Age', 'Annual Income (k$)', 'Spending Score (1-100)', etc.)
    labels : np.ndarray
        Cluster labels.
        
    Returns:
    --------
    pd.DataFrame
        Aggregated profiling table.
    """
    logger.info("Analyzing and profiling cluster characteristics...")
    df_copy = df.copy()
    df_copy['Cluster'] = labels
    
    # Calculate counts and percentages
    cluster_counts = df_copy['Cluster'].value_counts()
    cluster_pcts = df_copy['Cluster'].value_counts(normalize=True) * 100
    
    # Aggregate statistics
    profile = df_copy.groupby('Cluster').agg({
        'Age': 'mean',
        'Annual Income (k$)': 'mean',
        'Spending Score (1-100)': 'mean',
    }).rename(columns={
        'Age': 'Average_Age',
        'Annual Income (k$)': 'Average_Income',
        'Spending Score (1-100)': 'Average_Spending_Score'
    })
    
    profile['Customer_Count'] = cluster_counts
    profile['Percentage_of_Total'] = cluster_pcts
    
    # Sort by cluster ID
    profile = profile.sort_index()
    
    return profile
