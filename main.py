import os
import logging
import pandas as pd
import numpy as np
from src.utils import setup_logging, ensure_directories
from src.data_preprocessing import load_data, analyze_data_quality, preprocess_data
from src.clustering import (
    calculate_kmeans_metrics, fit_kmeans, fit_dbscan, 
    fit_hierarchical, run_pca, evaluate_model, analyze_clusters
)
from src.visualization import (
    plot_eda, plot_elbow, plot_kmeans_clusters, 
    plot_pca_clusters, plot_hierarchical_dendrogram, create_plotly_dashboard
)

def assign_business_personas(profile_df: pd.DataFrame) -> pd.DataFrame:
    """
    Dynamically profiles each cluster based on its mean annual income and spending score
    to assign descriptive customer personas and target marketing recommendations.
    
    Parameters:
    -----------
    profile_df : pd.DataFrame
        Profiling table with columns: Average_Income, Average_Spending_Score.
        
    Returns:
    --------
    pd.DataFrame
        DataFrame with added columns 'Persona' and 'Marketing_Strategy'.
    """
    personas = []
    strategies = []
    
    for idx, row in profile_df.iterrows():
        inc = row['Average_Income']
        sp = row['Average_Spending_Score']
        
        if inc > 65 and sp > 65:
            persona = "High Income, High Spending (Stars / Target Group)"
            strategy = "Focus on premium product lines, exclusive loyalty rewards, high-end memberships, VIP events, and personalized high-value services."
        elif inc > 65 and sp < 40:
            persona = "High Income, Low Spending (Careful / Sensible)"
            strategy = "Target with quality assurance campaigns, long-term value propositions, cash-back rewards, and educational marketing about product reliability."
        elif inc < 45 and sp > 65:
            persona = "Low Income, High Spending (Spendthrifts / Careless)"
            strategy = "Promote trendy, budget-friendly items, flash sales, impulse-buy discounts, and installment-based payment options (e.g. BNPL)."
        elif inc < 45 and sp < 40:
            persona = "Low Income, Low Spending (Frugal / Conservative)"
            strategy = "Focus on basic utility items, discount coupons, bulk purchase savings, and cost-effective value packs."
        else:
            persona = "Medium Income, Medium Spending (Standard / Moderate)"
            strategy = "Engage with regular newsletters, generic store discount sales, reward-point programs, and popular mainstream products."
            
        personas.append(persona)
        strategies.append(strategy)
        
    profile_df['Persona'] = personas
    profile_df['Marketing_Strategy'] = strategies
    return profile_df


def main():
    # 1. Setup Directories
    ensure_directories()
    
    # 2. Setup Logging
    logger = setup_logging("output/segmentation.log")
    logger.info("Starting Customer Segmentation Pipeline...")
    
    # Define file paths
    data_path = "data/Mall_Customers.csv"
    output_dir = "output"
    
    try:
        # 3. Load Data
        df = load_data(data_path)
        
        # 4. Check data quality
        quality_info = analyze_data_quality(df)
        logger.info(f"Missing values summary: {quality_info['total_missing']} missing values detected.")
        logger.info(f"Duplicate rows summary: {quality_info['duplicate_count']} duplicate rows detected.")
        
        # 5. Preprocessing & Scaling
        df_clean, scaled_df, scaler, gender_encoder = preprocess_data(df)
        
        # 6. Exploratory Data Analysis Visualizations
        plot_eda(df_clean, output_dir)
        
        # 7. Find Optimal Clusters for K-Means (Elbow & Silhouette)
        k_range, inertias, silhouette_scores = calculate_kmeans_metrics(scaled_df, max_k=10)
        plot_elbow(k_range, inertias, silhouette_scores, output_dir)
        
        # 8. Fit K-Means Clustering (Optimal K is 5 for the Mall Customer structure)
        optimal_k = 5
        kmeans_model, kmeans_labels = fit_kmeans(scaled_df, n_clusters=optimal_k)
        df_clean['KMeans_Cluster'] = kmeans_labels
        
        # 9. Plot K-Means Clusters in original dimensions (Annual Income vs Spending Score)
        plot_kmeans_clusters(df_clean, kmeans_labels, kmeans_model, output_dir)
        
        # 10. Run PCA and Plot PCA-reduced Clusters
        pca_model, pca_df = run_pca(scaled_df, n_components=2)
        plot_pca_clusters(pca_df, kmeans_labels, output_dir)
        
        # 11. Evaluate K-Means Performance
        logger.info("Evaluating K-Means Clustering model:")
        kmeans_eval = evaluate_model(scaled_df, kmeans_labels)
        
        # 12. Profile K-Means Clusters
        cluster_profile = analyze_clusters(df_clean, kmeans_labels)
        cluster_profile = assign_business_personas(cluster_profile)
        
        logger.info("\n================= K-MEANS CUSTOMER CLUSTER PROFILES =================")
        for c_id, row in cluster_profile.iterrows():
            logger.info(f"\nCluster {c_id}:")
            logger.info(f" - Segment Size: {int(row['Customer_Count'])} customers ({row['Percentage_of_Total']:.2f}%)")
            logger.info(f" - Profile: Avg Age = {row['Average_Age']:.1f} | Avg Income = ${row['Average_Income']:.1f}k | Avg Spending Score = {row['Average_Spending_Score']:.1f}")
            logger.info(f" - Persona: {row['Persona']}")
            logger.info(f" - Recommended Marketing Strategy: {row['Marketing_Strategy']}")
        logger.info("====================================================================")
        
        # Save profiles to output
        cluster_profile.to_csv(os.path.join(output_dir, 'customer_segments_summary.csv'))
        logger.info("Saved customer segments summary table to output/customer_segments_summary.csv")
        
        # 13. Create Interactive Plotly Dashboard
        create_plotly_dashboard(df_clean, kmeans_labels, output_dir)
        
        # 14. Bonus: Hierarchical Clustering
        logger.info("Running Hierarchical Clustering...")
        plot_hierarchical_dendrogram(scaled_df, output_dir)
        hc_model, hc_labels = fit_hierarchical(scaled_df, n_clusters=optimal_k)
        logger.info("Evaluating Hierarchical Clustering model:")
        hc_eval = evaluate_model(scaled_df, hc_labels)
        
        # 15. Bonus: DBSCAN Clustering
        logger.info("Running DBSCAN Clustering...")
        # Grid search to find a reasonable eps on standard scaled data (Age, Income, Spending)
        # Typically on 3-scaled dimensions, eps in [0.4, 0.85] works well. Let's use eps=0.6, min_samples=4
        dbscan_model, dbscan_labels = fit_dbscan(scaled_df, eps=0.6, min_samples=4)
        logger.info("Evaluating DBSCAN Clustering model:")
        dbscan_eval = evaluate_model(scaled_df, dbscan_labels)
        
        # 16. Cluster Comparison Table
        comparison_data = {
            'Algorithm': ['K-Means', 'Hierarchical (Ward)', 'DBSCAN'],
            'Clusters Found': [optimal_k, optimal_k, len(set(dbscan_labels)) - (1 if -1 in dbscan_labels else 0)],
            'Silhouette Score': [kmeans_eval['silhouette'], hc_eval['silhouette'], dbscan_eval['silhouette']],
            'Davies-Bouldin Index': [kmeans_eval['davies_bouldin'], hc_eval['davies_bouldin'], dbscan_eval['davies_bouldin']],
            'Calinski-Harabasz Score': [kmeans_eval['calinski_harabasz'], hc_eval['calinski_harabasz'], dbscan_eval['calinski_harabasz']]
        }
        comparison_df = pd.DataFrame(comparison_data)
        logger.info("\n================= ALGORITHM COMPARISON TABLE =================")
        logger.info("\n" + comparison_df.to_string(index=False))
        logger.info("==============================================================")
        
        comparison_df.to_csv(os.path.join(output_dir, 'clustering_algorithm_comparison.csv'), index=False)
        logger.info("Saved clustering algorithm comparison to output/clustering_algorithm_comparison.csv")
        
        logger.info("Customer Segmentation pipeline executed successfully!")
        
    except Exception as e:
        logger.error(f"Execution failed in pipeline: {e}", exc_info=True)
        raise

if __name__ == "__main__":
    main()
