import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import dendrogram, linkage
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import logging
from typing import List

logger = logging.getLogger("CustomerSegmentation.visualization")

# Configure seaborn styling for professional look
sns.set_theme(style="whitegrid")
plt.rcParams['font.sans-serif'] = 'Arial'
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['figure.titlesize'] = 16
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['axes.labelsize'] = 12

def plot_eda(df: pd.DataFrame, output_dir: str) -> None:
    """
    Generates exploratory data analysis plots and saves them to the output directory.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Cleaned dataset containing original numerical columns and Gender.
    output_dir : str
        Directory where plots should be saved.
    """
    logger.info("Generating EDA visualizations...")
    os.makedirs(output_dir, exist_ok=True)
    
    numeric_cols = ['Age', 'Annual Income (k$)', 'Spending Score (1-100)']
    
    # 1. Distributions of numerical features (Histograms with KDE)
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    colors = ['#4A90E2', '#50E3C2', '#D0021B']
    for idx, col in enumerate(numeric_cols):
        sns.histplot(df[col], kde=True, ax=axes[idx], color=colors[idx], bins=20)
        axes[idx].set_title(f'Distribution of {col}', fontweight='bold')
        axes[idx].set_xlabel(col)
        axes[idx].set_ylabel('Count')
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, 'distributions.png'), dpi=300)
    plt.close(fig)
    
    # 2. Gender Count Plot
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.countplot(x='Gender', data=df, ax=ax, palette=['#F5A623', '#4A90E2'])
    ax.set_title('Customer Count by Gender', fontweight='bold')
    ax.set_xlabel('Gender')
    ax.set_ylabel('Count')
    # Add counts on top of bars
    for p in ax.patches:
        ax.annotate(f'{int(p.get_height())}', (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha='center', va='center', xytext=(0, 5), textcoords='offset points', fontweight='bold')
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, 'gender_distribution.png'), dpi=300)
    plt.close(fig)
    
    # 3. Correlation Heatmap
    fig, ax = plt.subplots(figsize=(8, 6))
    # Select numeric columns including encoded gender
    corr_cols = numeric_cols + ['Gender_Encoded']
    corr_matrix = df[corr_cols].corr()
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5, ax=ax, vmin=-1, vmax=1)
    ax.set_title('Feature Correlation Matrix', fontweight='bold', pad=15)
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, 'correlation_heatmap.png'), dpi=300)
    plt.close(fig)
    
    # 4. Boxplots of features by Gender
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for idx, col in enumerate(numeric_cols):
        sns.boxplot(x='Gender', y=col, data=df, ax=axes[idx], palette=['#F5A623', '#4A90E2'])
        axes[idx].set_title(f'{col} by Gender', fontweight='bold')
        axes[idx].set_xlabel('Gender')
        axes[idx].set_ylabel(col)
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, 'gender_boxplots.png'), dpi=300)
    plt.close(fig)
    
    # 5. Pairplot (Required output)
    pair_plot = sns.pairplot(df[numeric_cols + ['Gender']], hue='Gender', palette=['#F5A623', '#4A90E2'], height=2.5)
    pair_plot.fig.suptitle('Pairplot of Customer Features', y=1.02, fontweight='bold')
    pair_plot.savefig(os.path.join(output_dir, 'pairplot.png'), dpi=300)
    plt.close()
    
    logger.info("EDA visualizations saved successfully.")


def plot_elbow(k_range: List[int], inertias: List[float], silhouette_scores: List[float], output_dir: str) -> None:
    """
    Generates Elbow and Silhouette score plot for determining optimal clusters.
    
    Parameters:
    -----------
    k_range : List[int]
        Range of K values.
    inertias : List[float]
        Inertias for each K.
    silhouette_scores : List[float]
        Silhouette scores for each K.
    output_dir : str
        Directory to save plot.
    """
    logger.info("Generating Elbow Method and Silhouette evaluation plot...")
    os.makedirs(output_dir, exist_ok=True)
    
    fig, ax1 = plt.subplots(figsize=(10, 5))
    
    # Plot Inertia (Elbow Method)
    color = '#E74C3C'
    ax1.set_xlabel('Number of Clusters (k)', fontweight='bold')
    ax1.set_ylabel('Inertia (Within-Cluster Sum of Squares)', color=color, fontweight='bold')
    ax1.plot(k_range, inertias, 'o-', color=color, linewidth=2, markersize=8, label='Inertia')
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.set_xticks(k_range)
    ax1.grid(True, linestyle='--', alpha=0.6)
    
    # Instantiate a second axes that shares the same x-axis
    ax2 = ax1.twinx()  
    color = '#2ECC71'
    ax2.set_ylabel('Silhouette Score', color=color, fontweight='bold')
    ax2.plot(k_range, silhouette_scores, 's--', color=color, linewidth=2, markersize=8, label='Silhouette')
    ax2.tick_params(axis='y', labelcolor=color)
    
    plt.title('Determining Optimal Clusters: Elbow Method & Silhouette Score', fontweight='bold', pad=15)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, 'elbow_method.png'), dpi=300)
    plt.close(fig)
    logger.info("Elbow plot saved as output/elbow_method.png")


def plot_kmeans_clusters(df: pd.DataFrame, labels: np.ndarray, model, output_dir: str) -> None:
    """
    Creates 2D cluster scatter plot using original features (Annual Income vs Spending Score)
    including cluster centroids.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Cleaned original dataframe.
    labels : np.ndarray
        Cluster labels from K-Means.
    model : KMeans
        Fitted K-Means model to extract centroids.
    output_dir : str
        Directory to save plot.
    """
    logger.info("Generating 2D K-Means cluster scatter plot...")
    os.makedirs(output_dir, exist_ok=True)
    
    df_plot = df.copy()
    df_plot['Cluster'] = labels
    
    # Get centroids (need to unscale them if they were scaled, but if model was trained on scaled data,
    # we unscale centroids using standard math or project centroids)
    # Centroids in scaled space
    scaled_centroids = model.cluster_centers_
    
    # Centroids are in [Age, Income, Spending] space.
    # To plot them on [Income vs Spending], we select indices 1 and 2
    # But wait, we want to plot the centroids in the actual scale. Let's unscale them:
    # Preprocessing scaler needs to be passed, but we can compute cluster centroids in original space directly:
    centroids_orig = df_plot.groupby('Cluster')[['Annual Income (k$)', 'Spending Score (1-100)']].mean().values
    
    unique_clusters = sorted(np.unique(labels))
    n_clusters = len(unique_clusters)
    
    fig, ax = plt.subplots(figsize=(10, 7))
    
    # Modern color palette for clusters
    colors = ['#FF5A5F', '#3B5998', '#00A699', '#FC642D', '#767676', '#484848', '#8A2BE2']
    cluster_colors = colors[:n_clusters]
    
    for i, cluster in enumerate(unique_clusters):
        cluster_data = df_plot[df_plot['Cluster'] == cluster]
        ax.scatter(cluster_data['Annual Income (k$)'], 
                   cluster_data['Spending Score (1-100)'], 
                   s=60, 
                   color=cluster_colors[i], 
                   label=f'Cluster {cluster}', 
                   alpha=0.8,
                   edgecolors='black',
                   linewidths=0.5)
        
    # Plot centroids
    ax.scatter(centroids_orig[:, 0], 
               centroids_orig[:, 1], 
               s=250, 
               color='yellow', 
               marker='*', 
               edgecolors='black', 
               linewidths=1.5, 
               label='Centroids')
    
    ax.set_title('Customer Segmentation Clusters (K-Means)', fontweight='bold', fontsize=15, pad=15)
    ax.set_xlabel('Annual Income (k$)', fontweight='bold')
    ax.set_ylabel('Spending Score (1-100)', fontweight='bold')
    ax.legend(frameon=True, facecolor='white', edgecolor='gray')
    ax.grid(True, linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, 'cluster_plot.png'), dpi=300)
    plt.close(fig)
    logger.info("Cluster plot saved as output/cluster_plot.png")


def plot_pca_clusters(pca_df: pd.DataFrame, labels: np.ndarray, output_dir: str) -> None:
    """
    Plots cluster distribution on 2D PCA projected space.
    
    Parameters:
    -----------
    pca_df : pd.DataFrame
        Dataframe containing columns PC1 and PC2.
    labels : np.ndarray
        Cluster labels.
    output_dir : str
        Directory to save plot.
    """
    logger.info("Generating 2D PCA cluster scatter plot...")
    os.makedirs(output_dir, exist_ok=True)
    
    df_plot = pca_df.copy()
    df_plot['Cluster'] = labels
    
    # Exclude noise points (-1) from PCA centroid calculation if DBSCAN
    unique_clusters = sorted([c for c in np.unique(labels) if c != -1])
    n_clusters = len(unique_clusters)
    
    fig, ax = plt.subplots(figsize=(10, 7))
    
    # Use standard color scheme
    colors = ['#FF5A5F', '#3B5998', '#00A699', '#FC642D', '#767676', '#484848', '#8A2BE2']
    
    # Plot noise if present (DBSCAN)
    if -1 in labels:
        noise_data = df_plot[df_plot['Cluster'] == -1]
        ax.scatter(noise_data['PC1'], noise_data['PC2'], s=30, color='#BDC3C7', label='Noise / Outliers', alpha=0.5, marker='x')
        
    for i, cluster in enumerate(unique_clusters):
        cluster_data = df_plot[df_plot['Cluster'] == cluster]
        ax.scatter(cluster_data['PC1'], 
                   cluster_data['PC2'], 
                   s=60, 
                   color=colors[i % len(colors)], 
                   label=f'Cluster {cluster}', 
                   alpha=0.8,
                   edgecolors='black',
                   linewidths=0.5)
        
    ax.set_title('Clustering Visualization in PCA 2D Space', fontweight='bold', fontsize=15, pad=15)
    ax.set_xlabel('Principal Component 1 (PC1)', fontweight='bold')
    ax.set_ylabel('Principal Component 2 (PC2)', fontweight='bold')
    ax.legend(frameon=True, facecolor='white', edgecolor='gray')
    ax.grid(True, linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, 'pca_cluster_plot.png'), dpi=300)
    plt.close(fig)
    logger.info("PCA cluster plot saved as output/pca_cluster_plot.png")


def plot_hierarchical_dendrogram(scaled_data: pd.DataFrame, output_dir: str) -> None:
    """
    Generates and saves hierarchical clustering dendrogram.
    
    Parameters:
    -----------
    scaled_data : pd.DataFrame
        Scaled feature dataset.
    output_dir : str
        Directory to save dendrogram image.
    """
    logger.info("Generating Hierarchical clustering dendrogram...")
    os.makedirs(output_dir, exist_ok=True)
    
    # Calculate linkage matrix
    linked = linkage(scaled_data, method='ward')
    
    fig, ax = plt.subplots(figsize=(12, 6))
    dendrogram(linked,
               orientation='top',
               distance_sort='descending',
               show_leaf_counts=True,
               ax=ax,
               no_labels=True)  # Hide individual indices as they clutter the plot
               
    ax.set_title('Hierarchical Clustering Dendrogram (Ward Linkage)', fontweight='bold', fontsize=15, pad=15)
    ax.set_xlabel('Customers', fontweight='bold')
    ax.set_ylabel('Euclidean Distance', fontweight='bold')
    
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, 'dendrogram.png'), dpi=300)
    plt.close(fig)
    logger.info("Dendrogram plot saved as output/dendrogram.png")


def create_plotly_dashboard(df: pd.DataFrame, labels: np.ndarray, output_dir: str) -> None:
    """
    Creates an interactive HTML dashboard using Plotly.
    Also saves a static PNG of the dashboard.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Cleaned original dataframe.
    labels : np.ndarray
        Cluster labels assigned to customers.
    output_dir : str
        Directory to save the dashboard assets.
    """
    logger.info("Generating Plotly interactive dashboard...")
    os.makedirs(output_dir, exist_ok=True)
    
    df_dash = df.copy()
    df_dash['Cluster'] = labels.astype(str)
    
    # Order categories so legend stays sequential
    df_dash = df_dash.sort_values(by='Cluster')
    
    # Define color sequence
    colors = ['#FF5A5F', '#3B5998', '#00A699', '#FC642D', '#767676', '#484848', '#8A2BE2']
    color_map = {str(c): colors[i % len(colors)] for i, c in enumerate(sorted(np.unique(labels)))}
    
    # Create interactive multi-plot layout
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            '<b>Annual Income vs Spending Score</b>', 
            '<b>Age Distribution by Cluster</b>',
            '<b>Annual Income Distribution</b>', 
            '<b>Customer Segment Distribution</b>'
        ),
        specs=[[{"type": "xy"}, {"type": "xy"}],
               [{"type": "xy"}, {"type": "domain"}]],
        vertical_spacing=0.15,
        horizontal_spacing=0.10
    )
    
    # 1. Scatter plot: Annual Income vs Spending Score
    for cluster in sorted(df_dash['Cluster'].unique()):
        sub_df = df_dash[df_dash['Cluster'] == cluster]
        fig.add_trace(
            go.Scatter(
                x=sub_df['Annual Income (k$)'],
                y=sub_df['Spending Score (1-100)'],
                mode='markers',
                marker=dict(size=9, color=color_map[cluster], line=dict(width=0.5, color='black')),
                name=f'Cluster {cluster}',
                legendgroup=f'Cluster {cluster}',
                text=f"ID: " + sub_df['CustomerID'].astype(str) + "<br>Age: " + sub_df['Age'].astype(str)
            ),
            row=1, col=1
        )
        
    # 2. Boxplot: Age distribution per cluster
    for cluster in sorted(df_dash['Cluster'].unique()):
        sub_df = df_dash[df_dash['Cluster'] == cluster]
        fig.add_trace(
            go.Box(
                y=sub_df['Age'],
                name=f'Cluster {cluster}',
                marker_color=color_map[cluster],
                boxmean=True,
                showlegend=False,
                legendgroup=f'Cluster {cluster}'
            ),
            row=1, col=2
        )
        
    # 3. Boxplot: Income distribution per cluster
    for cluster in sorted(df_dash['Cluster'].unique()):
        sub_df = df_dash[df_dash['Cluster'] == cluster]
        fig.add_trace(
            go.Box(
                y=sub_df['Annual Income (k$)'],
                name=f'Cluster {cluster}',
                marker_color=color_map[cluster],
                boxmean=True,
                showlegend=False,
                legendgroup=f'Cluster {cluster}'
            ),
            row=2, col=1
        )
        
    # 4. Pie Chart: Segment shares
    counts = df_dash['Cluster'].value_counts().sort_index()
    fig.add_trace(
        go.Pie(
            labels=[f'Cluster {c}' for c in counts.index],
            values=counts.values,
            marker=dict(colors=[color_map[c] for c in counts.index], line=dict(color='white', width=1.5)),
            hole=0.4,
            hoverinfo='label+percent+value',
            textinfo='percent'
        ),
        row=2, col=2
    )
    
    # Update layout details for elegant dashboard style
    fig.update_layout(
        title_text='<b>Customer Segmentation Dashboard (K-Means)</b>',
        title_font=dict(size=22, family='Arial', color='#2C3E50'),
        title_x=0.5,
        width=1200,
        height=800,
        plot_bgcolor='#F8F9FA',
        paper_bgcolor='white',
        hovermode='closest',
        legend=dict(
            title_text='<b>Segments</b>',
            orientation='h',
            yanchor='bottom',
            y=-0.1,
            xanchor='center',
            x=0.5
        )
    )
    
    # Axis titles
    fig.update_xaxes(title_text='Annual Income (k$)', row=1, col=1)
    fig.update_yaxes(title_text='Spending Score (1-100)', row=1, col=1)
    
    fig.update_xaxes(title_text='Clusters', row=1, col=2)
    fig.update_yaxes(title_text='Age', row=1, col=2)
    
    fig.update_xaxes(title_text='Clusters', row=2, col=1)
    fig.update_yaxes(title_text='Annual Income (k$)', row=2, col=1)
    
    # Save dashboard as interactive HTML
    html_output = os.path.join(output_dir, 'dashboard.html')
    fig.write_html(html_output, include_plotlyjs='cdn')
    logger.info(f"Interactive dashboard HTML saved to {html_output}")
    
    # Save dashboard as static image (Requirements say dashboard.png)
    png_output = os.path.join(output_dir, 'dashboard.png')
    logger.info("Generating static dashboard image via Matplotlib layout...")
    create_matplotlib_dashboard_fallback(df_dash, color_map, png_output)


def create_matplotlib_dashboard_fallback(df_dash: pd.DataFrame, color_map: dict, filepath: str) -> None:
    """
    Generates a high-quality matplotlib dashboard layout in case Kaleido static export fails.
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Customer Segmentation Dashboard (K-Means)', fontsize=18, fontweight='bold', color='#2C3E50')
    
    # Plot 1: Income vs Spending Scatter
    for cluster in sorted(df_dash['Cluster'].unique()):
        sub_df = df_dash[df_dash['Cluster'] == cluster]
        axes[0, 0].scatter(sub_df['Annual Income (k$)'], 
                           sub_df['Spending Score (1-100)'], 
                           s=50, 
                           color=color_map[cluster], 
                           label=f'Cluster {cluster}', 
                           alpha=0.8,
                           edgecolors='black',
                           linewidths=0.5)
    axes[0, 0].set_title('Annual Income vs Spending Score', fontweight='bold')
    axes[0, 0].set_xlabel('Annual Income (k$)')
    axes[0, 0].set_ylabel('Spending Score (1-100)')
    axes[0, 0].legend()
    axes[0, 0].grid(True, linestyle='--', alpha=0.5)
    
    # Plot 2: Boxplot Age
    sns.boxplot(x='Cluster', y='Age', data=df_dash, ax=axes[0, 1], palette=color_map)
    axes[0, 1].set_title('Age Distribution by Cluster', fontweight='bold')
    axes[0, 1].set_xlabel('Cluster')
    axes[0, 1].set_ylabel('Age')
    
    # Plot 3: Boxplot Income
    sns.boxplot(x='Cluster', y='Annual Income (k$)', data=df_dash, ax=axes[1, 0], palette=color_map)
    axes[1, 0].set_title('Annual Income Distribution by Cluster', fontweight='bold')
    axes[1, 0].set_xlabel('Cluster')
    axes[1, 0].set_ylabel('Annual Income (k$)')
    
    # Plot 4: Pie Chart
    counts = df_dash['Cluster'].value_counts().sort_index()
    axes[1, 1].pie(counts.values, 
                   labels=[f'Cluster {c}' for c in counts.index], 
                   colors=[color_map[c] for c in counts.index], 
                   autopct='%1.1f%%', 
                   startangle=140,
                   wedgeprops=dict(width=0.4, edgecolor='white', linewidth=1.5))
    axes[1, 1].set_title('Customer Segment Distribution', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(filepath, dpi=300)
    plt.close(fig)
    logger.info(f"Matplotlib fallback dashboard saved to {filepath}")
