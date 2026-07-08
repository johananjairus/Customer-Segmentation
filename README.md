# Customer Segmentation using Machine Learning

A complete end-to-end Python project for Customer Segmentation utilizing clustering algorithms to identify buyer personas and optimize target marketing campaigns.

This repository implements a production-grade machine learning pipeline to analyze mall customer purchasing behavior and demographics, segment them, evaluate model performance, and serve an interactive dashboard.

---

## 📌 Project Overview
In modern retail business, a "one-size-fits-all" marketing strategy is inefficient. Customer Segmentation groups customers with similar purchasing characteristics (demographics, income, and spending scores) into distinct cohorts. This enables businesses to:
- Maximize marketing return on investment (ROI).
- Design tailored product recommendation models.
- Run highly targeted promotional campaigns (e.g. VIP memberships, discount packages).

This project cleans demographic and purchasing data, runs exploratory data analysis, trains multiple clustering models (**K-Means**, **Hierarchical Clustering (Ward's Linkage)**, and **DBSCAN**), reduces feature dimensionality using **PCA**, dynamically profiles customer cohorts, and generates an interactive web-based dashboard.

---

## 📊 Dataset Description
The analysis uses the **Mall Customers Dataset** (`data/Mall_Customers.csv`), which contains 200 customer profiles:
*   `CustomerID`: Unique identifier for each customer.
*   `Gender`: Gender of the customer (`Male` / `Female`).
*   `Age`: Age of the customer (18 to 70 years).
*   `Annual Income (k$)`: Annual income of the customer in thousands of dollars ($15k to $140k).
*   `Spending Score (1-100)`: Score assigned by the mall based on customer behavior and spending history.

---

## 🗂️ Project Structure

```text
Customer-Segmentation/
│
├── data/
│   └── Mall_Customers.csv           # Customer profiles dataset
│
├── notebooks/
│   └── customer_segmentation.ipynb  # Step-by-step EDA & modeling walk-through
│
├── src/
│   ├── data_preprocessing.py        # Loading, cleaning, encoding, and scaling
│   ├── clustering.py                # K-Means, DBSCAN, Agglomerative models & metrics
│   ├── visualization.py             # Custom Matplotlib/Seaborn & Plotly dashboards
│   └── utils.py                     # Logging configuration & helper functions
│
├── output/
│   ├── distributions.png            # Feature histograms & KDEs
│   ├── correlation_heatmap.png      # Feature correlation matrix
│   ├── gender_boxplots.png          # Boxplots of numerical features by gender
│   ├── pairplot.png                 # Pairwise relationship grid
│   ├── elbow_method.png             # WCSS Elbow & Silhouette curves
│   ├── cluster_plot.png             # K-Means clusters and centroids scatter
│   ├── pca_cluster_plot.png         # Cluster visualization in PCA 2D space
│   ├── dendrogram.png               # Hierarchical clustering dendrogram
│   ├── dashboard.html               # Plotly interactive dashboard
│   ├── dashboard.png                # Static dashboard.html layout export
│   ├── customer_segments_summary.csv# Profiling results by segment
│   ├── clustering_algorithm_comparison.csv # Performance comparison table
│   └── segmentation.log             # Project run logging output
│
├── requirements.txt                 # Environment dependencies
├── README.md                        # Documentation
└── main.py                          # Execution script
```

---

## 🚀 Installation & Setup

1.  **Clone the Repository**:
    ```bash
    git clone https://github.com/your-username/Customer-Segmentation.git
    cd Customer-Segmentation
    ```

2.  **Create and Activate a Virtual Environment**:
    Using Python's standard `venv` module:
    ```bash
    python -m venv .venv
    # On Windows:
    .venv\Scripts\activate
    # On macOS/Linux:
    source .venv/bin/activate
    ```
    *(Alternatively, using `uv` for 10x faster installations)*:
    ```bash
    uv venv
    .venv\Scripts\activate
    ```

3.  **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

---

## 💻 Usage

To run the entire preprocessing, analysis, and clustering pipeline, execute the `main.py` entry point:
```bash
python main.py
```

This script will:
1.  Initialize the project directory structure.
2.  Clean, encode, and scale the raw customer profiles.
3.  Perform EDA and save visualizations to the `output/` directory.
4.  Run elbow and silhouette analysis to find the optimal number of clusters.
5.  Execute **K-Means**, **Hierarchical Clustering**, and **DBSCAN**.
6.  Generate a PCA-reduced 2D visualization of the customer segments.
7.  Generate the interactive Plotly dashboard (`output/dashboard.html`) and static charts.
8.  Produce algorithm comparison and customer segments profiles CSV reports.

To run the analysis interactively, start Jupyter and open the notebook:
```bash
jupyter notebook notebooks/customer_segmentation.ipynb
```

---

## 📈 Key Results & Insights

### Model Performance Comparison
The algorithms were evaluated using three standard metrics:
*   **Silhouette Score** (range [-1, 1], higher is better): Measures how similar an object is to its own cluster compared to other clusters.
*   **Davies-Bouldin Index** (minimum 0, lower is better): Measures the average similarity between each cluster and its most similar one.
*   **Calinski-Harabasz Score** (higher is better): Computes the ratio of the sum of between-clusters dispersion and of within-cluster dispersion.

| Algorithm | Clusters Found | Silhouette Score | Davies-Bouldin Index | Calinski-Harabasz Score |
| :--- | :---: | :---: | :---: | :---: |
| **K-Means** | 5 | **0.4775** | **0.8285** | **173.9161** |
| **Hierarchical (Ward)** | 5 | 0.4650 | 0.8583 | 163.0239 |
| **DBSCAN (eps=0.6, min=4)** | 5 | 0.5204* | 0.7660* | 170.0446* |

*\*Note: DBSCAN scores exclude 10 outlier/noise points (`-1` label). While DBSCAN shows high density metrics, K-Means is preferred for business operations as it groups 100% of the customer base without leaving unclassified outliers.*

---

### K-Means Customer Segment Profiling

Our pipeline identified 5 distinct customer groups based on their spending and income profiles:

#### 1. Cluster 0: Standard / Moderate Customers (28.5% of total)
*   **Profile**: Average Age = 35.4 | Average Income = $51.4k | Average Spending Score = 46.6
*   **Persona**: Medium income earners with moderate shopping habits. They represent the largest customer block.
*   **Strategy**: Engage with mainstream product newsletters, reward points programs, and seasonal store-wide discount sales.

#### 2. Cluster 1: Stars / Target Group (19.5% of total)
*   **Profile**: Average Age = 31.3 | Average Income = $87.5k | Average Spending Score = 81.2
*   **Persona**: Young, high earners who spend heavily. This is the most profitable segment for the mall.
*   **Strategy**: Target with premium product alerts, exclusive VIP event invites, high-end loyalty tiers, and personalized concierge services.

#### 3. Cluster 2: Spendthrifts / Careless Customers (12.5% of total)
*   **Profile**: Average Age = 25.2 | Average Income = $24.0k | Average Spending Score = 76.0
*   **Persona**: Young customers who earn less but shop excessively. They are highly responsive to trends.
*   **Strategy**: Engage via social media ads showing budget-friendly trendy apparel, flash sales, discount coupons, and Buy Now Pay Later (BNPL) options.

#### 4. Cluster 3: Frugal / Conservative Customers (22.5% of total)
*   **Profile**: Average Age = 53.4 | Average Income = $41.2k | Average Spending Score = 38.1
*   **Persona**: Older, lower-income shoppers who are highly budget-conscious.
*   **Strategy**: Target with utility-focused campaigns, multi-buy discount options (e.g. buy 1 get 1 free), and cost-effective value pack promotions.

#### 5. Cluster 4: Careful / Sensible Customers (17.0% of total)
*   **Profile**: Average Age = 42.1 | Average Income = $88.4k | Average Spending Score = 18.0
*   **Persona**: High earners who are extremely conservative spenders at the mall.
*   **Strategy**: Focus marketing on product quality, durability, premium materials, and cash-back card rewards to incentivize spending.

---

## 🖥️ Interactive Plotly Dashboard
The project generates an interactive dashboard saved to `output/dashboard.html`. It contains:
- **Income vs Spending score scatter plot** (colored by cluster).
- **Age distribution boxplot** across clusters.
- **Annual Income distribution boxplot** across clusters.
- **Pie chart** representing customer segment proportions.

Double click `output/dashboard.html` to open it in any web browser. A static representation is also saved as `output/dashboard.png`.

---

## 🔮 Future Improvements
- **Incorporate Temporal Data**: Add timestamps to track how customer cluster memberships shift over seasons (e.g., holiday spenders).
- **Add Recency, Frequency, Monetary (RFM) Features**: Transition from transactional averages to direct purchase history metrics.
- **Semi-supervised Refinement**: Build an active feedback loop where marketing team ratings adjust cluster thresholds.
- **Model Deployment**: Build an API (e.g. FastAPI) that accepts customer demographics and returns their segment and recommended campaign in real-time.
