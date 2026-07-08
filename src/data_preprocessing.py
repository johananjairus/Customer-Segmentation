import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
import logging
from typing import Tuple, Dict, Any

logger = logging.getLogger("CustomerSegmentation.preprocessing")

def load_data(filepath: str) -> pd.DataFrame:
    """
    Loads dataset from a CSV file.
    
    Parameters:
    -----------
    filepath : str
        Path to the CSV file.
        
    Returns:
    --------
    pd.DataFrame
        Loaded dataframe.
    """
    logger.info(f"Loading data from {filepath}...")
    try:
        df = pd.read_csv(filepath)
        logger.info(f"Successfully loaded data. Shape: {df.shape}")
        return df
    except FileNotFoundError:
        logger.error(f"File not found at {filepath}")
        raise
    except Exception as e:
        logger.error(f"Error loading file {filepath}: {e}")
        raise


def analyze_data_quality(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Checks missing values and duplicate rows.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe to inspect.
        
    Returns:
    --------
    Dict[str, Any]
        Dictionary containing counts of missing values and duplicates.
    """
    logger.info("Analyzing data quality...")
    
    # Check for missing values
    missing_values = df.isnull().sum().to_dict()
    total_missing = sum(missing_values.values())
    logger.info(f"Missing values check: {missing_values}")
    
    # Check for duplicates
    # Excluding CustomerID since it should be unique, but checking duplicates overall is standard
    duplicate_count = df.duplicated().sum()
    logger.info(f"Number of duplicate rows found: {duplicate_count}")
    
    # Output basic dataset information
    logger.info("Dataset Columns and Data Types:")
    for col, dtype in zip(df.columns, df.dtypes):
        logger.info(f" - {col}: {dtype}")
        
    return {
        "missing_values": missing_values,
        "total_missing": total_missing,
        "duplicate_count": duplicate_count
    }


def preprocess_data(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, StandardScaler, LabelEncoder]:
    """
    Preprocesses dataset by encoding the categorical columns and scaling numeric columns.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe.
        
    Returns:
    --------
    Tuple[pd.DataFrame, pd.DataFrame, StandardScaler, LabelEncoder]
        - preprocessed_df: Dataframe with encoded categorical variables and original numerical columns.
        - scaled_df: Scaled feature dataframe ready for clustering.
        - scaler: Fitted StandardScaler object.
        - gender_encoder: Fitted LabelEncoder object for Gender.
    """
    logger.info("Preprocessing dataset...")
    df_clean = df.copy()
    
    # Check if duplicate rows exist and drop them
    duplicate_rows = df_clean.duplicated().sum()
    if duplicate_rows > 0:
        df_clean = df_clean.drop_duplicates().reset_index(drop=True)
        logger.info("Dropped duplicate rows.")
        
    # Check and handle missing values if any (though synthetic data shouldn't have any)
    if df_clean.isnull().any().any():
        logger.warning("Missing values found during preprocessing. Filling with median/mode values.")
        for col in df_clean.columns:
            if df_clean[col].isnull().any():
                if df_clean[col].dtype == 'object':
                    df_clean[col] = df_clean[col].fillna(df_clean[col].mode()[0])
                else:
                    df_clean[col] = df_clean[col].fillna(df_clean[col].median())
                    
    # Categorical Variable Encoding (Gender)
    gender_encoder = LabelEncoder()
    df_clean['Gender_Encoded'] = gender_encoder.fit_transform(df_clean['Gender'])
    logger.info("Encoded 'Gender' to 'Gender_Encoded' using LabelEncoder (Female=0, Male=1).")
    
    # Display descriptive statistics of numerical variables
    numeric_cols = ['Age', 'Annual Income (k$)', 'Spending Score (1-100)']
    desc_stats = df_clean[numeric_cols].describe()
    logger.info("\nDescriptive Statistics for Numerical Features:\n" + desc_stats.to_string())
    
    # Feature Scaling using StandardScaler
    scaler = StandardScaler()
    
    # Features selected for clustering: Age, Annual Income, Spending Score
    features_to_scale = ['Age', 'Annual Income (k$)', 'Spending Score (1-100)']
    
    scaled_features = scaler.fit_transform(df_clean[features_to_scale])
    
    # Create scaled dataframe
    scaled_df = pd.DataFrame(scaled_features, columns=features_to_scale)
    
    # Log scaling mean and std dev details
    logger.info("Feature scaling completed.")
    for i, col in enumerate(features_to_scale):
        logger.info(f" - {col}: Mean after scaling = {scaled_features[:, i].mean():.4f}, Std Dev = {scaled_features[:, i].std():.4f}")
        
    return df_clean, scaled_df, scaler, gender_encoder
