import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import mutual_info_regression
import os
import json
import logging

logger = logging.getLogger(__name__)

def run_feature_selection(df, config):
    logger.info("Starting feature selection stage.")
    results_dir = config['paths']['results']
    target_col = config['data']['target_col']
    
    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    # 1. Correlation Matrix
    logger.info("Computing correlation matrix...")
    corr_matrix = df.corr()
    plt.figure(figsize=(15, 12))
    sns.heatmap(corr_matrix, annot=False, cmap='coolwarm')
    plt.title('Feature Correlation Heatmap')
    plt.savefig(os.path.join(results_dir, 'correlation_heatmap.png'))
    plt.close()
    
    # 2. Mutual Information
    logger.info("Computing Mutual Information regression scores...")
    # Sampling for speed if dataset is very large
    sample_size = min(20000, len(df))
    X_sample = X.sample(sample_size, random_state=42)
    y_sample = y.sample(sample_size, random_state=42)
    
    mi_scores = mutual_info_regression(X_sample, y_sample)
    mi_series = pd.Series(mi_scores, index=X.columns).sort_values(ascending=False)
    
    # 3. Random Forest Feature Importance
    logger.info("Training lightweight Random Forest for importance scores...")
    rf = RandomForestRegressor(n_estimators=50, max_depth=10, n_jobs=-1, random_state=42)
    rf.fit(X_sample, y_sample)
    rf_importance = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)
    
    # Selection logic: Top N features or union of top from both MI and RF
    # Preserving seasonal lags and key rolling stats regardless of scores if they are in top set
    top_mi = mi_series.head(30).index.tolist()
    top_rf = rf_importance.head(30).index.tolist()
    
    selected_features = list(set(top_mi) | set(top_rf))
    
    # Ensure mandatory seasonal lags are preserved if they are reasonably important
    mandatory_lags = ['lag_96', 'lag_672', 'hour_sin', 'hour_cos']
    for lag in mandatory_lags:
        if lag not in selected_features and lag in X.columns:
            selected_features.append(lag)
            
    logger.info(f"Feature selection complete. Selected {len(selected_features)} features.")
    
    # Save selected feature list
    output_path = os.path.join(config['paths']['feature_data'], 'selected_features.json')
    with open(output_path, 'w') as f:
        json.dump(selected_features, f, indent=4)
        
    return selected_features
