```python
import argparse
import json
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Lasso
import xgboost as xgb
import shap
from sklearn.metrics import mean_squared_error, r2_score

def load_data(data_path):
    print(f"Loading data from {data_path}")
    df = pd.read_csv(data_path)
    
    target_columns = ['hg/ha_yield', 'yield']
    target_col = next((col for col in target_cols if col in df.columns), None)
    
    if not target_col:
        raise ValueError("Target column not found in the dataset. Expected 'hg/ha_yield' or 'yield'")
    
    y = df[target_col]
    
    exclude_cols = [target_col, 'id', 'Year', 'year', 'location_id']
    X = df.drop([col for col in exclude_cols if col in df.columns], axis=1)
    
    print(f"Dataset loaded with {X.shape[1]} features and {X.shape[0]} samples")
    return X, y

def compression_analysis(X, threshold=0.85, output_dir=None):
    print(f"Performing compression analysis with threshold {threshold}...")
    
    corr_matrix = X.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [column for column in upper.columns if any(upper[column] > threshold)]
    
    print(f"Identified {len(to_drop)} features to drop based on correlation > {threshold}")
    
    high_corr_pairs = []
    for column in to_drop:
        correlated_features = upper[column][upper[column] > threshold].index.tolist()
        for feat in correlated_features:
            high_corr_pairs.append({
                'dropped_feature': column,
                'kept_feature': feat,
                'correlation': corr_matrix.loc[column, feat]
            })
    
    high_corr_df = pd.DataFrame(high_corr_pairs)
    
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        
        # Save correlation matrix heatmap
        plt.figure(figsize=(12, 10))
        sns.heatmap(corr_matrix, annot=False, cmap='coolwarm', vmin=-1, vmax=1)
        plt.title('Feature Correlation Matrix')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'correlation_matrix.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # Save high correlation pairs
        high_corr_df.to_csv(os.path.join(output_dir, 'high_correlation_pairs.csv'), index=False)
    
    X_compressed = X.drop(columns=to_drop)
    print(f"Features reduced from {X.shape[1]} to {X_compressed.shape[1]} after compression analysis")
    
    return X_compressed, to_drop, high_corr_df

def lasso_feature_selection(X, y, alpha=0.01, output_dir=None):
    print(f"Performing LASSO feature selection with alpha={alpha}...")
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    lasso = Lasso(alpha=alpha, max_iter=10000, random_state=42)
    lasso.fit(X_scaled, y)
    
    feature_importance = pd.DataFrame({
        'Feature': X.columns,
        'Coefficient': np.abs(lasso.coef_)
    }).sort_values('Coefficient', ascending=False)
    
    # Select features with non-zero coefficients
    selected_features = feature_importance[feature_importance['Coefficient'] > 0]['Feature'].tolist()
    
    print(f"LASSO selected {len(selected_features)} features with non-zero coefficients")
    
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        
        # Save feature importance
        feature_importance.to_csv(os.path.join(output_dir, 'lasso_feature_importance.csv'), index=False)
        
        # Plot feature importance
        plt.figure(figsize=(12, 8))
        plt.barh(feature_importance['Feature'][:20], feature_importance['Coefficient'][:20])
        plt.xlabel('Coefficient Magnitude')
        plt.ylabel('Feature')
        plt.title('Top 20 Features by LASSO Coefficient Magnitude')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'lasso_feature_importance.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    X_selected = X[selected_features]
    return X_selected, selected_features, lasso

def shap_analysis(X, y, selected_features, output_dir=None):
    print("Performing SHAP analysis for final feature ranking...")
    
    # Use a sample of the data for SHAP analysis if the dataset is large
    if len(X) > 1000:
        X_sample, _, y_sample, _ = train_test_split(X, y, train_size=1000, random_state=42)
    else:
        X_sample, y_sample = X, y
    
    # Train a simple XGBoost model for SHAP analysis
    model = xgb.XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42)
    model.fit(X_sample, y_sample)
    
    # Calculate SHAP values
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample)
    
    # Calculate feature importance based on SHAP values
    feature_importance = pd.DataFrame({
        'Feature': X_sample.columns,
        'SHAP_importance': np.abs(shap_values).mean(0)
    }).sort_values('SHAP_importance', ascending=False)
    
    print(f"SHAP analysis complete. Top 5 features: {', '.join(feature_importance['Feature'][:5].tolist())}")
    
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        
        # Save SHAP feature importance
        feature_importance.to_csv(os.path.join(output_dir, 'shap_feature_importance.csv'), index=False)
        
        # SHAP summary plot
        plt.figure(figsize=(12, 8))
        shap.summary_plot(shap_values, X_sample, show=False)
        plt.title('SHAP Feature Importance')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'shap_summary.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # SHAP bar plot
        plt.figure(figsize=(12, 6))
        shap.summary_plot(shap_values, X_sample, plot_type="bar", show=False)
        plt.title('SHAP Feature Importance (Bar)')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'shap_bar.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    return feature_importance

def evaluate_feature_selection(X, y, selected_features):
    print("Evaluating feature selection performance...")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train model with all features
    full_model = xgb.XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42)
    full_model.fit(X_train, y_train)
    full_pred = full_model.predict(X_test)
    full_rmse = np.sqrt(mean_squared_error(y_test, full_pred))
    full_r2 = r2_score(y_test, full_pred)
    
    # Train model with selected features
    X_train_selected = X_train[selected_features]
    X_test_selected = X_test[selected_features]
    selected_model = xgb.XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42)
    selected_model.fit(X_train_selected, y_train)
    selected_pred = selected_model.predict(X_test_selected)
    selected_rmse = np.sqrt(mean_squared_error(y_test, selected_pred))
    selected_r2 = r2_score(y_test, selected_pred)
    
    # Calculate improvement
    rmse_improvement = (full_rmse - selected_rmse) / full_rmse * 100
    r2_improvement = (selected_r2 - full_r2) / full_r2 * 100 if full_r2 > 0 else 0
    
    print(f"Model with all features: RMSE = {full_rmse:.2f}, R² = {full_r2:.4f}")
    print(f"Model with selected features: RMSE = {selected_rmse:.2f}, R² = {selected_r2:.4f}")
    print(f"Improvement: RMSE reduced by {rmse_improvement:.2f}%, R² increased by {r2_improvement:.2f}%")
    
    return {
        'full_model': {
            'rmse': full_rmse,
            'r2': full_r2
        },
        'selected_model': {
            'rmse': selected_rmse,
            'r2': selected_r2
        },
        'improvement': {
            'rmse_percent': rmse_improvement,
            'r2_percent': r2_improvement
        }
    }

def hybrid_feature_selection(X, y, correlation_threshold=0.85, lasso_alpha=0.01, output_dir=None):
    print("Starting hybrid feature selection process...")
    
    # Step 1: Compression Analysis
    compression_dir = os.path.join(output_dir, '1_compression_analysis') if output_dir else None
    X_compressed, dropped_features, high_corr_df = compression_analysis(
        X, threshold=correlation_threshold, output_dir=compression_dir
    )
    
    # Step 2: LASSO Selection
    lasso_dir = os.path.join(output_dir, '2_lasso_selection') if output_dir else None
    X_lasso, lasso_features, lasso_model = lasso_feature_selection(
        X_compressed, y, alpha=lasso_alpha, output_dir=lasso_dir
    )
    
    # Step 3: SHAP Analysis
    shap_dir = os.path.join(output_dir, '3_shap_analysis') if output_dir else None
    feature_importance = shap_analysis(X_lasso, y, lasso_features, output_dir=shap_dir)
    
    # Evaluate the feature selection performance
    evaluation = evaluate_feature_selection(X, y, lasso_features)
    
    # Compile results
    results = {
        'dropped_features': dropped_features,
        'selected_features': lasso_features,
        'feature_importance': feature_importance.to_dict('records'),
        'performance_evaluation': evaluation
    }
    
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        with open(os.path.join(output_dir, 'feature_selection_results.json'), 'w') as f:
            json.dump(results, f, indent=4)
    
    print("Hybrid feature selection completed successfully.")
    return results

def main():
    parser = argparse.ArgumentParser(description='Hybrid Feature Selection for Agricultural Yield Prediction')
    parser.add_argument('--input', required=True, help='Path to input CSV data')
    parser.add_argument('--output', default='selected_features.json', help='Path to save selected features')
    parser.add_argument('--output_dir', default='feature_selection_results', help='Directory to save detailed results')
    parser.add_argument('--threshold', type=float, default=0.85, help='Correlation threshold for compression analysis')
    parser.add_argument('--alpha', type=float, default=0.01, help='Alpha parameter for LASSO selection')
    args = parser.parse_args()
    
    # Load data
    X, y = load_data(args.input)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Perform hybrid feature selection
    results = hybrid_feature_selection(
        X, y, 
        correlation_threshold=args.threshold, 
        lasso_alpha=args.alpha,
        output_dir=args.output_dir
    )
    
    # Save selected features to specified output file
    with open(args.output, 'w') as f:
        json.dump(results['selected_features'], f, indent=4)
    
    print(f"Selected features saved to {args.output}")
    print(f"Detailed results saved to {args.output_dir}")

if __name__ == "__main__":
    main()
