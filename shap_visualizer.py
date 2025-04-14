```python
import argparse
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import shap
import tensorflow as tf
from sklearn.preprocessing import StandardScaler

def load_model_and_data(model_path, data_path):
    # Load data
    print(f"Loading data from {data_path}")
    df = pd.read_csv(data_path)
    
    # Identify target column
    target_columns = ['hg/ha_yield', 'yield']
    target_col = next((col for col in target_columns if col in df.columns), None)
    
    if target_col:
        print(f"Target column '{target_col}' found in data")
        y = df[target_col]
        # Remove target and non-feature columns
        exclude_cols = [target_col, 'id', 'Year', 'year', 'location_id']
        X = df.drop([col for col in exclude_cols if col in df.columns], axis=1)
    else:
        print("No target column found in data")
        # Assume all columns are features except common non-feature columns
        exclude_cols = ['id', 'Year', 'year', 'location_id']
        X = df.drop([col for col in exclude_cols if col in df.columns], axis=1)
        y = None
    
    # Load model and identify its type
    print(f"Loading model from {model_path}")
    
    model_type = None
    scaler = None
    
    if model_path.endswith('.joblib'):
        model = joblib.load(model_path)
        
        if 'XGBRegressor' in str(type(model)):
            model_type = 'xgb'
        elif 'RandomForestRegressor' in str(type(model)):
            model_type = 'rf'
        elif 'SVR' in str(type(model)):
            model_type = 'svm'
            # Look for corresponding scaler
            scaler_path = model_path.replace('svm_model.joblib', 'svm_scaler.joblib')
            if os.path.exists(scaler_path):
                scaler = joblib.load(scaler_path)
        elif 'KNeighborsRegressor' in str(type(model)):
            model_type = 'knn'
            # Look for corresponding scaler
            scaler_path = model_path.replace('knn_model.joblib', 'knn_scaler.joblib')
            if os.path.exists(scaler_path):
                scaler = joblib.load(scaler_path)
        elif 'LinearRegression' in str(type(model)):
            model_type = 'lr'
    else:
        # Assume it's a TensorFlow model if it's a directory
        try:
            model = tf.keras.models.load_model(model_path)
            model_type = 'ann'
            
            # Look for scaler in the same directory
            model_dir = os.path.dirname(model_path)
            scaler_path = os.path.join(model_dir, 'ann_scaler.joblib')
            if os.path.exists(scaler_path):
                scaler = joblib.load(scaler_path)
        except:
            raise ValueError(f"Could not load model from {model_path}")
    
    print(f"Loaded {model_type} model successfully")
    
    if scaler:
        print("Found and loaded feature scaler")
    
    return model, X, y, model_type, scaler

def generate_shap_values(model, X, model_type, scaler=None, sample_size=100):
    print("Generating SHAP values...")
    
    # Sample data for explanation (limit to sample_size rows for efficiency)
    X_sample = X.sample(min(sample_size, len(X)), random_state=42)
    
    # Apply scaler if provided (for neural network, SVM, KNN)
    if scaler is not None:
        X_sample_scaled = scaler.transform(X_sample)
    else:
        X_sample_scaled = X_sample
    
    # Create SHAP explainer based on model type
    if model_type in ['rf', 'xgb']:
        # Tree-based models
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_sample)
    else:
        # For non-tree models use KernelExplainer
        # Use a smaller sample for KernelExplainer as it's computationally expensive
        X_small_sample = X_sample.sample(min(50, len(X_sample)), random_state=42)
        if scaler is not None:
            X_small_sample_scaled = scaler.transform(X_small_sample)
        else:
            X_small_sample_scaled = X_small_sample
        
        # Define prediction function based on model type
        if model_type == 'ann':
            def predict_fn(x):
                return model.predict(x).flatten()
        else:
            def predict_fn(x):
                return model.predict(x)
        
        explainer = shap.KernelExplainer(predict_fn, X_small_sample_scaled)
        shap_values = explainer.shap_values(X_small_sample_scaled)
        X_sample = X_small_sample  # Use the same smaller sample for visualization
    
    return shap_values, X_sample, explainer

def create_summary_plot(shap_values, X_sample, output_dir, model_type):
    plt.figure(figsize=(12, 8))
    shap.summary_plot(shap_values, X_sample, show=False)
    plt.title(f'SHAP Summary Plot - {model_type.upper()}')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'shap_summary.png'), dpi=300, bbox_inches='tight')
    plt.close()

def create_bar_plot(shap_values, X_sample, output_dir, model_type):
    plt.figure(figsize=(12, 6))
    shap.summary_plot(shap_values, X_sample, plot_type="bar", show=False)
    plt.title(f'SHAP Feature Importance - {model_type.upper()}')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'shap_feature_importance.png'), dpi=300, bbox_inches='tight')
    plt.close()

def create_dependence_plots(shap_values, X_sample, output_dir, feature_importance, top_n=5):
    # Create dependence plots for top N features
    for i, feature in enumerate(feature_importance['Feature'][:top_n]):
        plt.figure(figsize=(10, 6))
        shap.dependence_plot(
            feature, 
            shap_values, 
            X_sample,
            show=False
        )
        plt.title(f'SHAP Dependence Plot - {feature}')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'shap_dependence_{i+1}_{feature}.png'), dpi=300, bbox_inches='tight')
        plt.close()

def create_force_plots(shap_values, X_sample, explainer, output_dir, n_samples=3):
    # Create force plots for a few samples
    for i in range(min(n_samples, len(X_sample))):
        plt.figure(figsize=(20, 3))
        shap.force_plot(
            explainer.expected_value, 
            shap_values[i:i+1], 
            X_sample.iloc[i:i+1],
            matplotlib=True,
            show=False
        )
        plt.title(f'SHAP Force Plot - Sample {i+1}')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'shap_force_plot_sample_{i+1}.png'), dpi=300, bbox_inches='tight')
        plt.close()

def create_waterfall_plots(shap_values, X_sample, explainer, output_dir, n_samples=3):
    # Create waterfall plots for a few samples
    for i in range(min(n_samples, len(X_sample))):
        plt.figure(figsize=(10, 6))
        shap.waterfall_plot(
            shap.Explanation(
                values=shap_values[i], 
                base_values=explainer.expected_value, 
                data=X_sample.iloc[i].values, 
                feature_names=X_sample.columns.tolist()
            ),
            show=False
        )
        plt.title(f'SHAP Waterfall Plot - Sample {i+1}')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'shap_waterfall_plot_sample_{i+1}.png'), dpi=300, bbox_inches='tight')
        plt.close()

def save_feature_importance(shap_values, X_sample, output_dir):
    # Calculate and save feature importance
    feature_importance = pd.DataFrame({
        'Feature': X_sample.columns,
        'SHAP_importance': np.abs(shap_values).mean(0)
    }).sort_values('SHAP_importance', ascending=False)
    
    feature_importance.to_csv(os.path.join(output_dir, 'feature_importance.csv'), index=False)
    
    return feature_importance

def main():
    parser = argparse.ArgumentParser(description='Generate SHAP visualizations for model interpretation')
    parser.add_argument('--model', required=True, help='Path to trained model')
    parser.add_argument('--data', required=True, help='Path to data CSV')
    parser.add_argument('--output', default='visualization_output', help='Output directory for visualizations')
    parser.add_argument('--sample_size', type=int, default=100, help='Number of samples to use for SHAP analysis')
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output, exist_ok=True)
    
    # Load model and data
    model, X, y, model_type, scaler = load_model_and_data(args.model, args.data)
    
    # Generate SHAP values
    shap_values, X_sample, explainer = generate_shap_values(
        model, X, model_type, scaler, sample_size=args.sample_size
    )
    
    # Calculate feature importance
    feature_importance = save_feature_importance(shap_values, X_sample, args.output)
    
    print("Generating visualizations...")
    
    # Create summary plot
    create_summary_plot(shap_values, X_sample, args.output, model_type)
    
    # Create bar plot
    create_bar_plot(shap_values, X_sample, args.output, model_type)
    
    # Create dependence plots
    create_dependence_plots(shap_values, X_sample, args.output, feature_importance)
    
    # Create force plots
    try:
        create_force_plots(shap_values, X_sample, explainer, args.output)
    except Exception as e:
        print(f"Warning: Could not create force plots: {e}")
    
    # Create waterfall plots
    try:
        create_waterfall_plots(shap_values, X_sample, explainer, args.output)
    except Exception as e:
        print(f"Warning: Could not create waterfall plots: {e}")
    
    print(f"Visualizations saved to {args.output}")
    
    # Print top features
    print("\nTop 10 features by SHAP importance:")
    for i, row in feature_importance.head(10).iterrows():
        print(f"{i+1}. {row['Feature']}: {row['SHAP_importance']:.4f}")

if __name__ == "__main__":
    main()
