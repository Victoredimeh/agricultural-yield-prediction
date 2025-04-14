import argparse
import json
import numpy as np
import pandas as pd
import joblib
import os
import matplotlib.pyplot as plt
import shap
import tensorflow as tf
from sklearn.preprocessing import StandardScaler

def load_model(model_path):
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
            scaler_path = model_path.replace('svm_model.joblib', 'svm_scaler.joblib')
            if os.path.exists(scaler_path):
                scaler = joblib.load(scaler_path)
        elif 'KNeighborsRegressor' in str(type(model)):
            model_type = 'knn'
            scaler_path = model_path.replace('knn_model.joblib', 'knn_scaler.joblib')
            if os.path.exists(scaler_path):
                scaler = joblib.load(scaler_path)
        elif 'LinearRegression' in str(type(model)):
            model_type = 'lr'
    else:
        try:
            model = tf.keras.models.load_model(model_path)
            model_type = 'ann'
            
            model_dir = os.path.dirname(model_path)
            scaler_path = os.path.join(model_dir, 'ann_scaler.joblib')
            if os.path.exists(scaler_path):
                scaler = joblib.load(scaler_path)
        except:
            raise ValueError(f"Could not load model from {model_path}")
    
    return model, scaler, model_type

def make_predictions(model, X, scaler=None, model_type=None):
    if scaler is not None:
        X_scaled = scaler.transform(X)
    else:
        X_scaled = X
    
    if model_type == 'ann':
        predictions = model.predict(X_scaled).flatten()
    else:
        predictions = model.predict(X_scaled)
    
    return predictions

def explain_predictions(model, X, model_type):
    X_sample = X.sample(min(100, len(X)), random_state=42)
    
    if model_type in ['rf', 'xgb']:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_sample)
    else:
        X_small_sample = X_sample.sample(min(50, len(X_sample)), random_state=42)
        
        if model_type == 'ann':
            def predict_fn(x):
                return model.predict(x).flatten()
        else:
            def predict_fn(x):
                return model.predict(x)
        
        explainer = shap.KernelExplainer(predict_fn, X_small_sample)
        shap_values = explainer.shap_values(X_small_sample)
    
    return shap_values, explainer

def visualize_explanations(shap_values, X, output_dir=None):
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    # 1. Summary plot
    plt.figure(figsize=(12, 8))
    shap.summary_plot(shap_values, X, show=False)
    plt.tight_layout()
    if output_dir:
        plt.savefig(os.path.join(output_dir, 'shap_summary.png'), dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()
    
    # 2. Feature importance plot
    plt.figure(figsize=(12, 6))
    shap.summary_plot(shap_values, X, plot_type="bar", show=False)
    plt.tight_layout()
    if output_dir:
        plt.savefig(os.path.join(output_dir, 'shap_feature_importance.png'), dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()
    
    # 3. Dependence plots for top 3 features
    feature_importance = np.abs(shap_values).mean(0)
    top_features_idx = np.argsort(-feature_importance)[:3]
    top_features = X.columns[top_features_idx]
    
    for feature in top_features:
        plt.figure(figsize=(10, 6))
        shap.dependence_plot(feature, shap_values, X, show=False)
        plt.tight_layout()
        if output_dir:
            plt.savefig(os.path.join(output_dir, f'shap_dependence_{feature}.png'), dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.show()

def main():
    parser = argparse.ArgumentParser(description='Make predictions and explain results')
    parser.add_argument('--input', required=True, help='Path to input CSV data')
    parser.add_argument('--model', required=True, help='Path to trained model')
    parser.add_argument('--output', default='predictions.csv', help='Path to save predictions')
    parser.add_argument('--explain', action='store_true', help='Generate SHAP explanations')
    parser.add_argument('--viz_dir', default='shap_visualizations', help='Directory to save SHAP visualizations')
    args = parser.parse_args()
    
    print(f"Loading data from {args.input}")
    df = pd.read_csv(args.input)
    
    target_columns = ['hg/ha_yield', 'yield']
    target_col = next((col for col in target_columns if col in df.columns), None)
    
    if target_col:
        print(f"Target column '{target_col}' found in input data")
        y_true = df[target_col]
        X = df.drop(target_col, axis=1)
    else:
        print("No target column found in input data")
        X = df.copy()
        y_true = None
    
    exclude_cols = ['id', 'Year', 'year', 'location_id']
    X = X.drop([col for col in exclude_cols if col in X.columns], axis=1)
    
    print(f"Loading model from {args.model}")
    model, scaler, model_type = load_model(args.model)
    print(f"Loaded {model_type} model successfully")
    
    print("Making predictions...")
    predictions = make_predictions(model, X, scaler, model_type)
    
    output_df = pd.DataFrame({'predicted_yield': predictions})
    if y_true is not None:
        output_df['actual_yield'] = y_true
        
        from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
        rmse = np.sqrt(mean_squared_error(y_true, predictions))
        r2 = r2_score(y_true, predictions)
        mae = mean_absolute_error(y_true, predictions)
        
        print(f"Prediction Metrics:")
        print(f"RMSE: {rmse:.2f}")
        print(f"R²: {r2:.4f}")
        print(f"MAE: {mae:.2f}")
    
    output_df.to_csv(args.output, index=False)
    print(f"Predictions saved to {args.output}")
    
    if args.explain:
        print("Generating SHAP explanations...")
        shap_values, explainer = explain_predictions(model, X, model_type)
        
        print("Creating visualizations...")
        visualize_explanations(shap_values, X, args.viz_dir)
        print(f"Visualizations saved to {args.viz_dir}")
        
        feature_importance = np.abs(shap_values).mean(0)
        feature_importance_df = pd.DataFrame({
            'Feature': X.columns,
            'SHAP_importance': feature_importance
        }).sort_values('SHAP_importance', ascending=False)
        
        feature_importance_df.to_csv(os.path.join(args.viz_dir, 'feature_importance.csv'), index=False)
        print(f"Feature importance saved to {os.path.join(args.viz_dir, 'feature_importance.csv')}")

if __name__ == "__main__":
    main()
