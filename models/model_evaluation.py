import argparse
import json
import numpy as np
import pandas as pd
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import tensorflow as tf
import shap

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

def evaluate_model(model, X_test, y_test, scaler=None, model_type=None):
    if scaler is not None:
        X_test_scaled = scaler.transform(X_test)
    else:
        X_test_scaled = X_test
    
    if model_type == 'ann':
        y_pred = model.predict(X_test_scaled).flatten()
    else:
        y_pred = model.predict(X_test_scaled)
    
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    
    print(f"Model Evaluation for {model_type if model_type else 'Model'}:")
    print(f"RMSE: {rmse:.2f}")
    print(f"R²: {r2:.4f}")
    print(f"MAE: {mae:.2f}")
    
    return {
        'model_type': model_type,
        'rmse': rmse,
        'r2': r2,
        'mae': mae,
        'y_pred': y_pred
    }

def evaluate_by_crop(model, X_test, y_test, scaler=None, model_type=None):
    if 'Item' not in X_test.columns:
        print("'Item' column not found in test data. Cannot evaluate by crop type.")
        return None
    
    crop_types = X_test['Item'].unique()
    
    results = {}
    
    for crop in crop_types:
        crop_indices = X_test['Item'] == crop
        X_crop = X_test[crop_indices]
        y_crop = y_test[crop_indices]
        
        if len(y_crop) < 10:
            print(f"Skipping {crop} - too few samples ({len(y_crop)})")
            continue
        
        if scaler is not None:
            X_crop_scaled = scaler.transform(X_crop)
        else:
            X_crop_scaled = X_crop
        
        if model_type == 'ann':
            y_pred = model.predict(X_crop_scaled).flatten()
        else:
            y_pred = model.predict(X_crop_scaled)
        
        rmse = np.sqrt(mean_squared_error(y_crop, y_pred))
        r2 = r2_score(y_crop, y_pred)
        mae = mean_absolute_error(y_crop, y_pred)
        
        results[crop] = {
            'rmse': rmse,
            'r2': r2,
            'mae': mae,
            'count': len(y_crop)
        }
        
        print(f"Evaluation for {crop} ({len(y_crop)} samples):")
        print(f"  RMSE: {rmse:.2f}")
        print(f"  R²: {r2:.4f}")
        print(f"  MAE: {mae:.2f}")
    
    return results

def visualize_predictions(y_test, y_pred, output_dir=None):
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred, alpha=0.5)
    
    min_val = min(min(y_test), min(y_pred))
    max_val = max(max(y_test), max(y_pred))
    plt.plot([min_val, max_val], [min_val, max_val], 'r--')
    
    plt.xlabel('Actual Yield')
    plt.ylabel('Predicted Yield')
    plt.title('Actual vs Predicted Yield')
    plt.grid(True)
    
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(os.path.join(output_dir, 'actual_vs_predicted.png'), dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()
    
    errors = y_test - y_pred
    
    plt.figure(figsize=(10, 6))
    sns.histplot(errors, kde=True)
    plt.xlabel('Prediction Error')
    plt.ylabel('Frequency')
    plt.title('Error Distribution')
    plt.grid(True)
    
    if output_dir:
        plt.savefig(os.path.join(output_dir, 'error_distribution.png'), dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()
    
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, errors, alpha=0.5)
    plt.axhline(y=0, color='r', linestyle='--')
    plt.xlabel('Actual Yield')
    plt.ylabel('Prediction Error')
    plt.title('Error vs Actual Yield')
    plt.grid(True)
    
    if output_dir:
        plt.savefig(os.path.join(output_dir, 'error_vs_actual.png'), dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

def analyze_crop_performance(crop_results, output_dir=None):
    crops = list(crop_results.keys())
    rmse_values = [crop_results[crop]['rmse'] for crop in crops]
    r2_values = [crop_results[crop]['r2'] for crop in crops]
    
    sorted_indices = np.argsort(r2_values)[::-1]
    crops = [crops[i] for i in sorted_indices]
    rmse_values = [rmse_values[i] for i in sorted_indices]
    r2_values = [r2_values[i] for i in sorted_indices]
    
    plt.figure(figsize=(12, 6))
    bars = plt.bar(crops, rmse_values)
    
    for i, bar in enumerate(bars):
        count = crop_results[crops[i]]['count']
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 50, 
                f'n={count}', ha='center', va='bottom', rotation=0)
    
    plt.xlabel('Crop Type')
    plt.ylabel('RMSE')
    plt.title('RMSE by Crop Type')
    plt.grid(axis='y')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(os.path.join(output_dir, 'rmse_by_crop.png'), dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()
    
    plt.figure(figsize=(12, 6))
    bars = plt.bar(crops, r2_values)
    
    for i, bar in enumerate(bars):
        count = crop_results[crops[i]]['count']
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, 
                f'n={count}', ha='center', va='bottom', rotation=0)
    
    plt.xlabel('Crop Type')
    plt.ylabel('R² Score')
    plt.title('R² Score by Crop Type')
    plt.grid(axis='y')
    plt.xticks(rotation=45, ha='right')
    plt.ylim(0, 1.0)
    plt.tight_layout()
    
    if output_dir:
        plt.savefig(os.path.join(output_dir, 'r2_by_crop.png'), dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

def generate_shap_analysis(model, X_test, model_type, output_dir=None):
    print("Generating SHAP analysis...")
    
    X_sample = X_test.sample(min(100, len(X_test)), random_state=42)
    
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
        X_sample = X_small_sample
    
    plt.figure(figsize=(12, 8))
    shap.summary_plot(shap_values, X_sample, show=False)
    plt.title(f'SHAP Feature Importance - {model_type.upper()}')
    plt.tight_layout()
    
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(os.path.join(output_dir, 'shap_summary.png'), dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()
    
    plt.figure(figsize=(12, 6))
    shap.summary_plot(shap_values, X_sample, plot_type="bar", show=False)
    plt.title(f'SHAP Feature Importance (Bar) - {model_type.upper()}')
    plt.tight_layout()
    
    if output_dir:
        plt.savefig(os.path.join(output_dir, 'shap_bar.png'), dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()
    
    feature_importance = np.abs(shap_values).mean(0)
    feature_importance_df = pd.DataFrame({
        'Feature': X_sample.columns,
        'SHAP_importance': feature_importance
    }).sort_values('SHAP_importance', ascending=False)
    
    if output_dir:
        feature_importance_df.to_csv(os.path.join(output_dir, 'shap_importance.csv'), index=False)
        print(f"SHAP importance saved to {os.path.join(output_dir, 'shap_importance.csv')}")
    
    return feature_importance_df

def main():
    parser = argparse.ArgumentParser(description='Evaluate trained model performance')
    parser.add_argument('--model', required=True, help='Path to trained model')
    parser.add_argument('--test', required=True, help='Path to test data CSV')
    parser.add_argument('--output_dir', default='evaluation_results', help='Directory to save evaluation results')
    parser.add_argument('--shap', action='store_true', help='Generate SHAP analysis')
    args = parser.parse_args()
    
    print(f"Loading model from {args.model}")
    model, scaler, model_type = load_model(args.model)
    print(f"Loaded {model_type} model successfully")
    
    print(f"Loading test data from {args.test}")
    test_df = pd.read_csv(args.test)
    
    target_cols = ['hg/ha_yield', 'yield']
    target_col = next((col for col in target_cols if col in test_df.columns), None)
    
    if not target_col:
        raise ValueError(f"Target column not found in {args.test}. Expected one of {target_cols}")
    
    y_test = test_df[target_col]
    
    exclude_cols = [target_col, 'id', 'Year', 'year', 'location_id']
    X_test = test_df.drop([col for col in exclude_cols if col in test_df.columns], axis=1)
    
    print(f"Test data loaded with {X_test.shape[1]} features and {X_test.shape[0]} samples")
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    evaluation = evaluate_model(model, X_test, y_test, scaler, model_type)
    
    with open(os.path.join(args.output_dir, 'evaluation_metrics.json'), 'w') as f:
        eval_for_json = {k: v for k, v in evaluation.items() if k != 'y_pred'}
        json.dump(eval_for_json, f, indent=4)
    
    visualize_predictions(y_test, evaluation['y_pred'], args.output_dir)
    
    if 'Item' in X_test.columns:
        print("\nEvaluating performance by crop type...")
        crop_results = evaluate_by_crop(model, X_test, y_test, scaler, model_type)
        
        if crop_results:
            with open(os.path.join(args.output_dir, 'crop_evaluation.json'), 'w') as f:
                json.dump(crop_results, f, indent=4)
            
            analyze_crop_performance(crop_results, args.output_dir)
    
    if args.shap:
        shap_dir = os.path.join(args.output_dir, 'shap_analysis')
        feature_importance = generate_shap_analysis(model, X_test, model_type, shap_dir)
        
        print("\nTop 10 features by SHAP importance:")
        print(feature_importance.head(10))
    
    print(f"\nEvaluation complete. Results saved to {args.output_dir}")

if __name__ == "__main__":
    main()
