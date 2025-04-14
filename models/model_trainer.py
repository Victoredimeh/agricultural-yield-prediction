import argparse
import json
import numpy as np
import pandas as pd
import joblib
import os
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression
import xgboost as xgb
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

def load_data(data_path, feature_path=None):
    print(f"Loading data from {data_path}")
    df = pd.read_csv(data_path)
    
    y = df['hg/ha_yield'] if 'hg/ha_yield' in df.columns else df['yield']
    
    if feature_path:
        with open(feature_path, 'r') as f:
            selected_features = json.load(f)
        X = df[selected_features]
        print(f"Using {len(selected_features)} selected features from {feature_path}")
    else:
        exclude_cols = ['hg/ha_yield', 'yield', 'Year', 'year', 'location_id', 'id']
        X = df.drop([col for col in exclude_cols if col in df.columns], axis=1, errors='ignore')
        print(f"Using all {X.shape[1]} available features")
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    print(f"Training with {X_train.shape[1]} features and {X_train.shape[0]} samples")
    return X_train, X_test, y_train, y_test

def train_random_forest(X_train, y_train, hyper_tuning=False):
    print("Training Random Forest model...")
    
    if hyper_tuning:
        param_grid = {
            'n_estimators': [100, 200, 300],
            'max_depth': [None, 10, 20, 30],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        }
        
        rf = RandomForestRegressor(random_state=42)
        grid_search = GridSearchCV(
            estimator=rf, 
            param_grid=param_grid,
            cv=5, 
            n_jobs=-1, 
            scoring='neg_mean_squared_error',
            verbose=1
        )
        grid_search.fit(X_train, y_train)
        
        best_rf = grid_search.best_estimator_
        print(f"Best RF parameters: {grid_search.best_params_}")
        
        return best_rf
    else:
        rf = RandomForestRegressor(
            n_estimators=100,
            random_state=42
        )
        rf.fit(X_train, y_train)
        return rf

def train_xgboost(X_train, y_train, hyper_tuning=False):
    print("Training XGBoost model...")
    
    if hyper_tuning:
        param_grid = {
            'n_estimators': [100, 200, 300],
            'learning_rate': [0.01, 0.05, 0.1],
            'max_depth': [3, 6, 9],
            'colsample_bytree': [0.7, 0.8, 0.9],
            'subsample': [0.8, 1.0]
        }
        
        xgb_model = xgb.XGBRegressor(random_state=42)
        grid_search = GridSearchCV(
            estimator=xgb_model, 
            param_grid=param_grid,
            cv=5, 
            n_jobs=-1, 
            scoring='neg_mean_squared_error',
            verbose=1
        )
        grid_search.fit(X_train, y_train)
        
        best_xgb = grid_search.best_estimator_
        print(f"Best XGB parameters: {grid_search.best_params_}")
        
        return best_xgb
    else:
        xgb_model = xgb.XGBRegressor(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=5,
            random_state=42
        )
        xgb_model.fit(X_train, y_train)
        return xgb_model

def train_ann(X_train, y_train):
    print("Training ANN model...")
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    model = Sequential([
        Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
        Dropout(0.2),
        Dense(32, activation='relu'),
        Dropout(0.2),
        Dense(1)
    ])
    
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                  loss='mean_squared_error')
    
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True
    )
    
    history = model.fit(
        X_train_scaled,
        y_train,
        epochs=50,
        batch_size=32,
        validation_split=0.2,
        callbacks=[early_stopping],
        verbose=1
    )
    
    return model, scaler, history

def train_svm(X_train, y_train):
    print("Training SVM model...")
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    svm_model = SVR(kernel='rbf', C=100, gamma=0.1)
    svm_model.fit(X_train_scaled, y_train)
    
    return svm_model, scaler

def train_knn(X_train, y_train):
    print("Training KNN model...")
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    knn_model = KNeighborsRegressor(n_neighbors=5)
    knn_model.fit(X_train_scaled, y_train)
    
    return knn_model, scaler

def train_linear_regression(X_train, y_train):
    print("Training Linear Regression model...")
    
    lr_model = LinearRegression()
    lr_model.fit(X_train, y_train)
    
    return lr_model

def evaluate_model(model, X_test, y_test, scaler=None, model_name=None):
    if scaler is not None:
        X_test = scaler.transform(X_test)
    
    if isinstance(model, tf.keras.Model):
        y_pred = model.predict(X_test).flatten()
    else:
        y_pred = model.predict(X_test)
    
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    
    print(f"Model Evaluation{' for ' + model_name if model_name else ''}:")
    print(f"RMSE: {rmse:.2f}")
    print(f"R²: {r2:.4f}")
    print(f"MAE: {mae:.2f}")
    
    return rmse, r2, mae, y_pred

def main():
    parser = argparse.ArgumentParser(description='Train yield prediction model')
    parser.add_argument('--input', required=True, help='Path to input CSV data')
    parser.add_argument('--features', help='Path to selected features JSON (optional)')
    parser.add_argument('--model', default='xgb', choices=['rf', 'xgb', 'ann', 'svm', 'knn', 'lr', 'all'],
                       help='Model type to train (rf=Random Forest, xgb=XGBoost, ann=Neural Network, svm=Support Vector Machine, knn=K-Nearest Neighbors, lr=Linear Regression, all=All models)')
    parser.add_argument('--output', default='../models/', help='Output directory for saved models')
    parser.add_argument('--hyper_tuning', action='store_true', help='Perform hyperparameter tuning')
    args = parser.parse_args()
    
    os.makedirs(args.output, exist_ok=True)
    
    X_train, X_test, y_train, y_test = load_data(args.input, args.features)
    
    results = {}
    
    if args.model == 'rf' or args.model == 'all':
        print("\nTraining Random Forest model...")
        rf_model = train_random_forest(X_train, y_train, args.hyper_tuning)
        rmse, r2, mae, _ = evaluate_model(rf_model, X_test, y_test, model_name="Random Forest")
        joblib.dump(rf_model, f"{args.output}rf_model.joblib")
        print(f"Random Forest model saved to {args.output}rf_model.joblib")
        results['Random Forest'] = {'rmse': rmse, 'r2': r2, 'mae': mae}
    
    if args.model == 'xgb' or args.model == 'all':
        print("\nTraining XGBoost model...")
        xgb_model = train_xgboost(X_train, y_train, args.hyper_tuning)
        rmse, r2, mae, _ = evaluate_model(xgb_model, X_test, y_test, model_name="XGBoost")
        joblib.dump(xgb_model, f"{args.output}xgb_model.joblib")
        print(f"XGBoost model saved to {args.output}xgb_model.joblib")
        results['XGBoost'] = {'rmse': rmse, 'r2': r2, 'mae': mae}
    
    if args.model == 'ann' or args.model == 'all':
        print("\nTraining Neural Network model...")
        ann_model, scaler, history = train_ann(X_train, y_train)
        rmse, r2, mae, _ = evaluate_model(ann_model, X_test, y_test, scaler, model_name="Neural Network")
        ann_model.save(f"{args.output}ann_model")
        joblib.dump(scaler, f"{args.output}ann_scaler.joblib")
        print(f"Neural Network model saved to {args.output}ann_model")
        print(f"Feature scaler saved to {args.output}ann_scaler.joblib")
        results['Neural Network'] = {'rmse': rmse, 'r2': r2, 'mae': mae}
    
    if args.model == 'svm' or args.model == 'all':
        print("\nTraining SVM model...")
        svm_model, scaler = train_svm(X_train, y_train)
        rmse, r2, mae, _ = evaluate_model(svm_model, X_test, y_test, scaler, model_name="SVM")
        joblib.dump(svm_model, f"{args.output}svm_model.joblib")
        joblib.dump(scaler, f"{args.output}svm_scaler.joblib")
        print(f"SVM model saved to {args.output}svm_model.joblib")
        print(f"Feature scaler saved to {args.output}svm_scaler.joblib")
        results['SVM'] = {'rmse': rmse, 'r2': r2, 'mae': mae}
    
    if args.model == 'knn' or args.model == 'all':
        print("\nTraining KNN model...")
        knn_model, scaler = train_knn(X_train, y_train)
        rmse, r2, mae, _ = evaluate_model(knn_model, X_test, y_test, scaler, model_name="KNN")
        joblib.dump(knn_model, f"{args.output}knn_model.joblib")
        joblib.dump(scaler, f"{args.output}knn_scaler.joblib")
        print(f"KNN model saved to {args.output}knn_model.joblib")
        print(f"Feature scaler saved to {args.output}knn_scaler.joblib")
        results['KNN'] = {'rmse': rmse, 'r2': r2, 'mae': mae}
    
    if args.model == 'lr' or args.model == 'all':
        print("\nTraining Linear Regression model...")
        lr_model = train_linear_regression(X_train, y_train)
        rmse, r2, mae, _ = evaluate_model(lr_model, X_test, y_test, model_name="Linear Regression")
        joblib.dump(lr_model, f"{args.output}lr_model.joblib")
        print(f"Linear Regression model saved to {args.output}lr_model.joblib")
        results['Linear Regression'] = {'rmse': rmse, 'r2': r2, 'mae': mae}
    
    with open(f"{args.output}model_evaluation_results.json", 'w') as f:
        json.dump(results, f, indent=4)
    print(f"Model evaluation results saved to {args.output}model_evaluation_results.json")

if __name__ == "__main__":
    main()
