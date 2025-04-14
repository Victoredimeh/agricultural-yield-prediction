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
    
    if hyper_tuni
