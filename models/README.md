# Models Directory

This directory contains the implementation of machine learning models used for agricultural yield prediction.

## Files

- `model_trainer.py`: Main script for training different ML models
- `prediction.py`: Functions for making predictions with trained models and explaining results
- `model_evaluation.py`: Utilities for evaluating and comparing model performance

## Models Implemented

The repository implements several machine learning models:

1. **XGBoost**: Gradient boosting algorithm that performs well with tabular data (best performer with R²=0.86, RMSE=2187.3)
2. **Artificial Neural Network (ANN)**: Deep learning approach for capturing complex patterns (R²=0.84, RMSE=2314.6)
3. **Random Forest**: Ensemble method that builds multiple decision trees (R²=0.83, RMSE=2492.1)
4. **Support Vector Machine (SVM)**: Kernel-based algorithm handling non-linear relationships (R²=0.75, RMSE=3204.8)
5. **K-Nearest Neighbors (KNN)**: Instance-based learning method (R²=0.68, RMSE=4103.2)
6. **Linear Regression**: Basic modeling technique (R²=0.62, RMSE=4557.6)

## Usage

To train a model:
```bash
python model_trainer.py --input ../data/your_data.csv --features ../feature_selection/selected_features.json --model xgb
```

To evaluate a model:
```bash
python model_evaluation.py --model ../models/xgb_model.joblib --test ../data/test_data.csv
```

To make predictions:
```bash
python prediction.py --input ../data/new_data.csv --model ../models/xgb_model.joblib
```

To generate explanations with SHAP:
```bash
python prediction.py --input ../data/new_data.csv --model ../models/xgb_model.joblib --explain
```

## Model Performance
Based on the thesis results, XGBoost outperformed other models, with ANN and Random Forest also showing strong performance. Linear Regression had the lowest accuracy, confirming the non-linear nature of agricultural relationships.
## Model Performance

Based on the thesis results, XGBoost outperformed other models, with ANN and Random Forest also showing strong performance. Linear Regression had the lowest accuracy, confirming the non-linear nature of agricultural relationships.

| Model | RMSE | R² | MAE |
|-------|------|------|------|
| XGBoost | 2187.3 | 0.86 | 1047.2 |
| ANN | 2314.6 | 0.84 | 1123.8 |
| Random Forest | 2492.1 | 0.83 | 1192.5 |
| SVM | 3204.8 | 0.75 | 1456.7 |
| KNN | 4103.2 | 0.68 | 1874.2 |
| Linear Regression | 4557.6 | 0.62 | 2083.1 |

## Implementation Details
 The hybrid feature selection method improved model performance by 10.0% for XGBoost and 9.3% for ANN
 SHAP analysis identified rainfall (SHAP value: 2.37), temperature (1.82), and pesticide application (1.43) as the most influential factors
 Models performed worse in areas with high climate variability, with accuracy declining by 16.4% in regions with high rainfall variability

