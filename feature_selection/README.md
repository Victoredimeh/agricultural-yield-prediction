# Feature Selection Directory

This directory contains the implementation of the Hybrid Feature Selection method developed in my Master's thesis research. The hybrid approach combines three complementary techniques to optimize model performance:

1. **Compression Analysis**: Eliminates highly correlated features to reduce redundancy
2. **LASSO Regression**: Identifies statistically significant features 
3. **SHAP Analysis**: Ranks features by importance for model interpretability

## Files

- `hybrid_feature_selector.py`: Main implementation of the hybrid feature selection method
- `feature_importance.py`: Utilities for analyzing feature importance
- `utils.py`: Helper functions for data processing and visualization

## Hybrid Feature Selection Method

The novel Hybrid Feature Selection method improved model performance significantly in the thesis research:

- Enhanced XGBoost accuracy by 10.0% (RMSE reduction from 2431.5 to 2187.3)
- Improved ANN performance by 9.3% (RMSE reduction from 2553.2 to 2314.6)
- Reduced dimensionality while preserving critical information for prediction

## Three-Stage Process

### Stage 1: Compression Analysis
This first step identifies and removes highly correlated features that provide redundant information. Features with correlation coefficients exceeding a threshold (typically r > 0.85) are removed to reduce multicollinearity.

### Stage 2: LASSO Regression
After removing redundant features, LASSO regression with L1 regularization is applied to identify statistically significant features. This step further reduces dimensionality by shrinking less important coefficients to zero.

### Stage 3: SHAP Analysis
The final stage uses SHAP (SHapley Additive exPlanations) to rank features by their impact on model predictions. This provides a clear, interpretable hierarchy of feature importance that enhances model transparency.

## Key Findings

The SHAP analysis identified the following as the most influential factors in crop yield prediction:

1. Rainfall (SHAP value: 2.37)
2. Temperature (SHAP value: 1.82)
3. Pesticide application (SHAP value: 1.43)

## Usage

To apply the hybrid feature selection method to your data:

```bash
python hybrid_feature_selector.py --input ../data/your_data.csv --output selected_features.json --threshold 0.85 --alpha 0.01
```
