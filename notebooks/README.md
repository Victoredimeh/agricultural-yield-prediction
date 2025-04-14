# Notebooks Directory

This directory contains Jupyter notebooks that demonstrate the analysis process used in this thesis research.

## Notebooks

1. **01_data_exploration.ipynb**: Exploratory data analysis of the agricultural dataset
2. **02_feature_selection.ipynb**: Implementation and analysis of the Hybrid Feature Selection method
3. **03_model_comparison.ipynb**: Training and evaluation of different machine learning models

## Data Exploration

The data exploration notebook provides insights into the dataset used for this research:
- Dataset overview and characteristics (28,242 records spanning 101 regions and 10 crop types)
- Visualizations of yield distributions across different crops
- Analysis of relationships between environmental variables and crop yields
- Identification of trends and patterns in the data

## Feature Selection

The feature selection notebook demonstrates the three-stage Hybrid Feature Selection method:
1. **Compression Analysis**: Eliminating highly correlated features
2. **LASSO Regression**: Identifying statistically significant features 
3. **SHAP Analysis**: Ranking features by importance

This process improved model performance significantly:
- Enhanced XGBoost accuracy by 10.0% (RMSE reduction from 2431.5 to 2187.3)
- Improved ANN performance by 9.3% (RMSE reduction from 2553.2 to 2314.6)

## Model Comparison

The model comparison notebook provides a comprehensive evaluation of six machine learning models:
1. **XGBoost**: Best performer (R²=0.86, RMSE=2187.3)
2. **Artificial Neural Network**: Second best (R²=0.84, RMSE=2314.6)
3. **Random Forest**: Third best (R²=0.83, RMSE=2492.1)
4. **Support Vector Machine**: Moderate (R²=0.75, RMSE=3204.8)
5. **K-Nearest Neighbors**: Lower (R²=0.68, RMSE=4103.2)
6. **Linear Regression**: Baseline (R²=0.62, RMSE=4557.6)

The notebook includes detailed visualizations of model performance, feature importance analysis, and crop-specific evaluations.

## Usage

To run these notebooks:
1. Ensure you have Jupyter installed (`pip install jupyter`)
2. Navigate to this directory
3. Start Jupyter Notebook server (`jupyter notebook`)
4. Open any of the notebooks to explore the analysis
