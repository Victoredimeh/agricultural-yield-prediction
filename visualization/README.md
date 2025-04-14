# Visualization Directory

This directory contains code for visualizing model outputs and feature importance using SHAP (SHapley Additive exPlanations) analysis. The visualizations help interpret the complex machine learning models developed in this thesis.

## Files

- `shap_visualizer.py`: Generates SHAP-based visualizations for model interpretation
- `dashboard.py`: Streamlit web application for interactive visualization
- `plot_utils.py`: Utility functions for creating various plots

## SHAP Analysis Results

The SHAP analysis identified the following as the most influential factors in crop yield prediction:

1. Rainfall (SHAP value: 2.37)
2. Temperature (1.82)
3. Pesticide application (1.43)

These findings align with established agricultural knowledge while quantifying the precise impact of each factor on yield predictions.

## Visualization Types

### SHAP Summary Plots
Summary plots show the impact of each feature on the model output. Features are ranked by importance, with each point representing a sample in the dataset. The color indicates whether the feature value is high (red) or low (blue) for that sample.

### SHAP Dependence Plots
Dependence plots show how the model output changes based on a single feature. These plots reveal the relationship between a feature and the predicted yield, accounting for the average effects of all other features.

### Feature Importance Plots
Bar charts showing the relative importance of each feature based on mean absolute SHAP values. These plots provide a clear ranking of feature importance.

## Web Dashboard

The Streamlit dashboard (`dashboard.py`) provides an interactive interface for:

1. Inputting agricultural and environmental parameters
2. Generating yield predictions using the trained models
3. Visualizing feature importance and prediction explanations
4. Comparing results across different models

## Usage

To generate SHAP visualizations for a trained model:

```bash
python shap_visualizer.py --model ../models/xgb_model.joblib --data ../data/test_data.csv --output visualization_output
```

To run the interactive Streamlit dashboard:
```bash
streamlit run dashboard.py
```
