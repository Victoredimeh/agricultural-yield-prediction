# Agricultural Yield Prediction with Explainable AI
This repository contains the machine learning models and analysis developed during my Master's thesis research on agricultural yield prediction using explainable AI techniques.
## Project Overview
This project investigates how hybrid feature selection and explainable AI can improve agricultural yield prediction models. The research focuses on:
- Developing a hybrid feature selection method combining filter, wrapper, and embedded techniques
- Implementing multiple ML models (XGBoost, Random Forest, ANN) for yield prediction
- Using SHAP analysis to provide model interpretability
- Creating visualization tools to understand feature importance
## Results Highlights
- Improved prediction accuracy by 10% compared to traditional single-method approaches
- Identified key environmental variables that impact crop yields
- Developed an interpretable model that can be used by agricultural stakeholders
## Repository Structure
- `/data` - Sample datasets (anonymized)
- `/models` - Model training and evaluation code
- `/feature_selection` - Implementation of hybrid feature selection technique
- `/visualization` - SHAP analysis and feature importance visualization
- `/notebooks` - Jupyter notebooks demonstrating the analysis process
## Getting Started
### Prerequisites
- Python 3.8 or higher
- pip (Python package installer)
### Installation
1. Clone this repository:
   ```bash
   git clone https://github.com/Victoredimeh/agricultural-yield-prediction.git
   cd agricultural-yield-prediction
   ```
2. Create and activate a virtual environment (recommended):
    ```bash
    python -m venv env
    # On Windows
    env\Scripts\activate
    # On macOS/Linux
    source env/bin/activate
    ```bash
4. Install the required packages:
    ```bash
    pip install -r requirements.txt
    ```

### Using the Models

1. Place your agricultural data in the /data folder following the sample data format
2. Run the feature selection process:
   ```bash
    python feature_selection/hybrid_selector.py --input data/your_data.csv --output features.json
   ```
4. Train the model:
   ```bash
    python models/train_model.py --features features.json --model rf
   ```
6.  Generate predictions and explanations:
   ```bash
    python models/predict.py --input data/test_data.csv --model models/trained_model.joblib
```

### Running the Notebooks
1.  Start Jupyter:
   ```bash
    jupyter notebook
```
3.  Navigate to the /notebooks directory
4.  Open 01_data_exploration.ipynb to begin exploring the analysis process

### Visualizations
Run the Streamlit dashboard for interactive visualization:
```bash
streamlit run visualization/dashboard.py
```

