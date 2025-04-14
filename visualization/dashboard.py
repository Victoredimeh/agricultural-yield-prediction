import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
import os
import tensorflow as tf
import shap
from sklearn.preprocessing import StandardScaler

# Set page configuration
st.set_page_config(
    page_title="Agricultural Yield Prediction Dashboard",
    page_icon="🌾",
    layout="wide"
)

# App title and description
st.title("Agricultural Yield Prediction Dashboard")
st.markdown("""
This interactive dashboard allows you to predict crop yields based on environmental and agricultural inputs.
The predictions are powered by machine learning models developed as part of a Master's thesis research project.
""")

# Function to load models
@st.cache_resource
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
            st.error(f"Could not load model from {model_path}")
            return None, None, None
    
    return model, scaler, model_type

# Function to make predictions
def predict(model, input_data, scaler=None, model_type=None):
    # Convert input data to DataFrame
    input_df = pd.DataFrame([input_data])
    
    # Apply scaler if provided
    if scaler is not None:
        input_scaled = scaler.transform(input_df)
    else:
        input_scaled = input_df
    
    # Make prediction based on model type
    if model_type == 'ann':
        prediction = model.predict(input_scaled).flatten()[0]
    else:
        prediction = model.predict(input_scaled)[0]
    
    return prediction

# Function to generate SHAP values
@st.cache_data
def get_shap_values(model, input_data, model_type, scaler=None):
    # Convert input data to DataFrame
    input_df = pd.DataFrame([input_data])
    
    # Apply scaler if provided
    if scaler is not None:
        input_scaled = scaler.transform(input_df)
    else:
        input_scaled = input_df
    
    # Generate SHAP values based on model type
    if model_type in ['rf', 'xgb']:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(input_df)
    else:
        # For non-tree models, use KernelExplainer
        # We need a background dataset for this
        # For simplicity, we'll use the input data itself as background
        if model_type == 'ann':
            def predict_fn(x):
                return model.predict(x).flatten()
        else:
            def predict_fn(x):
                return model.predict(x)
        
        explainer = shap.KernelExplainer(predict_fn, input_scaled)
        shap_values = explainer.shap_values(input_scaled)
    
    return shap_values, explainer, input_df

# Sidebar for model selection
st.sidebar.header("Model Selection")
model_options = {
    "XGBoost": "../models/xgb_model.joblib",
    "Neural Network": "../models/ann_model",
    "Random Forest": "../models/rf_model.joblib",
    "Support Vector Machine": "../models/svm_model.joblib",
    "K-Nearest Neighbors": "../models/knn_model.joblib",
    "Linear Regression": "../models/lr_model.joblib"
}
selected_model = st.sidebar.selectbox("Select Model", list(model_options.keys()))
model_path = model_options[selected_model]

# Load model
try:
    model, scaler, model_type = load_model(model_path)
    st.sidebar.success(f"Model loaded successfully: {model_type}")
except Exception as e:
    st.sidebar.error(f"Error loading model: {e}")
    model, scaler, model_type = None, None, None

# Sidebar for input parameters
st.sidebar.header("Input Parameters")

# Crop type selection
crop_options = ["Maize", "Rice", "Wheat", "Potatoes", "Cassava", "Soybeans", "Sorghum", "Sweet Potatoes", "Yams", "Plantains"]
selected_crop = st.sidebar.selectbox("Crop Type", crop_options)

# Environmental parameters
st.sidebar.subheader("Environmental Parameters")
rainfall = st.sidebar.slider("Average Rainfall (mm/year)", 300, 2000, 1000)
temperature = st.sidebar.slider("Average Temperature (°C)", 5, 35, 20)
pesticides = st.sidebar.slider("Pesticide Usage (tonnes)", 0, 40000, 20000)
soil_quality = st.sidebar.slider("Soil Quality Index (0-1)", 0.0, 1.0, 0.7, 0.01)
fertilizer = st.sidebar.slider("Fertilizer Application (kg/ha)", 0, 200, 100)

# Generate input data dictionary
input_data = {
    f"Item_{selected_crop}": 1,  # One-hot encoding for selected crop
    "average_rain_fall_mm_per_year": rainfall,
    "avg_temp": temperature,
    "pesticides_tonnes": pesticides,
    "soil_quality_index": soil_quality,
    "fertilizer_kg_ha": fertilizer
}

# For all other crops, set to 0 (one-hot encoding)
for crop in crop_options:
    if crop != selected_crop:
        input_data[f"Item_{crop}"] = 0

# Action button
predict_btn = st.sidebar.button("Predict Yield")

# Main content area
if model is not None and predict_btn:
    # Display selected parameters
    st.header("Selected Parameters")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Crop Type", selected_crop)
        st.metric("Average Rainfall", f"{rainfall} mm/year")
    with col2:
        st.metric("Average Temperature", f"{temperature} °C")
        st.metric("Pesticide Usage", f"{pesticides} tonnes")
    with col3:
        st.metric("Soil Quality Index", f"{soil_quality:.2f}")
        st.metric("Fertilizer Application", f"{fertilizer} kg/ha")
    
    # Make prediction
    prediction = predict(model, input_data, scaler, model_type)
    
    # Display prediction
    st.header("Yield Prediction")
    st.metric("Predicted Yield", f"{prediction:.2f} hg/ha")
    st.metric("Equivalent", f"{prediction/10000:.2f} tonnes/ha")
    
    # Create a gauge chart to visualize the prediction
    fig, ax = plt.subplots(figsize=(10, 2))
    
    # Define gauge ranges (based on typical yield ranges)
    low_yield = 50000  # 5 tonnes/ha
    medium_yield = 100000  # 10 tonnes/ha
    high_yield = 200000  # 20 tonnes/ha
    
    # Determine the yield category
    if prediction < low_yield:
        category = "Low"
        color = "red"
    elif prediction < medium_yield:
        category = "Medium"
        color = "orange"
    else:
        category = "High"
        color = "green"
    
    # Create a horizontal bar representing the gauge
    ax.barh(["Yield"], [high_yield], color="lightgrey", height=0.5)
    ax.barh(["Yield"], [prediction], color=color, height=0.5)
    
    # Add markers for the different yield categories
    for yield_val, label in [(0, "0"), (low_yield, "Low"), (medium_yield, "Medium"), (high_yield, "High")]:
        ax.axvline(x=yield_val, color="black", linestyle="--", alpha=0.5)
        ax.text(yield_val, 0.25, label, ha="center", fontsize=10)
    
    # Add prediction marker
    ax.axvline(x=prediction, color="black", linestyle="-", linewidth=2)
    ax.text(prediction, 1.5, f"{prediction:.0f} hg/ha\n({prediction/10000:.2f} t/ha)", 
            ha="center", fontsize=12, fontweight="bold", bbox=dict(facecolor="white", alpha=0.8))
    
    # Set title and remove axes ticks and labels
    ax.set_title(f"Yield Prediction: {category} Yield")
    ax.set_xlim(0, high_yield * 1.1)
    ax.set_ylim(0, 2)
    ax.set_yticks([])
    ax.set_xticks([])
    plt.tight_layout()
    
    st.pyplot(fig)
    
    # Generate SHAP explanation
    try:
        st.header("Prediction Explanation")
        st.write("The chart below shows how each factor contributed to the yield prediction:")
        
        shap_values, explainer, input_df = get_shap_values(model, input_data, model_type, scaler)
        
        # Create waterfall plot
        fig, ax = plt.subplots(figsize=(10, 6))
        shap.waterfall_plot(
            shap.Explanation(
                values=shap_values[0] if isinstance(shap_values, list) else shap_values,
                base_values=explainer.expected_value if hasattr(explainer, "expected_value") else 0,
                data=input_df.values[0],
                feature_names=input_df.columns.tolist()
            ),
            show=False
        )
        plt.title("SHAP Waterfall Plot - Feature Contributions")
        plt.tight_layout()
        
        st.pyplot(fig)
        
        # Feature importance summary
        st.subheader("Feature Importance")
        feature_importance = pd.DataFrame({
            'Feature': input_df.columns,
            'SHAP_value': np.abs(shap_values[0] if isinstance(shap_values, list) else shap_values)
        }).sort_values('SHAP_value', ascending=False)
        
        # Display top features table
        st.table(feature_importance.head(10))
        
    except Exception as e:
        st.warning(f"Could not generate SHAP explanation: {e}")

else:
    # Display dashboard information when no prediction is made
    st.header("Welcome to the Agricultural Yield Prediction Dashboard")
    st.write("""
    This tool uses machine learning models to predict crop yields based on environmental and agricultural inputs.
    
    ### How to use:
    1. Select a model from the sidebar
    2. Choose a crop type
    3. Adjust environmental parameters (rainfall, temperature, etc.)
    4. Click "Predict Yield" to see the results
    
    ### Models Available:
    - **XGBoost**: Best overall performer (R²=0.86, RMSE=2187.3)
    - **Neural Network**: Second best performer (R²=0.84, RMSE=2314.6)
    - **Random Forest**: Third best performer (R²=0.83, RMSE=2492.1)
    - **SVM**: Moderate performance (R²=0.75, RMSE=3204.8)
    - **KNN**: Lower performance (R²=0.68, RMSE=4103.2)
    - **Linear Regression**: Baseline model (R²=0.62, RMSE=4557.6)
    
    ### Key Factors Affecting Yield:
    Based on SHAP analysis, the following factors have the most impact on yield prediction:
    1. Rainfall (SHAP value: 2.37)
    2. Temperature (SHAP value: 1.82)
    3. Pesticide application (SHAP value: 1.43)
    """)
    
    # Display SHAP summary image
    st.image("../feature_selection/3_shap_analysis/shap_summary.png", 
             caption="SHAP Summary Plot showing feature importance", use_column_width=True)

# Footer
st.markdown("---")
st.markdown("""
**About**: This dashboard is based on research from the Master's thesis "Advancing Agricultural Yield Prediction with Explainable AI: An Integrated Approach Using Hybrid Feature Selection and Comparative Model Analysis" by Victor U. Edimeh at Cyprus International University, 2025.
""")
