# Data Directory

This directory contains sample datasets for the agricultural yield prediction model.

## Dataset Overview

The dataset used in this study consists of 28,242 records spanning multiple years (1990-2013) across 101 distinct geographical regions and 10 crop types, sourced from Kaggle. 

## Data Format

The sample dataset includes the following key variables:

- `hg/ha_yield`: Crop yield in hectograms per hectare (target variable)
- `average_rain_fall_mm_per_year`: Annual rainfall in millimeters
- `pesticides_tonnes`: Pesticide usage in tonnes
- `avg_temp`: Average temperature in degrees Celsius
- `fertilizer_kg_ha`: Fertilizer application in kg per hectare
- `soil_quality_index`: Index measuring soil quality
- `Area`: Geographic location where crops are grown
- `Item`: Type of crop (e.g., Potatoes, Maize, Rice)
- `Year`: Year of data collection

## Sample Data

The `sample_crop_data.csv` file contains a small subset of the full dataset, providing examples of the data structure without revealing the entire dataset. This sample is suitable for testing the prediction models and understanding the data format.

## Using Your Own Data

To use your own data with this model:
1. Format your CSV file to match the structure of the sample data
2. Ensure all required features are present
3. Place your file in this directory
4. Reference your filename in the model scripts

## Data Sourcing and Preprocessing

The original dataset was obtained from Kaggle. Prior to analysis, the following preprocessing steps were applied:

- Missing values were handled using median imputation for numerical variables (5.3% of the data contained missing values)
- Categorical variables were encoded using one-hot encoding
- Features were standardized to have zero mean and unit variance
- Outliers were identified using the IQR method and verified against agricultural literature

## Citation

When using this dataset, please cite the original source:

Edimeh, V. U. (2025). "Advancing agricultural yield prediction with explainable AI: An integrated approach using hybrid feature selection and comparative model analysis" (Master's thesis). Cyprus International University, Nicosia, Cyprus.
