# Data Directory

This directory contains sample datasets for the agricultural yield prediction model.

## Sample Data

`sample_crop_data.csv` - A small anonymized dataset demonstrating the required format for the prediction model.

## Data Format

The model expects CSV files with the following columns:
- temperature_mean: Average temperature during growing season (°C)
- rainfall_total: Total rainfall during growing season (mm)
- soil_nitrogen: Nitrogen content in soil (ppm)
- soil_phosphorus: Phosphorus content in soil (ppm)
- soil_potassium: Potassium content in soil (ppm)
- humidity_mean: Average humidity during growing season (%)
- sunshine_hours: Total sunshine hours during growing season
- irrigation_amount: Applied irrigation water (mm)
- pest_severity: Pest infestation severity score (0-10)
- yield: Target variable - crop yield in tons/hectare

## Data Sources

## Data Sources

The dataset used for this research was sourced from Kaggle.com, a platform for data science competitions and datasets. The specific dataset contains agricultural data including environmental variables, soil properties, and resulting crop yields from various growing seasons.

All personally identifiable information and specific location data have been anonymized to protect privacy while maintaining the statistical relationships between features.

All personally identifiable information and specific location data have been anonymized to protect privacy while maintaining the statistical relationships between features.

## Using Your Own Data

To use your own data with this model:
1. Format your CSV file following the column structure above
2. Ensure all required features are present
3. Place your file in this directory
4. Reference your filename in the model scripts

## Full Dataset

Due to size limitations, the full dataset used in the thesis research is not included in this repository. For research purposes, please contact the author for access to the complete dataset.
