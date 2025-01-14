<<<<<<< HEAD
# pet1
Predicting real estate prices using CatBoost and LightGBM
=======
# Predicting Apartment Prices in the Cities of the Russian Far East Based on Open Sources

## Project Description
This research project focuses on exploring data and selecting a classical machine learning model to predict apartment prices in various cities of the Russian Far East. The project utilizes data from open sources collected in August 2024. It includes EDA (Exploratory Data Analysis), feature engineering, metric analysis on baseline algorithms (CatBoost and LightGBM), tuning of selected models, model calibration, and stacking of the most effective algorithms.

## Project Structure
```
PET1-project/
│
├── notebooks/                  # Jupyter notebooks
│   ├── 00parsing.ipynb          # Data parsing.
│   ├── 01EDA.ipynb              # EDA, data analysis, and feature engineering.
│   ├── 02Baseline.ipynb         # Training baseline algorithms and analyzing metrics.
│   ├── 03Tuning_Stacking.ipynb  # Tuning, calibration, stacking, and final conclusions.
│
├── src/                        # Python scripts for data processing and custom functions
│   ├── parsing_fun.py           # Functions for data parsing.
│   ├── baseline_fun.py          # Functions for baseline algorithms.
│   ├── eda_fun.py               # Functions for EDA and data analysis.
│   ├── stack_fun.py             # Functions for tuning, calibration, and stacking.
│   ├── get_metrics.py           # Functions for obtaining metrics.
│
├── data/                       # Raw and processed data
│   ├── data.csv                 # Raw data.
│   ├── df_base.csv              # Data for baseline algorithms and feature development.
│   ├── df_eda_lat_lon.csv       # Intermediate data with geolocation information.
│   ├── df_eda.csv               # Data after EDA and feature engineering.
│
├── requirements.txt            # Python dependencies.
└── README.md                   # Project documentation.
```
>>>>>>> 8e6dfa8 (Initial commit: project setup)
