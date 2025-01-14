import locale
import re
import ssl
import time
import warnings
from datetime import datetime, timedelta

import certifi
import folium
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import seaborn as sns
from itertools import combinations
from folium import Choropleth, Circle, Marker
from folium.plugins import HeatMap, MarkerCluster
from geopy.distance import geodesic
from geopy.geocoders import Nominatim
from scipy import stats
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from tqdm import tqdm, tqdm_notebook
from transliterate import translit
from yandex_geocoder import Client
from scipy.stats import chi2_contingency

warnings.filterwarnings("ignore")

RAND = 10

def fill_mode_group_by(df, group_by_col, mode_for_col):
    """
    Fills missing values in the 'mode_for_col' feature with the mode, considering the value of 'group_by_col'.

    Parameters:
        df (pd.DataFrame): DataFrame containing the 'group_by' and 'mode_for' features.
        group_by_col (str): Name of the 'group_by' feature.
        mode_for_col (str): Name of the 'mode_for' feature.

    Returns:
        pd.DataFrame: DataFrame with missing values in the 'mode_for' feature filled.
    """
    # Group data by 'group_by' and compute the mode for 'district'
    mode = df.groupby(group_by_col)[mode_for_col].transform(lambda x: x.mode().iloc[0])

    # Fill missing values in 'district' with the mode
    df[mode_for_col] = df[mode_for_col].fillna(mode)

    return df

# Function to process date
def parse_date(date_str, current_date):
    """
    Converts a date string into a datetime object. Handles the following cases:
    1. If the date is represented as "today" or "yesterday," returns the current date or yesterday's date, respectively.
    2. If the year is not specified in the string, adds the current year.
    3. Converts the date string into a datetime object in the format "%d %B %Y" (day, month, year).

    Parameters:
    date_str (str): Date string (e.g., 'today', 'yesterday', '15 October').

    Returns:
    datetime: A datetime object corresponding to the provided date string.
    """
    # Check for "today" and "yesterday"
    if "сегодня" in date_str.lower():
        return current_date
    elif "вчера" in date_str.lower():
        return current_date - timedelta(days=1)

    # Get the current year
    current_year = current_date.year

    # If the year is not specified in the string, add the current year
    if not any(char.isdigit() for char in date_str.split()[-1]):
        date_str = f"{date_str} {current_year}"

    # Convert the string into a datetime object
    return datetime.strptime(date_str, "%d %B %Y")

# Function to calculate the age of an announcement in months
def calculate_announcement_age(parsed_date, current_date):
    """
    Calculates the age of an announcement in months based on the publication date and the current date.

    Parameters:
    parsed_date (datetime): The publication date of the announcement.
    current_date (datetime): The current date.

    Returns:
    int or float: The age of the announcement in months. If the date is not recognized (NaN), returns NaN.
    """
    if pd.isnull(parsed_date):
        return np.nan  # Return NaN if the date is not recognized

    # Calculate the difference between the current date and the publication date
    delta = current_date - parsed_date

    # Return the number of full months
    return delta.days // 30  # Round the difference in days to the number of full months

# Function to check data normality
def check_normal(data: pd.Series, p_value: float = 0.05):
    """
    Checks whether the distribution of data is normal using the Shapiro-Wilk test.

    Parameters:
    data (pd.Series): Data to check for normality.
    p_value (float): Threshold p-value for normality check (default is 0.05).

    Returns:
    list: A list containing a message about the normality of the distribution ("normal" or "not normal") 
    and the rounded p-value (up to three decimal places).
    """
    if stats.shapiro(data).pvalue >= p_value:
        message = "normal"
    else:
        message = "not normal"

    return [message, round(stats.shapiro(data).pvalue, 3)]


# Function to calculate Cramér's V coefficient
def cramers_v(confusion_matrix):
    """
    Computes Cramér's V coefficient for an adjacency matrix (confusion matrix).
    Cramér's V coefficient is a measure of association strength between two categorical variables.
    """
    chi2 = chi2_contingency(confusion_matrix)[0]
    n = confusion_matrix.sum().sum()
    r, k = confusion_matrix.shape
    return np.sqrt(chi2 / (n * (min(r, k) - 1)))
