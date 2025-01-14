import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, KFold

from sklearn.inspection import permutation_importance
from lightgbm import LGBMRegressor
from lightgbm import early_stopping
from catboost import CatBoostRegressor
from transliterate import translit

from src.get_metrics import get_metrics_regression, rmsle, mean_absolute_error

import matplotlib.pyplot as plt
import seaborn as sns
import shap


import warnings
from warnings import simplefilter

warnings.filterwarnings("ignore")
simplefilter("ignore", category=RuntimeWarning)

RAND = 19
N_FOLDS = 5


def cyrillic_to_latin(df):
    """
    Converts Cyrillic characters to Latin characters in all string columns of the DataFrame.

    The function processes all columns with the 'object' data type (strings) and applies transliteration 
    of Cyrillic text to Latin using the 'translit' method from the 'transliterate' library. 
    Non-string values remain unchanged.

    Arguments:
        df (pd.DataFrame): Input DataFrame containing text in Cyrillic in string columns.

    Returns:
        pd.DataFrame: A DataFrame where Cyrillic characters in string columns are replaced with Latin characters.
    """
    for col in df.columns:
        if df[col].dtype == 'object':  # Process only string columns
            df[col] = df[col].apply(
                lambda x: translit(x, 'ru', reversed=True) if isinstance(x, str) else x
            )
    return df


def check_overfitting(model, X_train, y_train, X_test, y_test, metric_fun):
    """
    Checks the model for overfitting in a regression task.

    The function calculates the metric value on both the training and testing datasets to determine 
    the accuracy difference and detect possible overfitting. The metric is specified by the user 
    through the `metric_fun` function.

    Arguments:
        model: Trained model for predictions.
        X_train (pd.DataFrame or np.array): Features of the training dataset.
        y_train (pd.Series or np.array): Target values of the training dataset.
        X_test (pd.DataFrame or np.array): Features of the testing dataset.
        y_test (pd.Series or np.array): Target values of the testing dataset.
        metric_fun (function): Function to calculate the metric (e.g., MAE, MSE, etc.).

    Returns:
        None. The results are printed to the console.
    """
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    value_train = metric_fun(y_train, y_pred_train)
    value_test = metric_fun(y_test, y_pred_test)

    print(f'{metric_fun.__name__} train: %.3f' % value_train)
    print(f'{metric_fun.__name__} test: %.3f' % value_test)
    print(f'delta = {(abs(value_train - value_test)/value_test*100):.1f} %')
    

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
    # Group data by 'group_by' and compute the mode for 'mode_for'
    mode = df.groupby(group_by_col)[mode_for_col].transform(lambda x: x.mode().iloc[0])

    # Fill missing values in 'mode_for' with the mode
    df[mode_for_col] = df[mode_for_col].fillna(mode)

    return df


def fill_missing_values(df, fill_dict):
    """
    Fills missing values in multiple features using the mode grouped by another feature.

    Parameters:
        df (pd.DataFrame): DataFrame containing features with missing values.
        fill_dict (dict): Dictionary where keys are features with missing values to be filled, 
                          and values are features to group by for filling using the mode.

    Returns:
        pd.DataFrame: DataFrame with missing values filled in the specified features.
    """
    for mode_for_col, group_by_col in fill_dict.items():
        df = fill_mode_group_by(df, group_by_col, mode_for_col)
    return df




