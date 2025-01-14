import numpy as np
import pandas as pd

from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LinearRegression
from sklearn.calibration import CalibratedClassifierCV

from lightgbm import LGBMRegressor
from lightgbm import early_stopping
import lightgbm as lgb
from catboost import Pool, CatBoostRegressor
from transliterate import translit

from optuna.visualization.matplotlib import plot_param_importances
from optuna.visualization.matplotlib import plot_optimization_history
from optuna.visualization.matplotlib import plot_timeline
from optuna.terminator import report_cross_validation_scores

from src.get_metrics import get_metrics_regression, rmsle, mean_absolute_error

import matplotlib.pyplot as plt
import seaborn as sns

from tqdm.notebook import tqdm
import optuna
import shap

import warnings
from warnings import simplefilter
warnings.filterwarnings("ignore")
simplefilter("ignore", category=RuntimeWarning)

RAND = 25
N_FOLDS = 5

def cyrillic_to_latin(df):
    """
    Converts text data in a DataFrame by replacing Cyrillic characters with Latin characters
    using transliteration.

    Parameters:
    df (pd.DataFrame): DataFrame containing the data to be transformed.

    Returns:
    pd.DataFrame: DataFrame with transformed string features where Cyrillic is replaced with Latin.
    """
    for col in df.columns:
        if df[col].dtype == 'object':  # Process only string features
            df[col] = df[col].apply(lambda x: translit(x, 'ru', reversed=True) if isinstance(x, str) else x)
    return df

def check_overfitting(model, X_train, y_train, X_test, y_test, metric_fun, target_name, loss_fun):
    """
    Checks for model overfitting by comparing metric values on the training and test sets.

    The function calculates the specified metric for the training and test sets and returns a DataFrame 
    with metrics for each set, the percentage difference between them (delta), and additional information.

    Parameters
    ----------
    model : estimator object
        Trained model with a `predict` method for generating predictions.
    X_train : array-like or DataFrame
        Features for the training set.
    y_train : array-like or Series
        True target values for the training set.
    X_test : array-like or DataFrame
        Features for the test set.
    y_test : array-like or Series
        True target values for the test set.
    metric_fun : function
        Function to evaluate model predictions, e.g., mean_absolute_error or mean_squared_error.
    target_name : str
        Name of the target variable.
    loss_fun : str
        Name of the loss function used (for informational purposes).

    Returns
    -------
    DataFrame
        DataFrame with the following columns:
        - 'target_name': Name of the target variable.
        - 'loss_fun': Name of the loss function used.
        - 'train': Metric value on the training set.
        - 'test': Metric value on the test set.
        - 'delta': Percentage difference between metric values on the training and test sets.
    """
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    
    value_train = metric_fun(y_train, y_pred_train)
    value_test = metric_fun(y_test, y_pred_test)
    delta = abs(value_train - value_test) / value_test * 100

    df_overfitting = pd.DataFrame({     
        'target_name': [target_name],
        'loss_fun': [loss_fun],
        'train': [value_train],
        'test': [value_test],
        'delta': [delta]
    })

    return df_overfitting


def cross_validation_regressor(X_train: pd.DataFrame,
                               y_train: pd.Series,
                               X_test: pd.DataFrame,
                               y_test: pd.Series,
                               clf,  # LGBMRegressor or CatBoostRegressor
                               params: dict,
                               cat_features: list = None,
                               eval_metric: str = 'mae',
                               early_stop: bool = False,
                               early_stopping_rounds: int = 100,
                               num_folds: int = 5,
                               random_state: int = 10,
                               shuffle: bool = True):
    """
    Perform cross-validation for regression tasks.

    Parameters:
    clf: Class of the model (LGBMRegressor or CatBoostRegressor).
    params: Model hyperparameters.
    cat_features: List of categorical features for CatBoost (not required for LGBM).
    eval_metric: Metric used for evaluation (default is MAE).
    early_stop: Flag for early stopping.
    early_stopping_rounds: Number of rounds for early stopping.
    num_folds: Number of folds for cross-validation.
    random_state: For reproducibility.

    Returns:
    score_oof: List of MAE scores for each fold.
    predictions_train: Predictions on the training set (Out-of-Fold).
    predictions_test: Averaged predictions on the test set.
    """
    folds = KFold(n_splits=num_folds, random_state=random_state, shuffle=shuffle)
    score_oof = []
    predictions_test = np.zeros(len(X_test))
    predictions_train = np.zeros(len(X_train))
    
    for fold, (train_index, val_index) in enumerate(folds.split(X_train)):
        X_train_, X_val = X_train.iloc[train_index], X_train.iloc[val_index]
        y_train_, y_val = y_train.iloc[train_index], y_train.iloc[val_index]

        # Create model with parameters
        if clf == lgb.LGBMRegressor:
            model = clf(**params, verbosity=-1)  # Pass verbosity=-1 at initialization
        else:
            model = clf(**params)

        if early_stop:
            if isinstance(model, lgb.LGBMRegressor):
                model.fit(X_train_, y_train_,
                          eval_set=[(X_val, y_val)],
                          eval_metric=eval_metric,
                          early_stopping_rounds=early_stopping_rounds)
            elif isinstance(model, CatBoostRegressor):
                model.fit(X_train_, y_train_,
                          eval_set=[(X_val, y_val)],
                          cat_features=cat_features,
                          early_stopping_rounds=early_stopping_rounds,
                          verbose=False)  # For CatBoostRegressor
        else:
            if isinstance(model, lgb.LGBMRegressor):
                model.fit(X_train_, y_train_,
                          categorical_feature=cat_features)
            elif isinstance(model, CatBoostRegressor):
                model.fit(X_train_, y_train_,
                          cat_features=cat_features,
                          verbose=False)  # For CatBoostRegressor

        # Predictions for validation set
        y_val_pred = model.predict(X_val)
        predictions_train[val_index] = y_val_pred
        
        # Evaluate quality for the current iteration
        mae = mean_absolute_error(y_val, y_val_pred)
        print(f"Fold", fold + 1, "Validation MAE: %.4f" % mae)
        score_oof.append(mae)
        
        # Predictions for the test set
        predictions_test += model.predict(X_test) / num_folds

    return score_oof, predictions_train, predictions_test, model


def calibrate_predictions(predictions_train, y_train, predictions_test):
    """
    Function to perform isotonic calibration based on OOF predictions.
    
    Arguments:
    predictions_train: np.array
        OOF predictions of the model on the training set (cross-validation).
    y_train: np.array
        True values for the training set.
    predictions_test: np.array
        Predictions of the model for the test set.
        
    Returns:
    calibrated_predictions_test: np.array
        Calibrated predictions for the test set.
    """

    # Initialize the isotonic regression model
    iso_reg = IsotonicRegression(out_of_bounds='clip')

    # Train the calibration model on OOF predictions and true values
    iso_reg.fit(predictions_train, y_train)

    # Apply isotonic regression to the test predictions
    calibrated_predictions_test = iso_reg.predict(predictions_test)

    return calibrated_predictions_test

def plot_residuals_vs_predictions(y_test, predictions_test):
    """
    Function to plot two graphs:
    1. Residuals vs Predicted values.
    2. Actual vs Predicted prices.
    
    Parameters:
    - y_test: Actual target values.
    - predictions_test: Predicted values from the model.
    """
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    
    # First plot: Residuals vs Predicted values
    residuals = y_test - predictions_test
    ax[0].scatter(predictions_test, residuals)
    ax[0].axhline(y=0, color='r', linestyle='--')
    ax[0].set_xlabel('Predicted prices')
    ax[0].set_ylabel('Residuals')
    ax[0].set_title('Residuals vs Predicted Prices')
    
    # Second plot: Actual vs Predicted prices
    ax[1].scatter(y_test, predictions_test)
    ax[1].plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red')  # Perfect line
    ax[1].set_xlabel('Real prices')
    ax[1].set_ylabel('Predicted prices')
    ax[1].set_title('Real vs Predicted Prices')
    
    # Show plots
    plt.tight_layout()
    plt.show()

class StackingRegressorCV:
    """
    Class to implement stacking regression with cross-validation.

    Parameters
    ----------
    trained_models : list
        List of pre-trained models to be used in stacking.
    
    meta_model : object
        Model to be used as the meta-model for predictions on meta-features.
    
    feature_sets : list
        List of feature sets corresponding to each pre-trained model.
    
    calibration_methods : list, optional
        List of calibration methods for each model (default None).
    
    n_folds : int, optional
        Number of folds for cross-validation (default 5).
    
    Attributes
    ----------
    calibrators : list
        List of calibrators used for each model.
    """

    def __init__(self, trained_models, meta_model, feature_sets, calibration_methods=None, n_folds=5):
        self.trained_models = trained_models
        self.meta_model = meta_model
        self.feature_sets = feature_sets
        self.calibration_methods = calibration_methods or [None] * len(trained_models)
        self.n_folds = n_folds
        self.calibrators = [None] * len(trained_models)  # List to store calibrators

    def fit_meta_model(self, X, y):
        """
        Trains the meta-model on meta-features derived from the validation set.

        Parameters
        ----------
        X : pandas.DataFrame
            Input data for training.
        
        y : pandas.Series
            Target values for training.
        
        Returns
        -------
        self : object
            Returns the instance of the class.
        """
        kf = KFold(n_splits=self.n_folds, shuffle=True, random_state=42)
        meta_features = np.zeros((X.shape[0], len(self.trained_models)))

        for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
            print(f"Fold {fold+1}/{self.n_folds}")
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

            for i, model in enumerate(self.trained_models):
                feature_subset = X_val[self.feature_sets[i]]  # Use validation feature set
                meta_features[val_idx, i] = model.predict(feature_subset)  # Get predictions on validation set

                # Calibrate predictions if required
                if self.calibration_methods[i] == 'isotonic':
                    calibrator = IsotonicRegression(out_of_bounds='clip')
                    calibrator.fit(meta_features[train_idx, i].reshape(-1, 1), y_train)  # Train calibrator
                    self.calibrators[i] = calibrator  # Save calibrator
                    meta_features[val_idx, i] = calibrator.predict(meta_features[val_idx, i].reshape(-1, 1))  # Calibrate predictions

        # Train the meta-model on meta-features
        self.meta_model.fit(meta_features, y)

        return self

    def predict(self, X):
        """
        Generates predictions based on meta-features derived from pre-trained models.

        Parameters
        ----------
        X : pandas.DataFrame
            Input data for prediction.

        Returns
        -------
        numpy.ndarray
            Predicted values from the meta-model.
        """
        # Generate meta-features
        meta_features = np.zeros((X.shape[0], len(self.trained_models)))

        for i, model in enumerate(self.trained_models):
            feature_subset = X[self.feature_sets[i]]  # Use feature set for prediction
            meta_features[:, i] = model.predict(feature_subset)  # Get predictions

            # Calibrate predictions if required
            if self.calibration_methods[i] == 'isotonic':
                calibrator = self.calibrators[i]
                if calibrator:  # Check if the calibrator was trained
                    meta_features[:, i] = calibrator.predict(meta_features[:, i].reshape(-1, 1))

        return self.meta_model.predict(meta_features)
    
    
def fill_mode_group_by(df, group_by_col, mode_for_col):
    """
    Fills missing values in the 'mode_for_col' feature with its mode, 
    considering the grouping by the 'group_by_col' feature.

    Parameters:
        df (pd.DataFrame): DataFrame containing the features 'group_by' and 'mode_for'.
        group_by_col (str): Name of the 'group_by' feature.
        mode_for_col (str): Name of the 'mode_for' feature.

    Returns:
        pd.DataFrame: DataFrame with missing values filled in the 'mode_for' feature.
    """
    # Group the data by 'group_by' and compute the mode for 'mode_for_col'
    mode = df.groupby(group_by_col)[mode_for_col].transform(lambda x: x.mode().iloc[0])

    # Fill missing values in 'mode_for_col' with the mode
    df[mode_for_col] = df[mode_for_col].fillna(mode)

    return df

def fill_missing_values(df, fill_dict):
    """
    Fills missing values in multiple features using the mode grouped by another feature.

    Parameters:
        df (pd.DataFrame): DataFrame containing the features with missing values.
        fill_dict (dict): Dictionary where keys are the features with missing values 
                          to be filled, and values are the features to group by 
                          for calculating the mode.

    Returns:
        pd.DataFrame: DataFrame with missing values filled in the specified features.
    """
    for mode_for_col, group_by_col in fill_dict.items():
        df = fill_mode_group_by(df, group_by_col, mode_for_col)
    return df

