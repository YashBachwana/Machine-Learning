from typing import Union
import pandas as pd


def accuracy(y_hat: pd.Series, y: pd.Series) -> float:
    """
    Function to calculate the accuracy
    """

    """
    The following assert checks if sizes of y_hat and y are equal.
    """
    
    assert y_hat.size == y.size
    # TODO: Write here
    correct_predictions = (y_hat == y).sum()
    total_predictions = len(y)
    return correct_predictions/total_predictions*100  


def precision(y_hat: pd.Series, y: pd.Series, cls: Union[int, str]) -> float:
    """
    Function to calculate the precision
    """
    true_positives = ((y_hat == cls) & (y == cls)).sum()
    false_positives = ((y_hat == cls) & (y != cls)).sum()

    # Ensure that the denominator is not zero
    if true_positives + false_positives == 0:
        return 0.0

    # Calculate precision
    precision_value = true_positives / (true_positives + false_positives)
    return precision_value


def recall(y_hat: pd.Series, y: pd.Series, cls: Union[int, str]) -> float:
    """
    Function to calculate the recall
    """
    assert y_hat.index.equals(y.index), "Indices of y_hat and y must be the same."

    # Calculate true positives (TP) and false negatives (FN)
    true_positives = ((y_hat == cls) & (y == cls)).sum()
    false_negatives = ((y_hat != cls) & (y == cls)).sum()

    # Ensure that the denominator is not zero
    if true_positives + false_negatives == 0:
        return 0.0

    # Calculate recall
    recall_value = true_positives / (true_positives + false_negatives)
    return recall_value


def rmse(y_hat: pd.Series, y: pd.Series) -> float:
    """
    Function to calculate the root-mean-squared-error(rmse)
    """
    assert y_hat.index.equals(y.index), "Indices of y_hat and y must be the same."

    # Calculate squared differences between predicted and true values
    squared_errors = (y_hat - y) ** 2

    # Calculate mean squared error
    mean_squared_error = squared_errors.mean()

    # Calculate RMSE by taking the square root of mean squared error
    rmse_value = (mean_squared_error) ** (1/2)
    return rmse_value

def mse(Y: pd.Series):
    mean_target = Y.mean()
    return ((Y - mean_target) ** 2).mean()

def mae(y_hat: pd.Series, y: pd.Series) -> float:
    """
    Function to calculate the mean-absolute-error(mae)
    """
    assert y_hat.index.equals(y.index), "Indices of y_hat and y must be the same."

    # Calculate absolute differences between predicted and true values
    absolute_errors = abs(y_hat - y)

    # Calculate mean absolute error
    mean_absolute_error = absolute_errors.mean()

    return mean_absolute_error
