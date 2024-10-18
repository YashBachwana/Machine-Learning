import pandas as pd
import numpy as np
from metrics import *

def check_ifreal(y: pd.Series) -> bool:
    """
    Function to check if the given series has real or discrete values
    """
    if y.nunique() < len(y)/5:
        return 0
    else:
        return 1
    
def entropy(Y: pd.Series) -> float:
    """
    Function to calculate the entropy
    """
    probabilities = Y.value_counts(normalize=True)
    probabilities = probabilities.replace(0, 1e-100)
    return -np.sum(probabilities * np.log2(probabilities))

def gini_index(Y: pd.Series) -> float:
    """
    Function to calculate the gini index
    """
    value_counts_series = Y.value_counts(normalize=True)
    gini = 1 - sum((p_i ** 2) for p_i in value_counts_series)
    return gini

def reduction_gini_index(Y: pd.Series, indices: list ) -> int:
    total_entropy = gini_index(Y)

    # Calculate the weighted average entropy after each split
    split_entropy = 0
    total_instances = len(Y)

    for split_indices in indices:
        split_size = len(split_indices)
        split_entropy += (split_size / total_instances) * gini_index(Y.loc[split_indices])

    # Overall Information Gain is the reduction in entropy
    overall_information_gain = total_entropy - split_entropy
    return overall_information_gain

def criteria_(criterion,Y: pd.Series, indices: list ) -> int:
    if criterion == "information_gain":
        return information_gain(Y,indices)        
    else :
        return reduction_gini_index(Y,indices)    

def information_gain(Y: pd.Series, indices: list ) -> float:
    """
    Function to calculate the information gain
    """
    total_entropy = entropy(Y)

    # Calculate the weighted average entropy after each split
    split_entropy = 0
    total_instances = len(Y)

    for split_indices in indices:
        split_size = len(split_indices)
        split_entropy += (split_size / total_instances) * entropy(Y.loc[split_indices])

    # Overall Information Gain is the reduction in entropy
    overall_information_gain = total_entropy - split_entropy
    return overall_information_gain

def reduction_mse(Y: pd.Series, indices: list):

    initial_mse = mse(Y)

    weighted_total_mse = 0.0
    total_instances = len(Y)

    for split_indices in indices:
        split_size = len(split_indices)
        weighted_total_mse += (split_size / total_instances) * mse(Y.loc[split_indices])
        
    return initial_mse - weighted_total_mse

def opt_split_attribute(X: pd.DataFrame, y: pd.Series, criterion, features: pd.Series):
    """
    Function to find the optimal attribute to split about.
    If needed you can split this function into 2, one for discrete and one for real valued features.
    You can also change the parameters of this function according to your implementation.

    features: pd.Series is a list of all the attributes we have to split upon

    return: attribute to split upon
    """

    # According to wheather the features are real or discrete valued and the criterion, find the attribute from the features series with the maximum information gain (entropy or varinace based on the type of output) or minimum gini index (discrete output).
    if check_ifreal(X[1]) == 1 and check_ifreal(y) == 1:
        mat = []
        for j in range(5):
            splits = []
            for i in range(len(X[i])-1):
                mean1 = np.mean(X.loc[X[j]] <= (X[j][i] + X[j][i]) / 2, )
                mse1 = np.sqrt(np.mean((y(lambda x : x < (X[i])+X[i])/2) - mean1)**2)
                mean2 = np.mean(y(lambda x : x > (X[i])+X[i])/2)
                mse2 = np.sqrt(np.mean((y(lambda x : x > (X[i])+X[i])/2) - mean1)**2)  
                splits.append(mse1+mse2)
            mat.append(splits)
