# Decision Tree Implementation

This repository contains a comprehensive implementation of a Decision Tree algorithm capable of handling various input-output types:
- Discrete input, discrete output
- Real input, real output
- Real input, discrete output
- Discrete input, real output
There are multiple versions of the Decision Tree algorithm like ID3, C4.5, CART, etc. This implementation is based on ID3 algorithm.

## Features

- **Tree Construction**: Handles both classification and regression tasks using a custom-built decision tree.
- **Split Criteria**: Supports Information Gain and Gini Index for classification, and Mean Squared Error (MSE) reduction for regression.
- **Evaluation Metrics**: Provides accuracy, precision, recall, RMSE (Root Mean Squared Error), and MAE (Mean Absolute Error) as evaluation metrics.
  
## File Descriptions

### `base.py`
This file contains the core Decision Tree implementation. The tree construction logic is split into four cases based on the type of input and output:

- **Discrete to Discrete**
- **Discrete to Real**
- **Real to Discrete**
- **Real to Real**

#### Key Classes and Methods:
- `DecisionTreeNode`: Defines the structure of the nodes used in the tree.
- `DecisionTree`: A class for building and fitting decision trees, with methods:
  - `fit()`: To train the decision tree.
  - `predict()`: To make predictions based on trained trees.
  - `plot()`: To visualize the tree structure.

### `metrics.py`
This file contains utility functions to evaluate the performance of the decision tree model. Metrics include:

- `accuracy()`: Accuracy for classification.
- `precision()`: Precision for classification.
- `recall()`: Recall for classification.
- `rmse()`: Root Mean Squared Error for regression.
- `mae()`: Mean Absolute Error for regression.

### `utils.py`
Helper functions for:
- Calculating entropy, Gini index, and information gain for classification.
- Checking if a feature is real-valued or discrete.

#### Key Functions:
- `entropy()`: Computes entropy for classification.
- `gini_index()`: Computes Gini Index.
- `information_gain()`: Computes Information Gain.
- `reduction_mse()`: Reduces MSE for regression splits.
- `check_ifreal()`: Determines if a feature is real or discrete.

### `usage.py`
This file contains examples demonstrating how to use the decision tree for different types of inputs and outputs, along with metrics evaluation:

- **Test Case 1**: Real input, real output (regression).
- **Test Case 2**: Real input, discrete output (classification).
- **Test Case 3**: Discrete input, discrete output (classification).

### sklearn_implementation_decision_tree.ipynb
This notebook provides the sklearn implementation of the Decision Tree algorithm. Sklearn internally uses CART algorithm for Decision Tree implementation.

