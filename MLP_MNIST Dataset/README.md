# MNIST Digit Classification using Multi-Layer Perceptron (MLP)

## Overview
This project focuses on training a Multi-Layer Perceptron (MLP) on the MNIST dataset, evaluating its performance against Random Forest (RF) and Logistic Regression models. The analysis includes F1-score and confusion matrix metrics, alongside visualizations using t-SNE to understand the embeddings from the MLP's hidden layers.

## Datasets
- **MNIST Dataset**: 
  - Contains 60,000 training images and 10,000 test images of handwritten digits (0-9).
- **Fashion-MNIST Dataset**: 
  - Used for testing the generalization of the trained MLP model on a different dataset containing fashion items.

## Methodology

### 1. Data Preparation
- Use the original MNIST dataset 

### 2. Model Architecture
- **MLP Architecture**:
  - Input Layer: 784 neurons (28x28 pixel images)
  - Hidden Layer 1: 30 neurons
  - Hidden Layer 2: 20 neurons
  - Output Layer: 10 neurons (corresponding to the 10 digit classes)

### 3. Model Training
- MLP on the MNIST dataset.
- Random Forest (RF) 
- Logistic Regression models.

### 4. Performance Analysis
![confusion_matrix](confusion_matrix.png)

### 5. Observations
- Analyze common misclassifications between the digits based on the confusion matrix.

### 6. t-SNE Visualization
![t-SNE](t-SNE.png)

