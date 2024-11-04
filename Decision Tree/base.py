"""
You will be expected to use this to make trees for:
> discrete input, discrete output
> real input, real output
> real input, discrete output
> discrete input, real output
"""                
# Import 
from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from utils import *
import graphviz
import networkx as nx
from metrics import *

np.random.seed(42)

class Node :
    def __init__(self,feature):
        self.feature = feature
        self.children = {}

class Node_Real :
    def __init__(self,feature,threshold):
        self.feature = feature
        self.threshold = threshold
        self.children = {'Less than' : None,'Greater than' : None}   


@dataclass
class DecisionTree:
    criterion: Literal["information_gain", "gini_index"]  # criterion won't be used for regression
    max_depth: int  # The maximum depth the tree can grow to

    def __init__(self, criterion, max_depth=5):
        self.criterion = criterion
        self.max_depth = max_depth
        self.root = None

    def fit(self, X: pd.DataFrame, y: pd.Series, iter_ = 0) -> None:
        """
        Function to train and construct the decision tree
        """
        
        if iter_ == 0 and check_ifreal(X.iloc[:,0]) == 0 and check_ifreal(y) == 0 : self.type = 1
        if iter_ == 0 and check_ifreal(X.iloc[:,0]) == 0 and check_ifreal(y) == 1 : self.type = 2
        if iter_ == 0 and check_ifreal(X.iloc[:,0]) == 1 and check_ifreal(y) == 0 : self.type = 3
        if iter_ == 0 and check_ifreal(X.iloc[:,0]) == 1 and check_ifreal(y) == 1 : self.type = 4
            
        # Discrete to Discrete
        if self.type == 1 :
            if X.shape[1] == 0 or iter_ == self.max_depth or len(y.unique()) == 1: 
                return y.value_counts().idxmax()

            L = []

            for i in X.columns:
                M = []
                for j in X[i].unique():
                    M.append(X.index[X[i] == j]) 
                
                L.append(criteria_(self.criterion,y,M))

            root_column = X.columns[np.argmax(L)]
            A = Node(root_column)

            for i in X[root_column].unique():
                indices = X.index[X[root_column] == i]
                A.children[i] = self.fit(X.loc[indices].drop(columns=[root_column]),y.loc[indices],iter_ + 1)
            A.children['Else'] = y.value_counts().idxmax()
            if iter_ == 0 : self.root = A 
            return A
        
        # Discrete to Real
        if self.type == 2:
            if X.shape[1] == 0 or iter_ == self.max_depth or len(y.unique()) == 1: 
                return y.mean()

            L = []

            for i in X.columns:
                M = []
                for j in X[i].unique():
                    M.append(X.index[X[i] == j]) 

                L.append(reduction_mse(y,M))

            root_column = X.columns[np.argmax(L)]
            A = Node(root_column)

            for i in X[root_column].unique():
                indices = X.index[X[root_column] == i]
                A.children[i] = self.fit(X.loc[indices].drop(columns=[root_column]),y.loc[indices],iter_ + 1)
            A.children['Else'] = y.value_counts().mean()
            if iter_ == 0 : self.root = A 
            return A  
        
        # Real to Discrete
        if self.type == 3:    
            if iter_ == self.max_depth or len(y.unique()) == 1: return y.value_counts().idxmax()

            weighted_info_gain = - np.inf
            root_feature = None
            split_index = None 

            for i in X.columns:
                A_x = X.sort_values(by = i)
                A_y = y.iloc[X[i].argsort()]
                B = pd.Series([(A_x[i].iloc[j] + A_x[i].iloc[j - 1]) / 2 for j in range(1, len(A_x))])

                for j in range(len(B)):
                    df1_x = A_x[A_x[i] > B[j]]
                    df1_y = A_y[A_x[i] > B[j]]

                    df2_x = A_x[A_x[i] < B[j]]
                    df2_y = A_y[A_x[i] < B[j]]

                    current_info_gain = criteria_(self.criterion,y,[df1_y.index,df2_y.index])
                    if current_info_gain >= weighted_info_gain:
                        weighted_info_gain = current_info_gain
                        root_feature = i
                        split_index = B[j]

            A_x = X.sort_values(by = root_feature)
            A_y = y.iloc[X[root_feature].argsort()]

            df1_x = A_x[A_x[root_feature] > split_index]
            df1_y = A_y[A_x[root_feature] > split_index]

            df2_x = A_x[A_x[root_feature] < split_index]
            df2_y = A_y[A_x[root_feature] < split_index]

            Root_Node = Node_Real(root_feature,split_index)
            Root_Node.children['Less than'] = self.fit(df2_x,df2_y,iter_ + 1)
            Root_Node.children['Greater than'] = self.fit(df1_x,df1_y,iter_ + 1)
            if iter_ == 0 : self.root = Root_Node
            return Root_Node 
        
        # Real to Real
        if self.type == 4:  
            if iter_ == self.max_depth + 1 or len(y.unique()) == 1: return y.mean()

            weighted_entropy = np.inf
            root_feature = None
            split_index = None 

            for i in X.columns:
                A_x = X.sort_values(by = i)
                A_y = y.iloc[X[i].argsort()]
                B = pd.Series([(A_x[i].iloc[j] + A_x[i].iloc[j - 1]) / 2 for j in range(1, len(A_x))])

                for j in range(len(B)):
                    df1_x = A_x[A_x[i] > B[j]]
                    df1_y = A_y[A_x[i] > B[j]]

                    df2_x = A_x[A_x[i] < B[j]]
                    df2_y = A_y[A_x[i] < B[j]]

                    current_entropy = (len(df1_x) * mse(df1_y) + len(df2_x) * mse(df2_y)) / (len(A_x))
                    if current_entropy < weighted_entropy : 
                        weighted_entropy = current_entropy
                        root_feature = i
                        split_index = B[j]

            A_x = X.sort_values(by = root_feature)
            A_y = y.iloc[X[root_feature].argsort()]

            df1_x = A_x[A_x[root_feature] > split_index]
            df1_y = A_y[A_x[root_feature] > split_index]

            df2_x = A_x[A_x[root_feature] < split_index]
            df2_y = A_y[A_x[root_feature] < split_index]

            Root_Node = Node_Real(root_feature,split_index)
            Root_Node.children['Less than'] = self.fit(df2_x,df2_y,iter_ + 1)
            Root_Node.children['Greater than'] = self.fit(df1_x,df1_y,iter_ + 1)
            if iter_ == 0 : self.root = Root_Node
            return Root_Node  
        
    def predict(self, X: pd.DataFrame) -> pd.Series:
        """
        Funtion to run the decision tree on test inputs
        """

        # Traverse the tree you constructed to return the predicted values for the given test inputs.

        if self.type in [1,2]:
            series = []
            for x in X.index:
                A = self.root
                while isinstance(A,Node):
                    if X[A.feature].loc[x] in A.children.keys():
                        A = A.children[X[A.feature].loc[x]]
                    else:
                        A = A.children['Else']
                series.append(A)
            return pd.Series(series,index = X.index,name = 'Prediction')
        
        # Real to Discrete / Real
        if self.type in [3,4]:
            series = [] 
            for x in X.index:
                A = self.root
                while isinstance(A,Node_Real):
                    if X.loc[x][A.feature] < A.threshold:
                        A = A.children['Less than']
                    else : 
                        A = A.children['Greater than']
                series.append(A)
            return pd.Series(series,index = X.index,name = 'Prediction')   
        
    def plot(self, A, indentation = 0) -> None:
        """
        Function to plot the tree

        Output Example:
        ?(X1 > 4)
            Y: ?(X2 > 7)
                Y: Class A
                N: Class B
            N: Class C
        Where Y => Yes and N => No
        """
        if self.type in [1,2]:
            if not isinstance(A,Node):
                print(f'Class {A}')
                return 

            print(f'?({A.feature})')

            for i in A.children.keys():
                if i == 'Else': continue 
                s = len(str(A.feature))
                print((indentation + 2) * ' ' + f'{i}',end = ' : ')
                self.plot(A.children[i],indentation + s + 1)
                
        if self.type in [3,4]:
            if not isinstance(A,Node_Real):
                print(f'Class {A}')
                return 
            s = len(str(A.feature)) + 2
            
            print(f'?({A.feature} > {A.threshold})')
            print((indentation + 2) * ' ' + 'Y : ',end = '')
            self.plot(A.children['Greater than'],indentation + s + 1)
            print((indentation + 2) * ' ' + 'N : ',end = '')
            self.plot(A.children['Less than'],indentation + s + 1)