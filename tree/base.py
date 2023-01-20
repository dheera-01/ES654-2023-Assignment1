"""
The current code given is for the Assignment 1.
You will be expected to use this to make trees for:
> discrete input, discrete output
> real input, real output
> real input, discrete output
> discrete input, real output
"""
from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tree.utils import entropy, information_gain, gini_index

np.random.seed(42)

@dataclass
class Node:
    atrribute:str
    isLeaf:bool
    children: dict
    output: str

@dataclass
class DecisionTree:
    # criterion: Literal["information_gain", "gini_index"]  # criterion won't be used for regression
    max_depth: int  # The maximum depth the tree can grow to
    root = Node('hello', True, {}, 'hello')

    def get_attributes_X(self, X: pd.DataFrame) -> list:
        return X.columns.tolist()

    def construct_tree(self, X, y, attr, cur_depth):
        max_gain_attr = ""
        max_gain = 0
        print(attr)
        if (self.max_depth == cur_depth):
            output = y.value_counts().idxmax()
            n1 = Node(atrribute=attr, isLeaf=True, children={}, output=output)
            return n1
        if (len(attr) == 0):
            output = y.value_counts().idxmax()
            n1 = Node(atrribute=None, isLeaf=True, output=output, children={})
            return n1

        for a in attr:
            g = information_gain(y, pd.Series(X[a]))
            if (max_gain < g):
                max_gain = g
                max_gain_attr = a

        # print()
        # print('max_gain', max_gain)
        children_name = X[max_gain_attr].unique()

        # print('maximum gain',max_gain_attr)
        children = {}
        for c in children_name:
            index = X.groupby([max_gain_attr]).groups[c].tolist()
            df_y = pd.DataFrame({'Y': y.values})
            y_mod = df_y.iloc[index]
            # print('index',index)
            temp = attr.copy()
            temp.remove(max_gain_attr)
            children[c] = self.construct_tree(X.groupby([max_gain_attr]).get_group(c), pd.Series(df_y['Y']), temp, cur_depth + 1)

        n1 = Node(atrribute=max_gain_attr, isLeaf=False, children=children, output=None)
        return n1

    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        """
        Function to train and construct the decision tree

        """
        
        # attributes=["outlook","humidity","rain"]
        attributes = self.get_attributes_X(X)
        self.root = self.construct_tree(X,y,attributes,-1);

    def predict(self, X: pd.DataFrame) -> pd.Series:
        """
        Funtion to run the decision tree on test inputs
        """
        pass

    def plot(self) -> None:
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
        pass

def test_decision_tree():
    """
    Function to test the decision tree
    """
    x = pd.DataFrame({
        'Outlook': ['Sunny', 'Sunny', 'Overcast', 'Rain', 'Rain', 'Rain', 'Overcast', 'Sunny', 'Sunny', 'Rain', 'Sunny',
                    'Overcast', 'Overcast', 'Rain'],
        'Temperature': ['Hot', 'Hot', 'Hot', 'Mild', 'Cool', 'Cool', 'Cool', 'Mild', 'Cool', 'Mild', 'Mild', 'Mild',
                        'Hot', 'Mild'],
        'Humidity': ['High', 'High', 'High', 'High', 'Normal', 'Normal', 'Normal', 'High', 'Normal', 'Normal', 'Normal',
                     'High', 'Normal', 'High'],
        'Wind': ['Weak', 'Strong', 'Weak', 'Weak', 'Weak', 'Strong', 'Strong', 'Weak', 'Weak', 'Weak', 'Strong',
                 'Strong', 'Weak', 'Strong'],
        })

    y = pd.DataFrame({
        'PlayTennis': ['No', 'No', 'Yes', 'Yes', 'Yes', 'No', 'Yes', 'No', 'Yes', 'Yes', 'Yes', 'Yes', 'Yes', 'No']
    })
    # print(x)

    # index = x.groupby(["Outlook"]).groups['Overcast'].tolist()
    # print(index)
    # new_x = y.iloc[index]
    # print(new_x)


    # tree1 = DecisionTree(max_depth=2)
    # print(tree1.get_attributes_X(x));
    # print(y.groupby(['PlayTennis']).groups)

    # print('entropy: ',entropy(y["PlayTennis"]))
    # print('Gini: ', gini_index(y["PlayTennis"]))
    # print(information_gain(pd.Series(x['Outlook']), pd.Series(y['PlayTennis'])))

    tree1 = DecisionTree(max_depth=10)
    tree1.fit(x,pd.Series(y['PlayTennis']))
    print(tree1.root)

