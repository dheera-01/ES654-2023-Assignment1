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


def gain()->str:
    pass

def construct_tree(X,y,attr,max_depth,cur_depth):
    max_gain_attr=""
    max_gain=0

    if(max_depth==cur_depth):
        output=y.value_counts().idxmax()
        n1=Node(atrribute=attr,isLeaf=True,children={},output=output)
    if(len(attr)==0):
        output=y.value_counts().idxmax()
        n1=Node(atrribute=None,isLeaf=True,output=output,children={})

    for a in attr:
        
        g=information_gain(y,attr)
        if(max_gain<g):
            max_gain=g
            max_gain_attr=a
    
    children_name=X[max_gain_attr].unique()
    children={}
    for c in children_name:
       
        index = X.groupby([max_gain_attr]).groups[c].tolist()
        y_mod=y.iloc[index]
        children[c]=construct_tree(X.groupby([max_gain_attr]).get_group(c),y_mod,attr.drop(labels=[a]),max_depth,cur_depth+1)
    n1=Node(atrribute=max_gain_attr,isLeaf=False,children=children,output=None)
    return n1

@dataclass
class DecisionTree:
    criterion: Literal["information_gain", "gini_index"]  # criterion won't be used for regression
    max_depth: int  # The maximum depth the tree can grow to

    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        """
        Function to train and construct the decision tree

        """
        attributes=["outlook","humidity","rain"]
        construct_tree(X,y,attributes,max_depth=self.max_depth,cur_depth=0);
        

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
