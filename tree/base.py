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
from tree.utils import entropy, information_gain, gini_index

import matplotlib.pyplot as plt

np.random.seed(42)




@dataclass
class Node:
    atrribute:str
    isLeaf:bool
    children: dict
    output: str
    # split_value: 0.0 #Represents which value to compare sample[attribute]

@dataclass
class DecisionTree:
    # criterion: Literal["information_gain", "gini_index"]  # criterion won't be used for regression
    max_depth: int  # The maximum depth the tree can grow to
    root = Node('hello', True, {}, 'hello')

    def get_attributes_X(self, X: pd.DataFrame) -> list:
        return X.columns.tolist()

    def get_split_attr_value(self, X: pd.DataFrame, y: pd.DataFrame, attr) -> dict:
        rtr = {'max_gain_attr' : "",
               'max_gain' : float('-inf'),
               'split_value' : 0}

        for a in attr:
            x_a_df = X[a]
            x_a_df_sorted = x_a_df.sort_values()
            x_a_series_sorted = pd.Series(x_a_df_sorted)

            #entropy of x_a_series_sorted
            entropy_x_a_series_sorted = entropy(x_a_series_sorted)

            for i in range(0, len(x_a_series_sorted) - 1):
                split_value_i = (x_a_series_sorted.iloc[i] + x_a_series_sorted.iloc[i+1])/2
                top = y.iloc[:i+1]
                bottom = y.iloc[i+1:]

                #entropy of top
                entropy_top = entropy(pd.Series(top['y']))
                #entropy of bottom
                entropy_bottom = entropy(pd.Series(bottom['y']))

                info_gain = entropy_x_a_series_sorted - (len(top)/len(x_a_series_sorted))*entropy_top - (len(bottom)/len(x_a_series_sorted))*entropy_bottom

                if (rtr['max_gain'] < info_gain):
                    rtr['max_gain_attr'] = a
                    rtr['max_gain'] = info_gain
                    rtr['split_value'] = split_value_i

        return rtr

    def construct_tree_real_discrete(self, X, y, attr, cur_depth):
        #notes
        #1. use the above function for finding the split
        #2. split and recursion for each child

        # if (self.max_depth == cur_depth):
        #     output = y.value_counts().idxmax()
        #     n1 = Node(atrribute=attr, isLeaf=True, children={}, output=output)
        #     return n1
        # if (len(attr) == 0):
        #     output = y.value_counts().idxmax()
        #     n1 = Node(atrribute=None, isLeaf=True, output=output, children={})
        #     return n1
        #
        # for a in attr:
        #     g = information_gain(y, pd.Series(X[a]))
        #     if (max_gain < g):
        #         max_gain = g
        #         max_gain_attr = a
        #
        # # print()
        # # print('max_gain', max_gain)
        # children_name = X[max_gain_attr].unique()
        #
        # # print('maximum gain',max_gain_attr)
        # children = {}
        # for c in children_name:
        #     index = X.groupby([max_gain_attr]).groups[c].tolist()
        #     df_y = pd.DataFrame({'Y': y.values})
        #     y_mod = df_y.iloc[index]
        #     # print('index',index)
        #     temp = attr.copy()
        #     temp.remove(max_gain_attr)
        #     children[c] = self.construct_tree(X.groupby([max_gain_attr]).get_group(c), pd.Series(df_y['Y']), temp,
        #                                       cur_depth + 1)
        #
        # n1 = Node(atrribute=max_gain_attr, isLeaf=False, children=children, output=None)
        # return n1

    def construct_tree(self, X, y, attr, cur_depth):
        max_gain_attr = ""
        max_gain = 0


        if(entropy(y)==0):
            output = y.value_counts().idxmax()


            n1 = Node(atrribute=attr, isLeaf=True, children={}, output=output)
            return n1
        if (self.max_depth == cur_depth):
            output = y.value_counts().idxmax()
            n1 = Node(atrribute=attr, isLeaf=True, children={}, output=output)
            print(X)
            return n1
        if (len(attr) == 0):
            output = y.value_counts().idxmax()
            n1 = Node(atrribute=None, isLeaf=True, output=output, children={})
            print(X)
            return n1

        for a in attr:
            g = information_gain(y, pd.Series(X[a]))
            if (max_gain < g):
                max_gain = g
                max_gain_attr = a

        print()
        # print('max_gain', max_gain)
        children_name = X[max_gain_attr].unique()

        # print('maximum gain',max_gain_attr)
        children = {}
        for c in children_name:
            index = X.groupby([max_gain_attr]).groups[c].tolist()
            # print(index)
            df_y = pd.DataFrame({'Y': y.values})
            # y_mod = df_y.iloc[index]




            temp = attr.copy()
            temp.remove(max_gain_attr)
            children[c] = self.construct_tree(X.groupby([max_gain_attr]).get_group(c), y, temp, cur_depth + 1)

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

    def print_tree(self, node):
        if (node.isLeaf):
            print(node.output)
            return

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
    print(tree1.root.children["Overcast"])

def test_decision_tree_real_discrete():
    x1 = np.random.uniform(0, 10, 100)
    x2 = np.random.uniform(0, 10, 100)
    y = np.random.choice(['red', 'blue', 'green'], 100)
    x = pd.DataFrame({'x1': x1, 'x2': x2})
    y = pd.DataFrame({'y': y})
    print(x)
    print(y)

    tree2 = DecisionTree(max_depth=2)
    print(tree2.get_split_attr_value(x,y,['x1','x2']))
