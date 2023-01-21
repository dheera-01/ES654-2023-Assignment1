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
    split_value: 0.0 #Represents which value to compare sample[attribute]

@dataclass
class DecisionTree:
    # criterion: Literal["information_gain", "gini_index"]  # criterion won't be used for regression
    max_depth: int  # The maximum depth the tree can grow to
    root = Node('hello', True, {}, 'hello',split_value=0.0)

    def get_attributes_X(self, X: pd.DataFrame) -> list:
        return X.columns.tolist()

    def get_split_attr_value(self, X: pd.DataFrame, y: pd.DataFrame, attr) -> dict:
        rtr = {'max_gain_attr' : "",
               'max_gain' : float('-inf'),
               'split_value' : 0,
               'split_index': 0}

        combined_df = pd.concat([X, y], axis=1)

        for a in attr:
            combined_df = combined_df.sort_values(by=[a])
            x_a_series_sorted = pd.Series(combined_df[a])

            #entropy of x_a_series_sorted
            entropy_x_a_series_sorted = entropy(x_a_series_sorted)

            for i in range(0, len(x_a_series_sorted) - 1):
                split_value_i = (x_a_series_sorted.iloc[i] + x_a_series_sorted.iloc[i+1])/2

                #entropy of top
                combined_df = combined_df.reset_index(drop=True)
                top = combined_df.iloc[:i + 1]
                entropy_top = entropy(pd.Series(top['y']))

                #entropy of bottom
                bottom = combined_df.iloc[i + 1:]
                entropy_bottom = entropy(pd.Series(bottom['y']))

                info_gain = entropy_x_a_series_sorted - (len(top)/len(x_a_series_sorted))*entropy_top - (len(bottom)/len(x_a_series_sorted))*entropy_bottom

                if (rtr['max_gain'] < info_gain):
                    rtr['max_gain_attr'] = a
                    rtr['max_gain'] = info_gain
                    rtr['split_value'] = split_value_i
                    rtr['split_index'] = i
        # print(rtr)
        return rtr

    def construct_tree_real_discrete(self, X, y, attr, cur_depth):
        if (entropy(y) == 0.0):
            output = y.value_counts().idxmax()
            n1 = Node(atrribute=attr, isLeaf=True, children={}, output=output, split_value=0.0)
            return n1

        if (self.max_depth == cur_depth):
            output = y.value_counts().idxmax()
            n1 = Node(atrribute=attr, isLeaf=True, children={}, output=output, split_value=0.0)
            return n1

        if (len(attr) == 0):
            output = y.value_counts().idxmax()
            n1 = Node(atrribute=None, isLeaf=True, output=output, children={}, split_value=0.0)
            return n1

        df_y = pd.DataFrame({'y': y.values})
        max_gain_dict = self.get_split_attr_value(X, df_y, attr)

        combined_df = pd.concat([X, y], axis=1)
        combined_df = combined_df.sort_values(by=[max_gain_dict['max_gain_attr']])
        combined_df = combined_df.reset_index(drop=True)
        top = combined_df.iloc[:max_gain_dict['split_index'] + 1]
        top_reset_index = top.reset_index(drop=True)

        bottom = combined_df.iloc[max_gain_dict['split_index'] + 1:]
        bottom_reset_index = bottom.reset_index(drop=True)

        node_less_than_split_value = self.construct_tree_real_discrete(top_reset_index.drop(columns=['y']), top_reset_index['y'], attr, cur_depth + 1)
        node_greater_than_split_value = self.construct_tree_real_discrete(bottom_reset_index.drop(columns=['y']), bottom_reset_index['y'], attr, cur_depth + 1)

        children = {}
        children['less_than_split_value'] = node_less_than_split_value
        children['greater_than_split_value'] = node_greater_than_split_value

        n1 = Node(atrribute=max_gain_dict['max_gain_attr'], isLeaf=False, children=children, output=None, split_value=max_gain_dict['split_value'])
        return n1

    def construct_tree(self, X, y, attr, cur_depth):
        print()
        print(attr)
        max_gain_attr = ""
        max_gain = float('-inf')

        if(entropy(y)==0.0):
            output = y.value_counts().idxmax()
            n1 = Node(atrribute=attr, isLeaf=True, children={}, output=output, split_value=0.0)
            return n1

        if (self.max_depth == cur_depth):
            output = y.value_counts().idxmax()
            n1 = Node(atrribute=attr, isLeaf=True, children={}, output=output, split_value=0.0)
            print(X)
            return n1

        if (len(attr) == 0):
            output = y.value_counts().idxmax()
            n1 = Node(atrribute=None, isLeaf=True, output=output, children={}, split_value=0.0)
            print(X)
            return n1

        for a in attr:
            g = information_gain(y, pd.Series(X[a]))
            if (max_gain < g):
                max_gain = g
                max_gain_attr = a

        print('max_gain_attr', max_gain_attr)

        children_name = X[max_gain_attr].unique()
        children = {}

        for c in children_name:
            index = X.groupby([max_gain_attr]).groups[c].tolist()

            df_y = pd.DataFrame({'Y':y.values})
            y_mod = df_y.iloc[index]
            y_mod = y_mod.reset_index(drop=True)
            y_mod = pd.Series(y_mod['Y'])

            x_mod = X.groupby([max_gain_attr]).get_group(c).reset_index(drop=True)

            temp = attr.copy()
            temp.remove(max_gain_attr)
            children[c] = self.construct_tree(x_mod, y_mod, temp, cur_depth + 1)

        n1 = Node(atrribute=max_gain_attr, isLeaf=False, children=children, output=None, split_value=0.0)
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

    def print_tree_discrete_discrete(self, node, indentation_value, prefix):
        if node.isLeaf == True:
            print('    '*indentation_value,prefix,'Value: ', node.output)
            return

        for key, value in node.children.items():
            print('    ' * indentation_value, prefix, '?', '(', node.atrribute, ' ', '=', ' ', key, ')')
            self.print_tree_discrete_discrete(node.children[key], indentation_value + 1, 'Y: ')

    def print_tree_real_discrete(self, node, indentation_value, prefix):
        if node.isLeaf == True:
            print('    '*indentation_value,prefix,'Value: ',node.output)
            return
        print('    '*indentation_value,prefix,'?','(',node.atrribute,' ','<=',' ',node.split_value,')')
        self.print_tree_real_discrete(node.children['less_than_split_value'], indentation_value+1,'Y: ')
        self.print_tree_real_discrete(node.children['greater_than_split_value'], indentation_value+1,'N: ')


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

    tree1 = DecisionTree(max_depth=10)
    tree1.fit(x, pd.Series(y['PlayTennis']))
    print(tree1.root)
    tree1.print_tree_discrete_discrete(tree1.root, 0, '')

def test_decision_tree_real_discrete():
    x1 = np.random.uniform(0, 10, 100)
    x2 = np.random.uniform(0, 10, 100)
    y = np.random.choice(['red', 'blue', 'green'], 100)
    x = pd.DataFrame({'x1': x1, 'x2': x2})
    y = pd.DataFrame({'y': y})
    # print(x)
    # print(y)

    tree2 = DecisionTree(max_depth=10)
    # print(tree2.get_split_attr_value(x, y, ['x1','x2']))
    tree2.root = tree2.construct_tree_real_discrete(x, pd.Series(y['y']), ['x1','x2'],0)
    tree2.print_tree_real_discrete(tree2.root, 0, '')

def test_get_split_tree_real_discrete():
    x1 = np.array([1,2,3])
    y = np.array(['red', 'red', 'blue'])
    x = pd.DataFrame({'x1': x1})
    y = pd.DataFrame({'y': y})

    tree2 = DecisionTree(max_depth=2)
    # print(tree2.construct_tree_real_discrete(x, pd.Series(y['y']), ['x1'], cur_depth=10))
    tree2.root = tree2.construct_tree_real_discrete(x, pd.Series(y['y']), ['x1'], cur_depth=10)
    tree2.print_tree_real_discrete(tree2.root, 0, '')
    # print(tree2.construct_tree_real_discrete(x, pd.Series(y['y']), ['x1','x2'],0))

test_decision_tree()
# test_decision_tree_real_discrete()
# test_get_split_tree_real_discrete()