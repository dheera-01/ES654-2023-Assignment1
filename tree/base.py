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
from tree.utils import entropy, information_gain, gini_index, variance_reduction, information_gain_gini

import matplotlib.pyplot as plt

np.random.seed(42)




@dataclass
class Node:
    atrribute:str
    isLeaf:bool
    children: dict
    output: str | int | float
    split_value: 0.0 #Represents which value to compare sample[attribute]

@dataclass
class DecisionTree:
    criterion: Literal["information_gain", "gini_index"]  # criterion won't be used for regression
    max_depth: int  # The maximum depth the tree can grow to
    root = Node('hello', True, {}, 'hello',split_value=0.0)
    tree_type = 'discrete_discrete'


    criteria_functions={
        "information_gain":information_gain,
        "gini_index":information_gain_gini
    }
    entropy_gini_dict={
        "information_gain":entropy,
        "gini_index":gini_index
    }
    def traverse_discrete_input(self, node: Node, data: dict):
        if (node.isLeaf == True):
            return node.output
        else:
            if (data[node.atrribute] in node.children):
                return self.traverse_discrete_input(node.children[data[node.atrribute]], data)
            else:
                return "Error"

    def traverse_real_input(self, node: Node, data: dict):
        if (node.isLeaf == True):
            return node.output
        else:
            if (data[node.atrribute] <= node.split_value):
                return self.traverse_real_input(node.children['less_than_split_value'], data)
            else:
                return self.traverse_real_input(node.children['greater_than_split_value'], data)

    def get_split_attr_value(self, X: pd.DataFrame, y: pd.DataFrame, attr: list) -> dict:
        rtr = {'max_gain_attr' : "",
               'max_gain' : float('-inf'),
               'split_value' : 0,
               'split_index': 0}

        if self.tree_type == 'real_discrete':
            rtr['max_gain'] = float('-inf')
        elif self.tree_type == 'real_real':
            rtr['max_gain'] = float('inf')

        combined_df = pd.concat([X, y], axis=1)

        for a in attr:
            combined_df = combined_df.sort_values(by=[a])
            # print('Combined_df',combined_df)
            x_a_series_sorted = pd.Series(combined_df[a])

            if self.tree_type == 'real_discrete':
                # entropy of x_a_series_sorted
                entropy_x_a_series_sorted = self.entropy_gini_dict[self.criterion](x_a_series_sorted)

            for i in range(0, len(x_a_series_sorted) - 1):
                split_value_i = (x_a_series_sorted.iloc[i] + x_a_series_sorted.iloc[i+1])/2

                combined_df = combined_df.reset_index(drop=True)
                top = combined_df.iloc[:i + 1]

                bottom = combined_df.iloc[i + 1:]

                if self.tree_type == 'real_discrete':
                    #entropy of top
                    entropy_top = self.entropy_gini_dict[self.criterion](pd.Series(top['y']))
                    #entropy of bottom
                    entropy_bottom = self.entropy_gini_dict[self.criterion](pd.Series(bottom['y']))
                    info_gain = entropy_x_a_series_sorted - (len(top)/len(x_a_series_sorted))*entropy_top - (len(bottom)/len(x_a_series_sorted))*entropy_bottom
                    if (rtr['max_gain'] < info_gain):
                        rtr['max_gain_attr'] = a
                        rtr['max_gain'] = info_gain
                        rtr['split_value'] = split_value_i
                        rtr['split_index'] = i
                elif self.tree_type == 'real_real':
                    y_hat_top = pd.Series([top['y'].mean()] * len(top))
                    y_hat_bottom = pd.Series([bottom['y'].mean()] * len(bottom))
                    y_hat = pd.concat([y_hat_top, y_hat_bottom], axis=0)
                    y_hat = y_hat.reset_index(drop=True)
                    y_series = pd.Series(combined_df['y'])
                    loss = np.power(y_hat - y_series, 2).sum()
                    if (rtr['max_gain'] >= loss):
                        rtr['max_gain_attr'] = a
                        rtr['max_gain'] = loss
                        rtr['split_value'] = split_value_i
                        rtr['split_index'] = i
        return rtr

    def construct_tree_real_input(self, X, y, attr, cur_depth) -> Node:
        if self.tree_type == 'real_discrete':
            if (self.entropy_gini_dict[self.criterion](y) == 0.0):
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
        elif self.tree_type == 'real_real':
            if y.nunique() == 1:
                output = y.mean(axis=0)
                n1 = Node(atrribute=attr, isLeaf=True, children={}, output=output, split_value=0.0)
                return n1

            if (self.max_depth == cur_depth):
                output = y.mean(axis=0)
                n1 = Node(atrribute=attr, isLeaf=True, children={}, output=output, split_value=0.0)
                return n1

            if (len(attr) == 0):
                output = y.value_counts().idxmax()
                n1 = Node(atrribute=None, isLeaf=True, output=output, children={}, split_value=0.0)
                return n1

        df_y = pd.DataFrame({'y': y.values})
        max_gain_dict = self.get_split_attr_value(X, df_y, attr)

        combined_df = pd.concat([X, df_y], axis=1)
        combined_df = combined_df.sort_values(by=[max_gain_dict['max_gain_attr']])
        combined_df = combined_df.reset_index(drop=True)

        top = combined_df.iloc[:max_gain_dict['split_index'] + 1]
        top_reset_index = top.reset_index(drop=True)
        top_reset_index_y = top_reset_index['y']
        top_reset_index = top_reset_index.drop(['y'], axis=1)

        bottom = combined_df.iloc[max_gain_dict['split_index'] + 1:]
        bottom_reset_index = bottom.reset_index(drop=True)
        bottom_reset_index_y = bottom_reset_index['y']
        bottom_reset_index = bottom_reset_index.drop(['y'], axis=1)

        node_less_than_split_value = self.construct_tree_real_input(top_reset_index, top_reset_index_y, attr, cur_depth + 1)
        node_greater_than_split_value = self.construct_tree_real_input(bottom_reset_index, bottom_reset_index_y, attr, cur_depth + 1)

        children = {}
        children['less_than_split_value'] = node_less_than_split_value
        children['greater_than_split_value'] = node_greater_than_split_value

        n1 = Node(atrribute=max_gain_dict['max_gain_attr'], isLeaf=False, children=children, output=None, split_value=max_gain_dict['split_value'])
        return n1

    def construct_tree_discrete_input(self, X, y, attr, cur_depth):
        max_var_gain_attr = ""
        max_var_gain = float('-inf')

        if self.tree_type == 'discrete_discrete':
            if (self.entropy_gini_dict[self.criterion](y) == 0.0):
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
        elif self.tree_type == 'discrete_real':
            if (len(y) < 2 or y.std() == 0.0):
                output = y.mean()
                n1 = Node(atrribute=attr, isLeaf=True, children={}, output=output, split_value=0.0)
                return n1

            if (self.max_depth == cur_depth):
                output = y.mean()
                n1 = Node(atrribute=attr, isLeaf=True, children={}, output=output, split_value=0.0)
                return n1

            if (len(attr) == 0):
                output = y.mean()
                n1 = Node(atrribute=None, isLeaf=True, output=output, children={}, split_value=0.0)
                return n1

        for a in attr:
            if self.tree_type == 'discrete_discrete':
                var = self.criteria_functions[self.criterion](y, pd.Series(X[a]))
            elif self.tree_type == 'discrete_real':
                var = variance_reduction(y, pd.Series(X[a]))

            if (max_var_gain < var):
                max_var_gain = var
                max_var_gain_attr = a

        children_name = X[max_var_gain_attr].unique()
        children = {}

        for c in children_name:
            index = X.groupby([max_var_gain_attr]).groups[c].tolist()

            df_y = pd.DataFrame({'y': y.values})
            y_mod = df_y.iloc[index]
            y_mod = y_mod.reset_index(drop=True)
            y_mod = pd.Series(y_mod['y'])

            x_mod = X.groupby([max_var_gain_attr]).get_group(c).reset_index(drop=True)

            temp = attr.copy()
            temp.remove(max_var_gain_attr)
            children[c] = self.construct_tree_discrete_input(x_mod, y_mod, temp, cur_depth + 1)

        n1 = Node(atrribute=max_var_gain_attr, isLeaf=False, children=children, output=None, split_value=0.0)
        return n1

    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        """
        Function to train and construct the decision tree
        """
        # # setting the type of tree in the class
        # if (X.iloc[:,0].dtype == 'int64' or X.iloc[:,0].dtype == 'float64') and (y.dtype == 'int64' or y.dtype == 'float64') :
        #     self.tree_type = 'real_real'
        # elif (X.iloc[:,0].dtype == 'int64' or X.iloc[:,0].dtype == 'float64') and (y.dtype != 'int64' and y.dtype != 'float64'):
        #     self.tree_type = 'real_discrete'
        # elif (X.iloc[:,0].dtype != 'int64' and X.iloc[:,0].dtype != 'float64') and (y.dtype == 'int64' or y.dtype == 'float64'):
        #     self.tree_type = 'discrete_real'
        # else:
        #     self.tree_type = 'discrete_discrete'

        if X.iloc[:,0].dtype == 'category' and y.dtype == 'category':
            self.tree_type = 'discrete_discrete'
        elif X.iloc[:,0].dtype == 'category' and y.dtype != 'category':
            self.tree_type = 'discrete_real'
        elif X.iloc[:,0].dtype != 'category' and y.dtype == 'category':
            self.tree_type = 'real_discrete'
        else:
            self.tree_type = 'real_real'

        # calling the construct tree functions according to the type of tree
        x_columns_lst = X.columns.tolist()
        # x_columns_lst = [str(col) for col in x_columns_lst]
        # X.columns = x_columns_lst
        if self.tree_type == 'real_real':
            self.root = self.construct_tree_real_input(X, y, x_columns_lst, 0)
        elif self.tree_type == 'real_discrete':
            self.root = self.construct_tree_real_input(X, y, x_columns_lst, 0)
        elif self.tree_type == 'discrete_real':
            self.root = self.construct_tree_discrete_input(X, y, x_columns_lst, 0)
        else:
            self.root = self.construct_tree_discrete_input(X, y, x_columns_lst, 0)

    def predict(self, X: pd.DataFrame) -> pd.Series:
        """
        Funtion to run the decision tree on test inputs
        """
        if self.tree_type == 'real_real' or self.tree_type == 'discrete_real':
            y_pred = pd.Series(dtype='float64')
        else:
            y_pred = pd.Series(dtype='category')
        for i in range(len(X)):
            if self.tree_type == 'real_real' or self.tree_type == 'real_discrete':
                y_pred._set_value(i, self.traverse_real_input(self.root, X.iloc[i]))
            else:
                y_pred._set_value(i, self.traverse_discrete_input(self.root, X.iloc[i]))
        return y_pred

    def print_tree_discrete_input(self, node, indentation_value, prefix):
        if node.isLeaf == True:
            print('    '*indentation_value,prefix,'Value: ', node.output)
            return

        for key, value in node.children.items():
            print('    ' * indentation_value, prefix, '?', '(', node.atrribute, ' ', '=', ' ', key, ')')
            self.print_tree_discrete_input(node.children[key], indentation_value + 1, 'Y: ')

    def print_tree_real_input(self, node, indentation_value, prefix):
        if node.isLeaf == True:
            print('    '*indentation_value,prefix,'Value: ',node.output)
            return
        print('    '*indentation_value,prefix,'?','(',node.atrribute,' ','<=',' ',node.split_value,')')
        self.print_tree_real_input(node.children['less_than_split_value'], indentation_value+1,'Y: ')
        self.print_tree_real_input(node.children['greater_than_split_value'], indentation_value+1,'N: ')

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
        if self.tree_type == 'real_real' or self.tree_type == 'real_discrete':
            self.print_tree_real_input(self.root, 0, '')
        else:
            self.print_tree_discrete_input(self.root, 0, '')