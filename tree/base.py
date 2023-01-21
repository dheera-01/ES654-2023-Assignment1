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
    # criterion: Literal["information_gain", "gini_index"]  # criterion won't be used for regression
    max_depth: int  # The maximum depth the tree can grow to
    root = Node('hello', True, {}, 'hello',split_value=0.0)

    def traverse_discrete_input(self, node: Node, data: dict):
        if (node.isLeaf == True):
            return node.output
        else:
            return self.traverse_discrete_input(node.children[data[node.atrribute]], data)

    def traverse_real_input(self, node: Node, data: dict):
        if (node.isLeaf == True):
            return node.output
        else:
            if (data[node.atrribute] <= node.split_value):
                return self.traverse_real_input(node.children['less_than_split_value'], data)
            else:
                return self.traverse_real_input(node.children['greater_than_split_value'], data)

    def get_attributes_X(self, X: pd.DataFrame) -> list:
        return X.columns.tolist()

    def get_split_attr_value(self, X: pd.DataFrame, y: pd.DataFrame, attr: list, case: str) -> dict:
        rtr = {'max_gain_attr' : "",
               'max_gain' : float('-inf'),
               'split_value' : 0,
               'split_index': 0}

        if case == 'real_discrete':
            rtr['max_gain'] = float('-inf')
        elif case == 'real_real':
            rtr['max_gain'] = float('inf')

        combined_df = pd.concat([X, y], axis=1)

        for a in attr:
            combined_df = combined_df.sort_values(by=[a])
            x_a_series_sorted = pd.Series(combined_df[a])

            if case == 'real_discrete':
                # entropy of x_a_series_sorted
                entropy_x_a_series_sorted = entropy(x_a_series_sorted)

            for i in range(0, len(x_a_series_sorted) - 1):
                split_value_i = (x_a_series_sorted.iloc[i] + x_a_series_sorted.iloc[i+1])/2

                combined_df = combined_df.reset_index(drop=True)
                top = combined_df.iloc[:i + 1]

                bottom = combined_df.iloc[i + 1:]

                if case == 'real_discrete':
                    #entropy of top
                    entropy_top = entropy(pd.Series(top['y']))
                    #entropy of bottom
                    entropy_bottom = entropy(pd.Series(bottom['y']))
                    info_gain = entropy_x_a_series_sorted - (len(top)/len(x_a_series_sorted))*entropy_top - (len(bottom)/len(x_a_series_sorted))*entropy_bottom
                    if (rtr['max_gain'] < info_gain):
                        rtr['max_gain_attr'] = a
                        rtr['max_gain'] = info_gain
                        rtr['split_value'] = split_value_i
                        rtr['split_index'] = i
                elif case == 'real_real':
                    #here calculate the loss using mse in metrics.py
                    #print('calculate loss')
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
        # print(rtr)
        return rtr

    def construct_tree_real_discrete(self, X, y, attr, cur_depth, case: str) -> Node:
        if case == 'real_discrete':
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
        elif case == 'real_real':
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
        max_gain_dict = self.get_split_attr_value(X, df_y, attr,case)

        combined_df = pd.concat([X, y], axis=1)
        combined_df = combined_df.sort_values(by=[max_gain_dict['max_gain_attr']])
        combined_df = combined_df.reset_index(drop=True)
        top = combined_df.iloc[:max_gain_dict['split_index'] + 1]
        top_reset_index = top.reset_index(drop=True)

        bottom = combined_df.iloc[max_gain_dict['split_index'] + 1:]
        bottom_reset_index = bottom.reset_index(drop=True)

        node_less_than_split_value = self.construct_tree_real_discrete(top_reset_index.drop(columns=['y']), top_reset_index['y'], attr, cur_depth + 1,case)
        node_greater_than_split_value = self.construct_tree_real_discrete(bottom_reset_index.drop(columns=['y']), bottom_reset_index['y'], attr, cur_depth + 1,case)

        children = {}
        children['less_than_split_value'] = node_less_than_split_value
        children['greater_than_split_value'] = node_greater_than_split_value

        n1 = Node(atrribute=max_gain_dict['max_gain_attr'], isLeaf=False, children=children, output=None, split_value=max_gain_dict['split_value'])
        return n1

    def construct_tree_real_real_cover(self, X, y, attr, cur_depth):
        print('hello')

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
            g = information_gain_gini(y, pd.Series(X[a]))
            print(g)
            if (max_gain < g):
                max_gain = g
                max_gain_attr = a

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

    def construct_tree_discrete_real(self, X, y, attr, cur_depth):

        max_var_red_attr = ""
        max_var_red = float('-inf')

        if (len(y)<2 or y.std() == 0.0):
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
            var = variance_reduction(y, pd.Series(X[a]))

            if (max_var_red < var):
                max_var_red = var
                max_var_red_attr = a

        children_name = X[max_var_red_attr].unique()

        children = {}

        for c in children_name:
            index = X.groupby([max_var_red_attr]).groups[c].tolist()

            df_y = pd.DataFrame({'Y': y.values})
            y_mod = df_y.iloc[index]
            y_mod = y_mod.reset_index(drop=True)
            y_mod = pd.Series(y_mod['Y'])

            x_mod = X.groupby([max_var_red_attr]).get_group(c).reset_index(drop=True)

            temp = attr.copy()
            temp.remove(max_var_red_attr)
            children[c] = self.construct_tree_discrete_real(x_mod, y_mod, temp, cur_depth + 1)

        n1 = Node(atrribute=max_var_red_attr, isLeaf=False, children=children, output=None, split_value=0.0)
        return n1

    def construct_tree_discrete_real(self, X, y, attr, cur_depth):

        max_var_red_attr = ""
        max_var_red = float('-inf')

        if (len(y)<2 or y.std() == 0.0):
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
            var = variance_reduction(y, pd.Series(X[a]))

            if (max_var_red < var):
                max_var_red = var
                max_var_red_attr = a

        children_name = X[max_var_red_attr].unique()

        children = {}

        for c in children_name:
            index = X.groupby([max_var_red_attr]).groups[c].tolist()

            df_y = pd.DataFrame({'Y': y.values})
            y_mod = df_y.iloc[index]
            y_mod = y_mod.reset_index(drop=True)
            y_mod = pd.Series(y_mod['Y'])

            x_mod = X.groupby([max_var_red_attr]).get_group(c).reset_index(drop=True)

            temp = attr.copy()
            temp.remove(max_var_red_attr)
            children[c] = self.construct_tree_discrete_real(x_mod, y_mod, temp, cur_depth + 1)

        n1 = Node(atrribute=max_var_red_attr, isLeaf=False, children=children, output=None, split_value=0.0)
        return n1

    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        """
        Function to train and construct the decision tree
        """
        
        # attributes=["outlook","humidity","rain"]
        attributes = self.get_attributes_X(X)
        # self.root = self.construct_tree(X,y,attributes,-1);
        self.root=self.construct_tree(X,y,attributes,0)


    # def predict_discrete_input(self, X: pd.DataFrame) -> pd.Series:
    #     y_pred = pd.Series()
    #     for i in range(len(X)):
    #         y_pred._set_value(i, self.traverse_discrete_input(self.root, X.iloc[i]))
    #     return y_pred

    def predict(self, X: pd.DataFrame) -> pd.Series:
        """
        Funtion to run the decision tree on test inputs
        """
        y_pred = pd.Series()
        for i in range(len(X)):
            #for real only now
            y_pred._set_value(i, self.traverse_real_input(self.root, X.iloc[i]))
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
    X_pred = pd.DataFrame({"Outlook": ['Sunny', 'Rain'], "Temperature": ['Hot', 'Mild'], "Humidity": ['High', 'Normal'],
                           "Wind": ['Strong', 'Weak']})
    y_pred = tree1.predict_discrete_input(X_pred)

    print(y_pred)
    tree1.print_tree_discrete_discrete(tree1.root,0,'')


def test_decision_tree_real_discrete():
    x1 = np.random.uniform(0, 10, 100)
    x2 = np.random.uniform(0, 10, 100)
    y = np.random.choice(['red', 'blue', 'green'], 100)
    x = pd.DataFrame({'x1': x1, 'x2': x2})
    y = pd.DataFrame({'y': y})

#
#     tree2 = DecisionTree(max_depth=2)
#     print(tree2.get_split_attr_value(x,y,['x1','x2']))

# test_decision_tree_real_discrete()


def test_decision_tree_discrete_real():
    N = 30
    P = 5
    X = pd.DataFrame({i: pd.Series(np.random.randint(P, size=N), dtype="category") for i in range(5)})
    y = pd.Series(np.random.randn(N))
    column=["f1","f2","f3"]
    lst=[[0,1,1,0,1],[1,0,1,0,0],[1,0,0,1,1]]
    y=pd.Series([5.6,2.3,4.5,3.2,6.1])
    X=pd.DataFrame({'f1':lst[0],'f2':lst[1],'f3':lst[2]})
    tree1 = DecisionTree(max_depth=2)
    print(X,y)
    tree1.fit(X, y)
    print(y.mean())
    print()
    print(tree1.root)
    tree1.print_tree_discrete_input(tree1.root, 0, '')
# test_decision_tree_discrete_real()

def test_decision_tree_real_discrete():
    x1 = np.random.uniform(0, 10, 100)
    x2 = np.random.uniform(0, 10, 100)
    y = np.random.choice(['red', 'blue', 'green'], 100)
    x = pd.DataFrame({'x1': x1, 'x2': x2})
    y = pd.DataFrame({'y': y})
    print(x)
    print(y)

    tree2 = DecisionTree(max_depth=10)
    # print(tree2.get_split_attr_value(x, y, ['x1','x2']))
    tree2.root = tree2.construct_tree_real_discrete(x, pd.Series(y['y']), ['x1','x2'],0,)
    tree2.print_tree_real_input(tree2.root, 0, '')

def test_decision_tree_real_real():
    x1 = np.random.uniform(0, 10, 100)
    x2 = np.random.uniform(0, 10, 100)
    y = np.random.uniform(0, 10, 100)
    x = pd.DataFrame({'x1': x1, 'x2': x2})
    y = pd.DataFrame({'y': y})
    print(x)
    print(y)

    # tree3 = DecisionTree(max_depth=10)
    # print(tree3.get_split_attr_value(x, y, ['x1','x2'],'real_real'))

    tree_real_real = DecisionTree(max_depth=10)
    # # print(tree2.get_split_attr_value(x, y, ['x1','x2']))
    tree_real_real.root = tree_real_real.construct_tree_real_discrete(x, pd.Series(y['y']), ['x1','x2'],0,'real_real')
    print(tree_real_real.root)
    tree_real_real.print_tree_real_input(tree_real_real.root, 0, '')
    print(tree_real_real.predict(x))

test_decision_tree_real_real()



def test_get_split_tree_real_discrete():
    x1 = np.array([1,2,3])
    y = np.array(['red', 'red', 'blue'])
    x = pd.DataFrame({'x1': x1})
    y = pd.DataFrame({'y': y})

    tree2 = DecisionTree(max_depth=2)
    # print(tree2.construct_tree_real_discrete(x, pd.Series(y['y']), ['x1'], cur_depth=10))
    tree2.root = tree2.construct_tree_real_discrete(x, pd.Series(y['y']), ['x1'], cur_depth=10)
    tree2.print_tree_real_input(tree2.root, 0, '')
    # print(tree2.construct_tree_real_discrete(x, pd.Series(y['y']), ['x1','x2'],0))

# test_decision_tree()
# test_decision_tree_real_discrete()
# test_get_split_tree_real_discrete()
