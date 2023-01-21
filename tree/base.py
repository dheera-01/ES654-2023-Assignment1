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
from tree.utils import entropy, information_gain, gini_index, variance_reduction

import matplotlib.pyplot as plt

np.random.seed(42)




@dataclass
class Node:
    atrribute:str
    isLeaf:bool
    children: dict
    output: str | int | float
    # split_value: 0.0 #Represents which value to compare sample[attribute]

@dataclass
class DecisionTree:
    # criterion: Literal["information_gain", "gini_index"]  # criterion won't be used for regression
    max_depth: int  # The maximum depth the tree can grow to
    root = Node('hello', True, {}, 'hello')


    def traverse_discrete_input(self,node:Node,data:dict):
        if(node.isLeaf==True):
            return node.output
        else:
            return self.traverse_discrete_input(node.children[data[node.atrribute]],data)

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
        print("hello")
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
        print()
        print(attr)
        max_gain_attr = ""
        max_gain = float('-inf')

        if(entropy(y)==0.0):
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

        n1 = Node(atrribute=max_gain_attr, isLeaf=False, children=children, output=None)
        return n1

    def construct_tree_discrete_real(self, X, y, attr, cur_depth):

        max_var_red_attr = ""
        max_var_red = float('-inf')

        if (len(y)<2 or y.std() == 0.0):
            output = y.mean()
            n1 = Node(atrribute=attr, isLeaf=True, children={}, output=output)
            return n1

        if (self.max_depth == cur_depth):
            output = y.mean()
            n1 = Node(atrribute=attr, isLeaf=True, children={}, output=output)
            return n1

        if (len(attr) == 0):
            output = y.mean()
            n1 = Node(atrribute=None, isLeaf=True, output=output, children={})
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

        n1 = Node(atrribute=max_var_red_attr, isLeaf=False, children=children, output=None)
        return n1

    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        """
        Function to train and construct the decision tree
        """
        
        # attributes=["outlook","humidity","rain"]
        attributes = self.get_attributes_X(X)
        # self.root = self.construct_tree(X,y,attributes,-1);
        self.root=self.construct_tree(X,y,attributes,0)


    def predict_discrete_input(self,X:pd.DataFrame)->pd.Series:
        y_pred = pd.Series()
        for i in range(len(X)):
            y_pred._set_value(i, self.traverse_discrete_input(self.root, X.iloc[i]))
        return y_pred

    def predict(self, X: pd.DataFrame) -> pd.Series:
        """
        Funtion to run the decision tree on test inputs
        """
        return self.predict_discrete_input(X)

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

    tree1 = DecisionTree(max_depth=10)
    tree1.fit(x, pd.Series(y['PlayTennis']))
    X_pred=pd.DataFrame({"Outlook":['Sunny','Rain'],"Temperature":['Hot','Mild'],"Humidity":['High','Normal'],"Wind":['Strong','Weak']})
    y_pred=tree1.predict(X_pred)
    print(y_pred)

# def test_decision_tree_real_discrete():
#     x1 = np.random.uniform(0, 10, 100)
#     x2 = np.random.uniform(0, 10, 100)
#     y = np.random.choice(['red', 'blue', 'green'], 100)
#     x = pd.DataFrame({'x1': x1, 'x2': x2})
#     y = pd.DataFrame({'y': y})
#     print(x)
#     print(y)
#
#     tree2 = DecisionTree(max_depth=2)
#     print(tree2.get_split_attr_value(x,y,['x1','x2']))





# def test_decision_tree_discrete_real():
#     N = 30
#     P = 5
#     X = pd.DataFrame({i: pd.Series(np.random.randint(P, size=N), dtype="category") for i in range(5)})
#     y = pd.Series(np.random.randn(N))
#     column=["f1","f2","f3"]
#     lst=[[0,1,1,0,1],[1,0,1,0,0],[1,0,0,1,1]]
#     y=pd.Series([5.6,2.3,4.5,3.2,6.1])
#     X=pd.DataFrame({'f1':lst[0],'f2':lst[1],'f3':lst[2]})
#     tree1 = DecisionTree(max_depth=0)
#     tree1.fit(X, y)
#     X_pred=pd.DataFrame({'f1':[1,0,0],'f2':[1,0,0],'f3':[1,0,1]})
#     print("Mean of the data",y.mean())
#     print(tree1.predict(X_pred))


# test_decision_tree_discrete_real()
test_decision_tree()
