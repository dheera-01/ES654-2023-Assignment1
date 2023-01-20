import pandas as pd
#manually imported
import math

def entropy(Y: pd.Series) -> float:
    """
    Function to calculate the entropy
    """
    n = len(Y)
    rtr = 0.0
    for i in Y.value_counts():
        rtr = rtr - i/n * math.log2(i/n)
    return rtr

def gini_index(Y: pd.Series) -> float:
    """
    Function to calculate the gini index
    """
    n = len(Y)
    rtr = 0.0
    for i in Y.value_counts():
        rtr = rtr + (i/n)*(1-i/n)

    return rtr

def information_gain(Y: pd.Series, attr: pd.Series) -> float:
    """
    Function to calculate the information gain
    """
    entropy_Y = entropy(Y)
    # df_attr = pd.DataFrame({:Y, 'attr':attr})
    index = attr.groupby(["Outlook"]).groups['Overcast'].tolist()
    print(index)
    new_x = y.iloc[index]
    print(new_x)
