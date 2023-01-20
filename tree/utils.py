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
    info_gain = entropy(Y)
    df_attr = pd.DataFrame({'attr':attr.values})
    df_Y = pd.DataFrame({'Y':Y.values})

    for grp in df_attr.groupby(['attr']).groups.keys():
        index = df_attr.groupby(['attr']).groups[grp].tolist()
        new_Y = df_Y.iloc[index]
        entropy_new_Y = entropy(pd.Series(new_Y['Y']))
        info_gain = info_gain - len(new_Y)/len(Y) * entropy_new_Y

    return info_gain
