import pandas as pd
#manually imported
import math

def entropy(Y: pd.Series) -> float:
    """
    Function to calculate the entropy
    """
    n = len(Y)
    if n == 1:
        return 0.0
    rtr = 0.0
    for i in Y.value_counts():
        temp = i/n
        if temp != 0:
            rtr = rtr - temp * math.log(temp,2)
    return rtr

# y = pd.DataFrame({'y':[1]})
# print('entropy',pd.Series(y['y'] ,dtype='category'))

def gini_index(Y: pd.Series) -> float:
    """
    Function to calculate the gini index
    """
    n = len(Y)
    rtr = 0.0
    for i in Y.value_counts():
        rtr = rtr + (i/n)*(1-i/n)

    return rtr

def information_gain_gini(Y:pd.Series,attr:pd.Series):
    info_gain = gini_index(Y)
    df_attr = pd.DataFrame({'attr': attr.values})
    df_Y = pd.DataFrame({'Y': Y.values})

    for grp in df_attr.groupby(['attr']).groups.keys():
        index = df_attr.groupby(['attr']).groups[grp].tolist()
        new_Y = df_Y.iloc[index]
        new_Y = new_Y.reset_index(drop=True)
        gini_new_Y = gini_index(pd.Series(new_Y['Y']))
        info_gain = info_gain - len(new_Y) / len(Y) * gini_new_Y

    return info_gain

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
        new_Y = new_Y.reset_index(drop=True)
        entropy_new_Y = entropy(pd.Series(new_Y['Y']))
        info_gain = info_gain - len(new_Y)/len(Y) * entropy_new_Y

    return info_gain


def variance_reduction(Y:pd.Series,attr: pd.Series)->float:

    if(len(Y)<=1):
        return 0
    intial_std=Y.std()*Y.std()
    inital_var=intial_std*intial_std


    df_attr = pd.DataFrame({'attr': attr.values})
    df_Y = pd.DataFrame({'Y': Y.values})

    for grp in df_attr.groupby(['attr']).groups.keys():
        index = df_attr.groupby(['attr']).groups[grp].tolist()
        new_Y = df_Y.iloc[index]

        if(len(new_Y["Y"])>1):
            std_new_Y = pd.Series(new_Y['Y']).std()
            variance_new_Y = std_new_Y * std_new_Y

            inital_var = inital_var - (len(new_Y) / len(Y)) * variance_new_Y


    return inital_var