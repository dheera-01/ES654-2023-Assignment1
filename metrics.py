from typing import Union
import pandas as pd
import numpy as np

def accuracy(y_hat: pd.Series, y: pd.Series) -> float:
    """
    Function to calculate the accuracy
    """

    """
    The following assert checks if sizes of y_hat and y are equal.
    Students are required to add appropriate assert checks at places to
    ensure that the function does not fail in corner cases.
    """
    assert y_hat.size == y.size
    # TODO: Write here
    ct=0
    total=len(y_hat)
    for i in range(len(y_hat)):
        if(y_hat[i]==y[i]):
            ct=ct+1
    return float(ct)/total


def precision(y_hat: pd.Series, y: pd.Series, cls: Union[int, str]) -> float:
    """
    Function to calculate the precision
    """
    assert y_hat.size == y.size
    total=0
    ct=0
    for i in range(len(y_hat)):
        if(y_hat[i]==cls and y[i]==cls):
            ct=ct+1
        if(y_hat[i]==cls):
            total=total+1
    return float(ct)/total



def recall(y_hat: pd.Series, y: pd.Series, cls: Union[int, str]) -> float:
    """
    Function to calculate the recall
    """

    assert y_hat.size==y.size
    total=0
    ct=0
    for i in range(len(y_hat)):
        if(y_hat[i]==cls and y[i]==cls):
            ct=ct+1
        if(y[i]==cls):
            total=total+1
    return float(ct)/total


def rmse(y_hat: pd.Series, y: pd.Series) -> float:
    """
    Function to calculate the root-mean-squared-error(rmse)
    """
    assert y_hat.size == y.size
    return np.sqrt(np.power(y_hat-y,2).sum()/len(y))



def mae(y_hat: pd.Series, y: pd.Series) -> float:
    """
    Function to calculate the mean-absolute-error(mae)
    """
    assert y_hat.size == y.size
    return abs(y_hat-y).sum()/len(y)

y=pd.Series(["Good","Good","Good","Bad","Bad","Bad"])
y_hat=pd.Series(["Good","Good","Bad","Bad","Bad","Bad"])
print(accuracy(y_hat,y))
print(precision(y_hat,y,"Good"))
print(recall(y_hat,y,"Good"))