
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tree.base import DecisionTree
from sklearn.tree import DecisionTreeRegressor
from metrics import *

np.random.seed(42)

# Read real-estate data set
# ...
# 
columns=['mpg','cylinders','displacement','horsepower','weight','acceleration','model year','origin','car name']
df=pd.DataFrame(pd.read_fwf("auto-mpg.data",names=columns,na_values='?'))
df=df.dropna()
df=df.reset_index(drop=True)
# print(df)
y=pd.Series(df['mpg'])
X=df.drop(columns=['mpg','car name'])
max_depth=5
df_fold_depth = pd.DataFrame(columns=['Depth','RMSE_Our_Tree','RMSE_Sklearn_Tree'])

df_y = pd.DataFrame(y)
for depth in range(1,max_depth):


    split_index = int(0.8 * len(X))
    X_train = X.iloc[:split_index]
    y_train = df_y.iloc[:split_index]

    X_test = X.iloc[split_index:].reset_index(drop=True)
    y_test = df_y.iloc[split_index:].reset_index(drop=True)
    y_test_series = pd.Series(y_test.iloc[:, 0])

    tree1 = DecisionTree(max_depth=depth,criterion="gini_index")
    tree1.fit(X_train, pd.Series(y_train.iloc[:, 0]))
    result_our = tree1.predict(X_test)
    # print("RMSE:", rmse(result, pd.Series(y_test_series)))
    regressor = DecisionTreeRegressor(max_depth=depth)
    regressor.fit(X_train, pd.Series(y_train.iloc[:, 0]))
    result = regressor.predict(X_test)
    # print("RMSE:", rmse(result, pd.Series(y_test_series)))
    df_fold_depth = df_fold_depth.append({'Depth': depth, 'RMSE_Our_Tree': rmse(result_our, pd.Series(y_test_series)),'RMSE_Sklearn_Tree': rmse(result, pd.Series(y_test_series))}, ignore_index=True)

print(df_fold_depth)