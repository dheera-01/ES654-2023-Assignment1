
import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
from tree.base import DecisionTree
from metrics import *

np.random.seed(42)
num_average_time = 100


def plot_data(df_fold_depth: pd.DataFrame,case:str):
    n_axis = df_fold_depth['N']
    m_axis = df_fold_depth['M']
    fit_time_axis = df_fold_depth['Fit Time']
    predict_time_axis = df_fold_depth['Predict Time']
    fig = plt.figure()

    ax2 = fig.add_subplot(111, projection='3d')
    ax2.set_title('Fit Time for '+ case)
    ax2.plot_trisurf(n_axis, m_axis, fit_time_axis, cmap="hot")

    ax2.set_xlabel('N')
    ax2.set_ylabel('M')
    ax2.set_zlabel('Fit Time')

    fig2 = plt.figure()
    ax = fig2.add_subplot(111, projection='3d')
    ax.set_title('Predict Time for ' + case)
    ax.plot_trisurf(n_axis, m_axis, predict_time_axis, cmap="hot")

    ax.set_xlabel('N')
    ax.set_ylabel('M')
    ax.set_zlabel('Predict Time')

max_samples=50
max_features=10
max_depth=5 # fixed value

df_fold_depth = pd.DataFrame(columns=['N','M','Fit Time','Predict Time'])
case_list=["Real_Real","Real_Discrete","Discrete_Discrete","Discrete_Real"]

for case in case_list:
    df_fold_depth=pd.DataFrame(columns=['N','M','Fit Time','Predict Time'])
    for n in range(2,max_samples):
        for m in range(1,max_features):
            # print("Case: ",case," N: ",n," M: ",m)
            if(case=="Real_Real"):
                X = pd.DataFrame(np.random.randn(n, m))
                y = pd.Series(np.random.randn(n))
            elif(case=="Real_Discrete"):
                X = pd.DataFrame(np.random.randn(n, m))
                y = pd.Series(np.random.randint(m, size=n), dtype="category")
            elif(case=="Discrete_Discrete"):
                X = pd.DataFrame({i: pd.Series(np.random.randint(low=0,high=2, size=n), dtype="category") for i in range(m)})
                y = pd.Series(np.random.randint(low=0,high=2, size=n), dtype="category")
            elif(case=="Discrete_Real"):
                X = pd.DataFrame({i: pd.Series(np.random.randint(low=0,high=2, size=n), dtype="category") for i in range(m)})
                y = pd.Series(np.random.randn(n))
            df_y = pd.DataFrame(y)
            split_index = int(0.8 * len(X))
            X_train = X.iloc[:split_index]
            y_train = df_y.iloc[:split_index]

            X_test = X.iloc[split_index:].reset_index(drop=True)
            y_test = df_y.iloc[split_index:].reset_index(drop=True)
            y_test_series = pd.Series(y_test.iloc[:, 0])

            tree1 = DecisionTree(max_depth=max_depth, criterion="gini_index")
            fit_begin=time.time()
            tree1.fit(X_train, pd.Series(y_train.iloc[:, 0]))
            fit_end=time.time()
            total_fit_time=fit_end-fit_begin

            predict_begin = time.time()
            result_our = tree1.predict(X_test)
            predict_end = time.time()
            predict_time_total=predict_end-predict_begin
            df_fold_depth = df_fold_depth.append({'N': n, 'M': m,'Fit Time':total_fit_time,'Predict Time':predict_time_total},ignore_index=True)
    plot_data(df_fold_depth,case)

plt.show()
