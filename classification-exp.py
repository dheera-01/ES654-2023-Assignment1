import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tree.base import DecisionTree
from sklearn.datasets import make_classification
from metrics import *

np.random.seed(42)


# Genenerate a random dataset
X, y = make_classification(n_features=2, n_redundant=0, n_informative=2, random_state=1, n_clusters_per_class=2, class_sep=0.5)
# plt.scatter(X[:, 0], X[:, 1], c=y)
# plt.show(block=True)
# plt.hold(True)
# Splitting dataset into 70-30 train-test split
df_X = pd.DataFrame(X)
df_y = pd.DataFrame(y, dtype="category")

split_index = int(0.7 * len(df_X))
X_train = df_X.iloc[:split_index]
y_train = df_y.iloc[:split_index]

X_test = df_X.iloc[split_index:].reset_index(drop=True)
y_test = df_y.iloc[split_index:].reset_index(drop=True)

tree1 = DecisionTree(max_depth=20)
tree1.fit(X_train, pd.Series(y_train[0]))
# tree1.plot()
result = tree1.predict(X_test)

#convert to np array X_test and result
X_test = np.array(X_test)
result = np.array(result)
plt.scatter(X_test[:, 0], X_test[:, 1], c=result)
plt.show()
print(result)
