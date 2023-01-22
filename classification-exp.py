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
# plt.show()

# Splitting dataset into 70-30 train-test split
df_X = pd.DataFrame(X)
df_y = pd.DataFrame(y, dtype="category")

split_index = int(0.7 * len(df_X))
X_train = df_X.iloc[:split_index]
y_train = df_y.iloc[:split_index]

X_test = df_X.iloc[split_index:].reset_index(drop=True)
y_test = df_y.iloc[split_index:].reset_index(drop=True)
y_test_series = pd.Series(y_test.iloc[:, 0])

tree1 = DecisionTree(criterion='information_gain',max_depth=1)
tree1.fit(X_train, pd.Series(y_train.iloc[:, 0]))
result = tree1.predict(X_test)

# printing metrices for question 2 a
print('Accuracy: ', accuracy(result, pd.Series(y_test_series)))
for cls in y_test_series.unique():
    print(cls,': ')
    print('Precision: ', precision(result, pd.Series(y_test_series), cls))
    print('Recall: ', recall(result,  pd.Series(y_test_series), cls))

def tuning(X_y: pd.DataFrame) -> float:
    max_depth = 10
    df_depths_accuracy = pd.DataFrame(columns=['depth', 'fold_accuracy1', 'fold_accuracy2', 'fold_accuracy3', 'fold_accuracy4'])
    df_depths_accuracy.iloc[:, 0] = range(0,max_depth)

    for first_split in range(3,-1,-1):
        first_split_index = int(first_split * 0.25 * len(X_y))
        X_y_validation = X_y.iloc[first_split_index: first_split_index + int(0.25 * len(X_y))].reset_index(drop=True)
        X_y_train = pd.concat([X_y.iloc[:first_split_index], X_y.iloc[first_split_index + int(0.25 * len(X_y)):]], axis=0).reset_index(drop=True)

        for depth in range(0,max_depth):
            tree = DecisionTree(criterion='information_gain',max_depth=depth)

            tree.fit(X_y_train.iloc[:, :-1], pd.Series(X_y_train.iloc[:, -1]))
            y_hat = tree.predict(X_y_validation.iloc[:,:-1]) #y_hat series
            df_depths_accuracy.iloc[depth, first_split + 1] = accuracy(y_hat, pd.Series(X_y_validation.iloc[:, -1]))

    df_depths_accuracy['mean_accuracy'] = df_depths_accuracy.iloc[:, 1:].mean(axis=1)
    max_index = df_depths_accuracy['mean_accuracy'].idxmax()
    optimal_depth = df_depths_accuracy.iloc[max_index, 0]
    return optimal_depth

# returns overall accuracy of the model
def nested_train_test(X_y: pd.DataFrame) -> pd.DataFrame:
    df_fold_depth = pd.DataFrame(columns=['Fold_number','Optimal_depth','Accuracy'])
    df_fold_depth.iloc[:, 0] = range(1,6)

    for first_split in range(4, -1, -1):
        first_split_index = int(first_split * 0.2 * len(X_y))
        X_y_test = X_y.iloc[first_split_index: first_split_index + int(0.2 * len(X_y))].reset_index(drop=True)
        X_y_train = pd.concat([X_y.iloc[:first_split_index], X_y.iloc[first_split_index + int(0.2 * len(X_y)):]],axis=0).reset_index(drop=True)

        optimal_depth = tuning(X_y_train)

        optimal_tree = DecisionTree(criterion='information_gain',max_depth=optimal_depth)
        optimal_tree.fit(X_y_train.iloc[:, :-1], pd.Series(X_y_train.iloc[:, -1]))
        y_hat = optimal_tree.predict(X_y_test.iloc[:,:-1])

        optimal_tree_accuracy = accuracy(y_hat, pd.Series(X_y_test.iloc[:, -1]))

        df_fold_depth.iloc[first_split, 1] = optimal_depth
        df_fold_depth.iloc[first_split, 2] = optimal_tree_accuracy

    return df_fold_depth

# For question 2 b where it prints the optimal depths and accuracy for 5 folds using nested cross validation
print(nested_train_test(pd.concat([X_train, y_train], axis=1)))
