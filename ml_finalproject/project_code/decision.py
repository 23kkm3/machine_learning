# import pandas and numpy
import pandas as pd
import numpy as np

# imports for various machine learning algorithms
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from scipy import stats

# modules for splitting and evaluating the data
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# module for visualizing decision trees
from sklearn.tree import export_graphviz
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pydot
from IPython.display import Image
from six import StringIO
from numpy.core.numeric import full

# def classification_report(Y, Yhat, train=True):
#     """
#     Produce a classification report including
#     accuracy, precision, recall, and f1 scores
#     for the model.
#     """
    
#     ################
#     # COMPONENT 1
#     ################
#     if train:
#         print("-"*10)
#         print("Training results")
#         print("-"*10)
#     else:
#         print("-"*10)
#         print("Testing results")
#         print("-"*10)
#     print("Accuracy", round(accuracy_score(Y, Yhat), 3))
#     print("F1 score", round(f1_score(Y, Yhat), 3))
#     print("Precision", round(precision_score(Y, Yhat), 3))
#     print("Recall", round(recall_score(Y, Yhat), 3))

#     full_data = pd.read_csv("thoracic_data.csv")

#     print("Data columns", full_data.columns)
#     #print(full_data.StandardHours)
#     print("data: ", full_data)
#     print("Data columns", full_data.columns)


def main():

    ################
    # PRE-PROCESSING
    ################

    data = pd.read_csv("thoracic_data.csv")
    print("FULL DATA", data)

    # make outcome array
    Y = np.array([1 if outcome=="T" else 0 for outcome in data["Risk1Yr"]])

    # drop irrelevant features
    data = data.drop(columns=["DGN", "PRE6", "PRE14", "PRE5", "PRE19"])

    # separate features from the outcome
    data_features = data.drop(["Risk1Yr"], axis="columns")


    feat1 = np.array([1 if risk=="T" else 0 for risk in data["PRE7"]])
    data["PRE7"] = feat1
    feat2 = np.array([1 if risk=="T" else 0 for risk in data["PRE8"]])
    data["PRE8"] = feat2
    feat3 = np.array([1 if risk=="T" else 0 for risk in data["PRE9"]])
    data["PRE9"] = feat3
    feat4 = np.array([1 if risk=="T" else 0 for risk in data["PRE10"]])
    data["PRE10"] = feat4
    feat5 = np.array([1 if risk=="T" else 0 for risk in data["PRE11"]])
    data["PRE11"] = feat5
    feat6 = np.array([1 if risk=="T" else 0 for risk in data["PRE17"]])
    data["PRE17"] = feat6
    feat8 = np.array([1 if risk=="T" else 0 for risk in data["PRE25"]])
    data["PRE25"] = feat8
    feat9 = np.array([1 if risk=="T" else 0 for risk in data["PRE30"]])
    data["PRE30"] = feat9
    feat10 = np.array([1 if risk=="T" else 0 for risk in data["PRE32"]])
    data["PRE32"] = feat10

    data = data.drop(columns=["Risk1Yr"])
    Xmat = data_features.to_numpy()
    print("REVISED DATA", data)

    # create train and test splits
    X_train, X_test, Y_train, Y_test = train_test_split(Xmat, Y, test_size=0.3, random_state=42)
    X_train, Xmat_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size=0.33, random_state=42)
    
    # classification_report(Y, Yhat, train=True)

    ################
    # COMPONENT 3
    ################
    model = DecisionTreeClassifier(max_depth=2, criterion="entropy", random_state=0)
    model.fit(X_train, Y_train)

    dot_data = StringIO()
    export_graphviz(model, out_file=dot_data, feature_names=list(Xmat.columns), filled=True)
    graph = pydot.graph_from_dot_data(dot_data.getvalue())
    Image(graph[0].create_png())


if __name__ == "__main__":
    main()
