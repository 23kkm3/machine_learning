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
#import pydot
from IPython.display import Image
from six import StringIO

def classification_report(Y, Yhat, train=True):
    """
    Produce a classification report including
    accuracy, precision, recall, and f1 scores
    for the model.
    """
    
    if train:
        print("-"*10)
        print("Training results")
        print("-"*10)
    else:
        print("-"*10)
        print("Testing results")
        print("-"*10)
    print("Accuracy", round(accuracy_score(Y, Yhat), 3))
    print("F1 score", round(f1_score(Y, Yhat), 3))
    print("Precision", round(precision_score(Y, Yhat), 3))
    print("Recall", round(recall_score(Y, Yhat), 3))

    full_data = pd.read_csv("thoracic_data.csv")

    from numpy.core.numeric import full
    print("Data columns", full_data.columns)
    #print(full_data.StandardHours)
    full_data.head()