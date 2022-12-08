# import pandas and numpy
import pandas as pd
import numpy as np

# imports for various machine learning algorithms
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from scipy import stats
from sklearn import tree 

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

def accuracy(Y, Yhat):
    """
    Function for computing accuracy
    
    Y is a vector of the true labels and Yhat is a vector of estimated 0/1 labels
    """
    
    return np.sum(Y==Yhat)/len(Y)

def main():

    ################
    # PRE-PROCESSING
    ################

    data = pd.read_csv("thoracic_data.csv")

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
    feature_names = data.columns
    # print("features: ", feature_names)
    data_features = data[feature_names]
    Xmat = data_features.to_numpy()
    # print("REVISED DATA", Xmat)

    #################
    # Simulated data
    #################
    np.random.seed(333)
    data = Xmat.copy()


    # simulated data componenet cut out from below 
    model = DecisionTreeClassifier()

    # test for your fit method
    model.fit(Xmat, Y)
    print("-"*60 + "\n" + "Algorithmically generated tree for testing build_tree\n" + "-"*60)
    text_representation = tree.export_text(model)
    print("tree repres: ", text_representation)
    Yhat = model.predict(data)
    print("Accuracy of algorithmically generated tree", round(accuracy(Y, Yhat), 2), "\n")

    # create a train test split
    Xtrain, Xtest, Ytrain, Ytest = train_test_split(Xmat, Y, test_size=0.25, random_state=0)
    # Xtrain.reset_index(inplace=True, drop=True)
    # Xtest.reset_index(inplace=True, drop=True)

    # find best depth using a form of cross validation/bootstrapping
    possible_depths = [1, 2, 3, 4, 5]
    best_depth = 0
    best_accuracy = 0
    for depth in possible_depths:
    
        accuracies = []
        for i in range(5):
            print("training tree", i)
            Xtrain_i, Xval, Ytrain_i, Yval = train_test_split(Xtrain, Ytrain, test_size=0.3, random_state=i)
            # Xtrain_i.reset_index(inplace=True, drop=True)
            # Xval.reset_index(inplace=True, drop=True)
            model = DecisionTreeClassifier(max_depth=depth)
            model.fit(Xtrain_i, Ytrain_i)
            # model.fit(Xtrain_i, Ytrain_i, "Risk1Yr")
            accuracies.append(accuracy(Yval, model.predict(Xval)))
           
    
        mean_accuracy = sum(accuracies)/len(accuracies)
        if mean_accuracy > best_accuracy:
            best_accuracy = mean_accuracy
            best_depth = depth
    

        print("-"*60 + "\n" + "Hyperparameter tuning on transplant data\n" + "-"*60)
        print("Best depth =", best_depth, "\n")
        model = DecisionTreeClassifier(max_depth=best_depth)
        model.fit(Xtrain, Ytrain)
        # model.fit(Xtrain, Ytrain, Y)
        print("-"*60 + "\n" + "Final tree for transplant data\n" + "-"*60)
        # model.print_tree()
        print("Test accuracy", round(accuracy(Ytest, model.predict(Xtest)), 2), "\n")

    classifier = tree.DecisionTreeClassifier(random_state=0)
    #iris = load_iris()
    classifier = classifier.fit(Xtrain, Ytrain)
    tree.plot_tree(classifier, filled=True)
    plt.show()
    
if __name__ == "__main__":
    main()
