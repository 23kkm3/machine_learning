# import pandas and numpy
import pandas as pd
import numpy as np

# imports for various machine learning algorithms
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from scipy import stats

# modules for splitting and evaluating the data
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix


def accuracy(Y, Yhat):
    """
    Function for computing accuracy
    
    Y is a vector of the true labels and Yhat is a vector of estimated 0/1 labels
    """
    
    return np.sum(Y==Yhat)/len(Y)

def main():

    data = pd.read_csv("thoracic_data.csv")

    #####################
    # PRE-PROCESSING DATA
    #####################

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

    # create a train test split
    Xtrain, Xtest, Ytrain, Ytest = train_test_split(Xmat, Y, test_size=0.25, random_state=0)


    model = LogisticRegression()
    model.fit(Xtrain, Ytrain)
    model_tree = DecisionTreeClassifier(max_depth=None, criterion="entropy")
    model_tree.fit(Xtrain, Ytrain)
    model_forest = RandomForestClassifier(n_estimators=500)
    model_forest.fit(Xmat, Ytrain)

    print("Logistic regression train acc", accuracy(Ytrain, model.predict(Xtrain)), "test acc", accuracy(Ytest, model.predict(Xtest)))
    print("Decision tree train acc", accuracy(Ytrain, model_tree.predict(Xtrain)), "test acc", accuracy(Ytest, model_tree.predict(Xtest)))
    print("Random forest train acc", accuracy(Ytrain, model_forest.predict(Xtrain)), "test acc", accuracy(Ytest, model_forest.predict(Xtest)))

    # classifier = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5,2), random_state=1)

if __name__ == "__main__":
    main() 
