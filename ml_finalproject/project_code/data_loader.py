import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


# def load_simulated_data():
#     """
#     Load simulated data and return an X matrix
#     for features and Y vector for outcomes
#     """

#     # load data with pandas
#     data = pd.read_csv("simulated_data.csv")
#     feature_names = ["intercept"] + list(data.columns)
#     feature_names.remove("Y")

#     # convert to numpy matrix
#     Dmat = data.to_numpy()
#     n, d = Dmat.shape

#     # separate X matrix and Y vector
#     Xmat = Dmat[:, 0:-1]
#     Y = Dmat[:, -1]

#     # add a column of 1s for intercept term and return
#     Xmat = np.column_stack((np.ones(n), Xmat))
#     return Xmat, Y, feature_names


def load_thoracic_data():
    
    # load in data with pandas
    data = pd.read_csv("thoracic_data.csv")
    
    # convert T/F outcomes to 1/0
    Y = np.array([1 if risk=="T" else 0 for risk in data["Risk1Yr"]])
    print("Y values: ", Y)

    # get the feature matrix
    data = data.drop(columns=["DGN", "PRE6", "PRE14", "PRE5", "PRE19", "Risk1Yr"])
    print("data: ", data)

    # feat = np.array([1 if risk=="T" else 0 for risk in data["PRE5"]]) # repeat for all of the other columns or write loop to iterate over column names
    # data["PRE5"] = feat
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
    # feat7 = np.array([1 if risk=="T" else 0 for risk in data["PRE19"]])
    # data["PRE19"] = feat7
    feat8 = np.array([1 if risk=="T" else 0 for risk in data["PRE25"]])
    data["PRE25"] = feat8
    feat9 = np.array([1 if risk=="T" else 0 for risk in data["PRE30"]])
    data["PRE30"] = feat9
    feat10 = np.array([1 if risk=="T" else 0 for risk in data["PRE32"]])
    data["PRE32"] = feat10
    # feat11 = np.array([1 if risk=="T" else 0 for risk in data["Risk1Yr"]])
    # data["Risk1Yr"] = feat11

    print("data: ", data)
    # Xmat = data.drop(columns=["Risk1Yr"])

    feature_names = data.columns
    print("feature names: ", feature_names)
    data_features = data[feature_names]
    Xmat = data_features.to_numpy()
    print("Xmat: ", Xmat)
    # split into training, validation, testing
    Xmat_train, Xmat_test, Y_train, Y_test = train_test_split(Xmat, Y, test_size=0.33, random_state=42)
    Xmat_train, Xmat_val, Y_train, Y_val = train_test_split(Xmat_train, Y_train, test_size=0.33, random_state=42)
    
    # get rid of any features that end up with a std of 0 
    # standardize the data
    mean = np.mean(Xmat_train, axis=0)
    print("mean: ", mean)
    std = np.std(Xmat_train, axis=0)
    print("std: ", std)
    Xmat_train = (Xmat_train - mean)/std
    Xmat_val = (Xmat_val - mean)/std
    Xmat_test = (Xmat_test - mean)/std
    
    # add a column of ones for the intercept term
    Xmat_train = np.column_stack((np.ones(len(Xmat_train)), Xmat_train))
    Xmat_val = np.column_stack((np.ones(len(Xmat_val)), Xmat_val))
    Xmat_test = np.column_stack((np.ones(len(Xmat_test)), Xmat_test))
    feature_names = ["intercept"] + feature_names
    
    # return the train/validation/test datasets
    return feature_names, {"Xmat_train": Xmat_train, "Xmat_val": Xmat_val, "Xmat_test": Xmat_test,
                           "Y_train": Y_train, "Y_val": Y_val, "Y_test": Y_test}
