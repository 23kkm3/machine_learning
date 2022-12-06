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
    data = data.drop(columns=["DGN", "PRE6", "PRE14"])
    print("data: ", data)

    feat = np.array([1 if risk=="T" else 0 for risk in data["PRE5"]]) # repeat for all of the other columns or write loop to iterate over column names
    data["PRE5"] = feat
    print("pre5: ", data["PRE5"])

    #data_features = [1 if data.loc[column]=="T" else 0 for column in data.columns]
    print("revised features: ", )
    # Xmat = data.drop(columns=["Risk1Yr"])

    feature_names = data.columns
    print("feature names: ", feature_names)
    data_features = data[feature_names]

    Xmat = data_features.to_numpy()

    # split into training, validation, testing
    Xmat_train, Xmat_test, Y_train, Y_test = train_test_split(Xmat, Y, test_size=0.33, random_state=42)
    Xmat_train, Xmat_val, Y_train, Y_val = train_test_split(Xmat_train, Y_train, test_size=0.33, random_state=42)
    
    # standardize the data
    mean = np.mean(Xmat_train, axis=0)
    std = np.std(Xmat_train, axis=0)
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
