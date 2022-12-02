import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


def load_simulated_data():
    """
    Load simulated data and return an X matrix
    for features and Y vector for outcomes
    """

    # load data with pandas
    data = pd.read_csv("thoracic_data.csv") # used to be loading simulated data

    # separate features and outcomes
    Xmat = data.drop(["Risk1Yr"], axis="columns")
    Y = np.array([outcome for outcome in data["Risk1Yr"]])  # COME BACK AND FIX HERE

    return Xmat, Y


def load_thoracic_data():
    """
    Helper function for loading bone
    marrow transplant data
    """
    # open file and process all variable names
    all_variables = []

    with open("thoracic_data.csv", "r") as f:
        
        for line in f:
            line = line.strip()
            
            if len(line) == 0 or line[0] == "$" or line[0] == "%" or line == "@data":
                continue
                
            if line[0] == "@":
                var_name = line.split()[-2]
                all_variables.append(var_name)

    # subset data to a small number of important variables
    relevant_variables = set(["PRE4", "PRE5", "DGN",
                              "PRE10", "PRE14", "PRE17", "PRE19",
                              "PRE25", "AGE", "Risk1Yr"])
    
    data = pd.read_csv("thoracic_data.csv", names=all_variables)
    data.drop(columns=set(data.columns) - relevant_variables, inplace=True) 
    print("columns: ", data.columns)

    # convert disease type to numeric values
    # disease_map = {disease: i for i, disease in enumerate(set(data["DGN"]))}
    # data["DGN"] = [disease_map[d] for d in data["DGN"]]

    # ignore missing data
    # can probably get rid of this part because the data was already pretty clean
    data.replace({'?': np.nan}, regex=False, inplace=True)
    data.dropna(inplace=True)

     # Code to pre-process the data here
    data_clean = data #data.drop(columns=["id", "name"])

    Y = data_clean['Risk1Yr']
    # TODO: more pre-processing if needed and model training, return the predictions on the test
    # Xmat = data_clean.drop(columns=["PRE4"]).to_numpy()
    # print("cleaned data: ", data)
    # Y = data_clean["Risk1Yr"].replace("T", 1) # added to make T/F values into numerical 0/1
    # Y = data_clean["Risk1Yr"].replace("F", 0)
    # Xmat = data_clean
    # print("cleaned outcomes: ", Xmat["Risk1Yr"])

    Xmat_train, Xmat_test, Y_train, Y_test = train_test_split(Xmat, Y, test_size=0.33, random_state=42)
    Xmat_train, Xmat_val, Y_train, Y_val = train_test_split(Xmat_train, Y_train, test_size=0.33, random_state=42)
    n, d = Xmat_train.shape

    # NEW ADDITION: STANDARDIZING DATA

    # standardize the data ; need it here because need training split completed
    mean = np.mean(Xmat_train, axis=0)
    std = np.std(Xmat_train, axis=0)
    Xmat_train = (Xmat_train - mean)/std
    Xmat_val = (Xmat_val - mean)/std
    Xmat_test = (Xmat_test - mean)/std

    Xmat = data # data.drop(["survival_status"], axis="columns") # got rid of drop portion here bc not dropping anything

    Y = np.array([survival for survival in data['Risk1Yr']])

    return Xmat, Y