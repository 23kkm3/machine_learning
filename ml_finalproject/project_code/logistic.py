#!/usr/bin/env python

import numpy as np
import pandas as pd
import random
import math
from sklearn.model_selection import train_test_split
# from data_loader import make_simulated_data
from data_loader import load_thoracic_data


def mean_negative_loglikelihood(Y, pYhat):
    """
    Function for computing the mean negative loglikelihood

    Y is a vector of the true 0/1 labels.
    pYhat is a vector of the estimated probabilities, where each entry is p(Y=1 | ...)
    """

    first_component = (Y * (np.log(pYhat)))

    second_component = ((1-Y) * np.log(1-pYhat))

    mean_neg_loglike = np.mean(first_component + second_component)

    return (-1) * mean_neg_loglike

def accuracy(Y, Yhat):
    """
    Function for computing accuracy

    Y is a vector of the true labels and Yhat is a vector of estimated 0/1 labels
    """

    return np.sum(Y==Yhat)/len(Y)

def sigmoid(V):
    """
    Function for mapping a vector of floats to probabilities via the sigmoid function
    """

    return 1/(1+np.exp(-V))

class LogisticRegression:

    def __init__(self, learning_rate=0.1, lamda=None):
        """
        Constructor for the class. Learning rate is any positive number controllings tep size of gradient descent.
        Lamda is a positive number controlling the strength of regularization.
        When None, no penalty is added.
        """

        self.learning_rate = learning_rate
        self.lamda = lamda
        self.theta = None # theta is initialized once we fit the model

    def _calculate_gradient(self, Xmat, Y, theta_p, h=1e-5):
        """
        Helper function for computing the gradient at a point theta_p.
        """

        #initialize an empty gradient vector
        n, d = Xmat.shape

        grad_vec = np.zeros(d)

        # initial guess = theta p in this case

        # retrieving the Yhat (predicted) value
        # once we pass it through the sigmoid func, we know it will be a valid probability between 0 and 1
        # applying the sigmoid function provides a more smooth, differentiable mapping
        Yhat = sigmoid(Xmat@theta_p)

        for x in range(0,d):

            # theta_p_plus_h is theta perturbed
            # copy vector
            theta_p_plus_h = theta_p.copy()

            # increment the value stored at index x in perturbed vector by h
            theta_p_plus_h[x] = theta_p_plus_h[x] + h

            #retrieve new Yhat with the perturbed theta p vector
            new_Yhat = sigmoid(Xmat@theta_p_plus_h)

            #implementation of L2 regularization; adding lamda value to loss function
            if self.lamda != None:
                grad = ((mean_negative_loglikelihood(Y, new_Yhat) + self.lamda * (np.sum((theta_p_plus_h)**2))) - (mean_negative_loglikelihood(Y, Yhat) + self.lamda * (np.sum((theta_p)**2))))/h
            else:
            #gradient calculation using mean negative log likelihood as loss function
                grad = (mean_negative_loglikelihood(Y, new_Yhat) - mean_negative_loglikelihood(Y, Yhat))/h

            grad_vec[x] = grad

        return grad_vec

    def fit(self, Xmat, Y, max_iterations=1000, tolerance=1e-6, verbose=False):
        """
        Fit a logistic regression model using training data Xmat and Y.
        """

        # add a column of ones for the intercept
        n, d = Xmat.shape

        # initialize theta and theta new randomly
        theta = np.random.uniform(-1, 1, d)
        theta_new = np.random.uniform(-1, 1, d)
        iteration = 0

        # keep going until convergence
        while iteration < max_iterations and np.mean(np.abs(theta_new-theta )) >= tolerance:

            if verbose:
                print("Iteration", iteration, "theta=", theta)

            # Implementation of gradient descent
            theta = theta_new.copy()

            theta_new = (theta - self.learning_rate * self._calculate_gradient(Xmat, Y, theta, h=1e-5))

            iteration +=1

        self.theta = theta_new.copy()

    def predict(self, Xmat):
        """
        Predict 0/1 labels for a data matrix Xmat based on the following rule:
        if p(Y=1|X) > 0.5 output a label of 1, else output a label of 0
        """
        # Check the vector for probability at ea. row
        # We have already obtained out optimal theta bc logistic regression has executed at this point
        Yhat = sigmoid(Xmat@self.theta)
        new_list = []

        for i in range(0, len(Yhat)):
            if Yhat[i] > 0.5:
                new_list.append(1)
            else:
                new_list.append(0)
        
        return new_list

def main():

    # option to sort by relevant variables here later down the line (see work in dataloader2.py)

    #####################
    # TESTING : Thoracic data
    #####################
    feature_names, data = load_thoracic_data()
    model_base = LogisticRegression(learning_rate=0.2, lamda=0.0)
    model_base.fit(data["Xmat_train"], data["Y_train"])
    model_lowl2 = LogisticRegression(learning_rate=0.2, lamda=0.01)
    model_lowl2.fit(data["Xmat_train"], data["Y_train"])
    model_highl2 = LogisticRegression(learning_rate=0.2, lamda=0.2)
    model_highl2.fit(data["Xmat_train"], data["Y_train"])
    
    Yhat_val_base = model_base.predict(data["Xmat_val"])
    Yhat_val_lowl2 = model_lowl2.predict(data["Xmat_val"])
    Yhat_val_highl2 = model_highl2.predict(data["Xmat_val"])

    accuracy_base = accuracy(data["Y_val"], Yhat_val_base)
    accuracy_lowl2 = accuracy(data["Y_val"], Yhat_val_lowl2)
    accuracy_highl2 = accuracy(data["Y_val"], Yhat_val_highl2)

    print("\nThoracic data results:\n" + "-"*4)
    print("Validation accuracy no regularization", accuracy_base)
    print("Validation accuracy lamda=0.01", accuracy_lowl2)
    print("Validation accuracy lamda=0.2", accuracy_highl2)

    # choose best model
    best_model = model_lowl2 # edited from model_base to model_lowl2 to select the preferred model
    Yhat_test = best_model.predict(data["Xmat_test"])
    print("Test accuracy", accuracy(data["Y_test"], Yhat_test))
    print("Thoracic data weights", {feature_names[i]: round(best_model.theta[i], 2) for i in range(len(feature_names))})

if __name__ == "__main__":
    main()

