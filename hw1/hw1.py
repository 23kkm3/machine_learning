#!/usr/bin/env python

import pandas as pd
import numpy as np
from data_loader import load_simulated_data, load_insurance_data



def mse(Y, Yhat):
    """
    Calculates mean squared error between ground-truth Y and predicted Y

    """

    return np.mean((Y-Yhat)**2)

def mad(theta, theta_new):
    
    diff = 0
    # calculate mean absolute deviation between theta and theta new
    for i in range(len(theta)):
        diff += abs((theta[i] - theta_new[i]))

        # final calculation is dividing the difference by 'n' aka the length of theta
        mad_val = diff/len(theta)

    return mad_val

def rsquare(Y, Yhat):
    """
    Implementation of the R squared metric based on ground-truth Y and predicted Y
    """

    # TODO: implement the R squared metric
    # compute mean of Y, and for each row of data simply output the mean as the predicted value
    
    # r^2 compares performance of learned model against the naive algorithm that simply outputs
    # the mean value as it's prediction
    numer = 0
    denom = 0

    for i in range(len(Y)):
        numer += ((Y[i] - Yhat[i])**2)
        denom += ((Y[i]-(np.mean(Y)))**2)

    r_squared = 1 - (numer/denom)
    
    return r_squared 


class LinearRegression:
    """
    Class for linear regression
    """
    
    def __init__(self, learning_rate=0.1):
        """
        Constructor for the class. Learning rate is
        any positive number controlling step size of gradient descent.
        """

        self.learning_rate = learning_rate
        self.theta = None # theta is initialized once we fit the model
    
    def _calculate_gradient(self, Xmat, Y, theta_p, h=1e-5):
        """
        Helper function for computing the gradient at a point theta_p.
        """

        # get dimensions of the matrix
        n, d = Xmat.shape

        grad_vec = np.zeros(d)

        # TODO: implement gradient calculation code here
        # must be able to compute gradient for any vector w/ d entries (possibly using a loop)
        # h is used for the numerical approach

        # initial guess = theta p in this case

        # retrieving the Yhat (predicted) value 
        Yhat = Xmat@theta_p

        for x in range(0, d):

            # theta_p_plus_h is theta perturbed 
            # copy vector
            theta_p_plus_h = theta_p.copy()

            # increment the value stored at index x in perturbed vector by h
            theta_p_plus_h[x] = theta_p_plus_h[x] + h

            # retrieve new Yhat with the perturbed theta p vector
            new_Yhat = Xmat@theta_p_plus_h 

            # gradient caclulation using mse as loss function
            grad = (mse(Y, new_Yhat) - mse(Y, Yhat))/h

            grad_vec[x] = grad

        return grad_vec

    def fit(self, Xmat, Y, max_iterations=1000, tolerance=1e-6, verbose=False):
        """
        Fit a linear regression model using training data Xmat and Y.
        """

        # get dimensions of the matrix
        n, d = Xmat.shape        
        
        # initialize the first theta and theta new randomly
        theta = np.random.uniform(-5, 5, d)
        theta_new = np.random.uniform(-5, 5, d)
        iteration = 0

        # TODO: Implement code that performs gradient descent until "convergence"
        # i.e., until max_iterations or until the change in theta measured by mean absolute difference
        # is less than the tolerance argument

        x = 0

        while x < max_iterations:
            # if we don't update our new theta to be the old, we will just keep iterating on the same points
            theta = theta_new.copy()

            # calculate theta_new
            theta_new = (theta - self.learning_rate * self._calculate_gradient(Xmat, Y, theta, h=1e-5))

            # mad is a measure of variation; the dist. btwn ea. data pt. and the mean of data
            if mad(theta, theta_new) < tolerance:
                break

            x += 1

        # set the theta attribute of the model to the final value from gradient descent
        self.theta = theta_new.copy()

def main():
    """
    Do not edit this function. This function is used for grading purposes only.
    """

    np.random.seed(0)

    #################
    # Simulated data
    #################
    Xmat, Y, feature_names = load_simulated_data()
    model = LinearRegression()
    model.fit(Xmat, Y)
    Yhat = Xmat @ model.theta
    print("Simulated data results:\n" + "-"*4)
    print("Simulated data fitted weights", {feature_names[i]: round(model.theta[i], 2) for i in range(len(feature_names))})
    print("R squared simulated data", rsquare(Y, Yhat), "\n")

    #################
    # Insurance data
    #################
    Xmat_train, Y_train, Xmat_test, Y_test, feature_names = load_insurance_data()
    model = LinearRegression()
    model.fit(Xmat_train, Y_train) # only use training data for fitting
    Yhat_test = Xmat_test @ model.theta # evaluate on the test data
    print("Insurance data results:\n" + "-"*4)
    print("Insurance data fitted weights", {feature_names[i]: round(model.theta[i], 2) for i in range(len(feature_names))})
    print("R squared insurance data", rsquare(Y_test, Yhat_test))


if __name__ == "__main__":
    main()
