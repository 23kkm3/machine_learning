#!/usr/bin/env python

from xxlimited import new
import numpy as np
import pandas as pd
from data_loader2 import load_simulated_data, load_thoracic_data
from sklearn.model_selection import train_test_split


def accuracy(Y, Yhat):
    """
    Function for computing accuracy
    
    Y is a vector of the true labels and Yhat is a vector of estimated 0/1 labels
    """
    
    return np.sum(Y==Yhat)/len(Y)
    
def entropy(data, outcome_name):
    """
    Compute entropy of assuming the outcome is binary
    
    data: a pandas dataframe
    outcome_name: a string corresponding to name of the outcome varibale
    """
    
    # compute p(Y=1)
    proba_Y1 = np.sum(data[outcome_name])
    
    # check if entropy is 0
    if proba_Y1 == 0 or proba_Y1 == 1:
        return 0
    
    # compute and return entropy
    return -(proba_Y1)*np.log2(proba_Y1) - (1-proba_Y1)*np.log2(1-proba_Y1)

def weighted_entropy(data1, data2, outcome_name):
    """
    Calculate the weighted entropy of two datasets
    
    data1: a pandas dataframe
    data2: a pandas dataframe
    outcome_name: a string corresponding to name of the outcome varibale
    """

    # denominator of weighted entropy function
    combined_lengths = len(data1) + len(data2)

    # first component of entropy function numerator
    dataset_one = len(data1) * entropy(data1, outcome_name)

    # second component of entropy function numerator
    dataset_two = len(data2) * entropy(data2, outcome_name)

    # weighted entropy equation
    wgt_entr = (dataset_one + dataset_two)/combined_lengths

    return wgt_entr

class Vertex:
    """
    Class for defining a vertex in a decision tree
    """

    def __init__(self, feature_name=None, threshold=None, prediction=None):
        
        self.left_child = None
        self.right_child = None
        self.feature_name = feature_name # name of feature to split on
        self.threshold = threshold # threshold of feature to split on
        self.prediction = prediction # predicted value -- applies only to leaf nodes


class DecisionTree:
    """
    Class for building decision trees
    """
    
    def __init__(self, max_depth=np.inf):
        
        self.max_depth = max_depth
        self.root = None
        
    def _get_best_split(self, data, outcome_name):
        """
        Method to compute the best split of the data to minimize entropy

        data: pandas dataframe
        outcome_name: a string corresponding to name of the outcome varibale

        Returns
        ------
        A tuple consisting of:
        (i) String corresponding to name of the best feature
        (ii) Float corresponding to value to split the feature on
        (iii) pandas dataframe consisting of subset of rows of data where best_feature < best_threshold
        (iv) pandas dataframe consisting of subset of rows of data where best_feature >= best_threshold
        """
         # Implementation of get best split:
         # Naive approach to finding best split = compute entropy for scratch for each possible split
         # One split: gets all rows of data where feat. is <= val
         # Other split: gets all rows of data where the feature is >= value

        best_entropy = entropy(data, outcome_name)
        best_feature = None
        best_threshold = 0
        data_left = None
        data_right = None

        # loop over each feature
        # list of all names of features
        for feature in data.columns:
            # skip Y outcome column feature
            if feature == outcome_name:
                continue 
            # loop over each value of the feature
            # look at computing the unique values for that particular feature 

            for i in data[feature]:
                unique_vals = []
                # check to avoid recomputing recurring values
                if i not in unique_vals: 
                    unique_vals.append(i)
                    # list of indicies less than i
                    indices = data[feature] < i 
                    # all rows that are less than some value in the pandas df
                    data_left = data[indices] 

                    # list of indicies  greater than or = to i
                    indices2 = data[feature] >= i
                    # all rows that are greater than or equal to some value in the pandas df
                    data_right = data[indices2] 
                    new_entropy = weighted_entropy(data_left, data_right, outcome_name)
                
                    if new_entropy < best_entropy:
                        # reassign best_entropy with updated value
                        best_entropy = new_entropy
                        best_feature = feature
                        best_threshold = i
                
        # Assign the values in the data frame:
        # data_left = less than the threshold
        # data_right = greater than or equal to the threshold
         
        data_left = data[data[best_feature] < best_threshold]
        data_right = data[data[best_feature] >= best_threshold]

        return best_feature, best_threshold, data_left, data_right
        
    def _build_tree(self, data, outcome_name, curr_depth=0):
        """
        Recursive function to build a decision tree. Refer to the HW pdf
        for more details on the implementation of this function.

        data: pandas dataframe
        outcome_name: a string corresponding to name of the outcome varibale
        curr_depth: integer corresponding to current depth of the tree
        """

        # Reasonable base cases:
        # 1. All values of the outcome in the provided data are the same (all 1s or 0s)
        if (entropy(data, outcome_name) == 0):
            final = Vertex()
            if np.mean(data[outcome_name]) == 1:
                # set the prediciton value are the leaf node
                final.prediction = 1
            if np.mean(data[outcome_name]) == 0:
                final.prediction = 0
            return final
        
        # 2. Current depth is greater than or equal to max depth
        # At leaf nodes, always output the majority class
        if curr_depth >= self.max_depth:
            # If more outcomes of 1 than 0 or more 0s than 1s, output majority
            final2 = Vertex()
            zero_count = 0
            one_count = 0
            for i in data[outcome_name]:
                if i == 1:
                    one_count += 1
                if i == 0:
                    zero_count += 1
            if one_count >= zero_count:
                final2.prediction = 1
            else:
                final2.prediction = 0
            
            return final2


        new_vertex = Vertex()
        # create new vertex, recursively build left subtree and right subtree for this 
        # vertex based on best split and returns vertex

        best_feature, best_threshold, data_left, data_right = self._get_best_split(data, outcome_name)
        
        new_vertex.feature_name = best_feature
        new_vertex.threshold = best_threshold
        new_vertex.left_child = self._build_tree(data_left, outcome_name, curr_depth+1)
        new_vertex.right_child = self._build_tree(data_right, outcome_name, curr_depth+1)

        # curr_depth += 1

        return new_vertex
        

    def fit(self, Xmat, Y, outcome_name="Risk1Yr"):
        """
        Fit a decision tree model using training data Xmat and Y.
        
        Xmat: pandas dataframe of features
        Y: numpy array of 0/1 outcomes
        outcome_name: string corresponding to name of outcome variable
        """

        data = Xmat.copy()
        data[outcome_name] = Y
        self.root = self._build_tree(data, outcome_name, 0)

     
    def _dfs_to_leaf(self, sample):
        """
        Perform a depth first traversal to find the leaf node that the given sample belongs to

        sample: dictionary mapping from feature names to values of the feature
        """
        n = len(sample)
        visited = [] 
        s = [] # empty stack initialized to store nodes from sample  

        # self.root of decision tree 
        # only one thing will be on the stack at a time
        # if all nodes have been visited, end
        s.append(self.root)

        # If we run out of things to add to stack, we have reached a leaf node and we're done
        while len(s) > 0: 
            eval_node = s.pop()

            if eval_node.prediction is not None:
                return eval_node.prediction
            
            # Left tree traversal
            # eval_node is the feature in the tree we are currently looking at
            if sample[eval_node.feature_name] < eval_node.threshold:
                visited.append(eval_node)

                # move to lefthand side of tree and eval. a new feature
                eval_node = eval_node.left_child
                s.append(eval_node)

            # Right tree traversal
            else:
                visited.append(eval_node)
                eval_node = eval_node.right_child
                s.append(eval_node)

    def predict(self, Xmat):
        """
        Predict 0/1 labels for a data matrix

        Xmat: pandas dataframe
        """
        
        predictions = []

        for i in range(len(Xmat)):
            
            example_i = {feature: Xmat[feature][i] for feature in Xmat.columns}
            predictions.append(self._dfs_to_leaf(example_i))
        
        return np.array(predictions)
    
    def print_tree(self, vertex=None, indent="  "):
        """
        Function to produce text representation of the tree
        """
        
        # initialize to root node
        if not vertex:
            vertex = self.root

        # if we're at the leaf output the prediction
        if vertex.prediction is not None:
            print("Output", vertex.prediction)

        else:
            print(vertex.feature_name, "<", round(vertex.threshold, 2), "?")
            print(indent, "Left child: ", end="")
            self.print_tree(vertex.left_child, indent + indent)
            print(indent, "Right child: ", end="")
            self.print_tree(vertex.right_child, indent + indent)


def main():
    """
    Edit only the one line marked as # EDIT ME in this function. The rest is used for grading purposes
    """


    #################
    # Simulated data
    #################
    np.random.seed(333)
    Xmat, Y  = load_simulated_data()
    data = Xmat.copy()
    data["Y"] = Y

    # test for your predict method
    # by manually creating a decision tree
    model = DecisionTree()
    model.root = Vertex(feature_name="PRE4", threshold=2.2)
    model.root.left_child = Vertex(prediction=0)
    model.root.right_child = Vertex(feature_name="PRE5", threshold=2.2)
    model.root.right_child.left_child = Vertex(prediction=0)
    model.root.right_child.right_child = Vertex(prediction=1)
    print("-"*60 + "\n" + "Hand crafted tree for testing predict\n" + "-"*60)
    model.print_tree()
    Yhat = model.predict(Xmat)
    print("Accuracy of hand crafted tree", round(accuracy(Y, Yhat), 2), "\n")

    # test for your best split method
    print("-"*60 + "\n" + "Simple test for finding best split\n" + "-"*60)
    model = DecisionTree(max_depth=2)
    best_feature, threshold, _, _ = model._get_best_split(data, "Y")
    print("Best feature and threshold found", best_feature, round(threshold, 2), "\n")


    # test for your fit method
    model.fit(Xmat, Y)
    print("-"*60 + "\n" + "Algorithmically generated tree for testing build_tree\n" + "-"*60)
    model.print_tree()
    Yhat = model.predict(data)
    print("Accuracy of algorithmically generated tree", round(accuracy(Y, Yhat), 2), "\n")

    #####################
    # Thoracic data
    #####################
    Xmat, Y = load_thoracic_data()

    # create a train test split
    Xtrain, Xtest, Ytrain, Ytest = train_test_split(Xmat, Y, test_size=0.25, random_state=0)
    Xtrain.reset_index(inplace=True, drop=True)
    Xtest.reset_index(inplace=True, drop=True)

    # find best depth using a form of cross validation/bootstrapping
    possible_depths = [1, 2, 3, 4, 5]
    best_depth = 0
    best_accuracy = 0
    for depth in possible_depths:
    
        accuracies = []
        for i in range(5):
            print("training tree", i)
            Xtrain_i, Xval, Ytrain_i, Yval = train_test_split(Xtrain, Ytrain, test_size=0.3, random_state=i)
            Xtrain_i.reset_index(inplace=True, drop=True)
            Xval.reset_index(inplace=True, drop=True)
            model = DecisionTree(max_depth=depth)
            model.fit(Xtrain_i, Ytrain_i, "Risk1Yr")
            accuracies.append(accuracy(Yval, model.predict(Xval)))
           
    
        mean_accuracy = sum(accuracies)/len(accuracies)
        if mean_accuracy > best_accuracy:
            best_accuracy = mean_accuracy
            best_depth = depth
    

    print("-"*60 + "\n" + "Hyperparameter tuning on thoracic data\n" + "-"*60)
    print("Best depth =", best_depth, "\n")
    model = DecisionTree(max_depth=best_depth)
    model.fit(Xtrain, Ytrain, "survival_status")
    print("-"*60 + "\n" + "Final tree for thoracic data\n" + "-"*60)
    model.print_tree()
    print("Test accuracy", round(accuracy(Ytest, model.predict(Xtest)), 2), "\n")


if __name__ == "__main__":
    main()
