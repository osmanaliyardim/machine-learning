# -*- coding: utf-8 -*-
"""
Created on Fri Feb 22 10:28:35 2019

@author: Osman Ali Yardim

Machine Learning - Logistic Regression Implementation

Cancerous Cell Detection / with Kaggle data
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# import data
data = pd.read_csv('logistic-regression-dataset.csv')

# data manipulation
data.info()
data.drop(['Unnamed: 32', 'id'], axis=1, inplace=True)
data.diagnosis = [1 if each == 'M' else 0 for each in data.diagnosis]

y = data.diagnosis.values
x_data = data.drop(['diagnosis'], axis=1)

# normalization
x = (x_data - np.min(x_data))/(np.max(x_data) - np.min(x_data)).values

# %% train test split
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

x_train = x_train.T
x_test = x_test.T
y_train = y_train.T
y_test = y_test.T

# %% parameter initialize and sigmoid function

def initialize_weights_and_bias(dimension):
    
    w = np.full((dimension, 1), 0.01)
    b = 0.0
    return w, b

# w, b = initialize_weights_and_bias(30)
    
def sigmoid(z):
    
    y_head = 1/(1 + np.exp(-z))
    return y_head

# print(sigmoid(0))

# %% forward & backward propagation

def forward_backward_propagation(w, b, x_train, y_train):   
    # forward propagation
    z = np.dot(w.T, x_train) + b
    y_head = sigmoid(z)
    loss = -y_train*np.log(y_head) - (1-y_train)*np.log(1-y_head)
    cost = (np.sum(loss))/x_train.shape[1]
    
    # backward propagation
    derivative_weight = (np.dot(x_train, ((y_head - y_train).T))) / x_train.shape[1]
    derivative_bias = np.sum(y_head - y_train)/ x_train.shape[1]
    gradients = {"derivative_weight": derivative_weight, "derivative_bias": derivative_bias}
    
    return cost, gradients

# %% update (learning) parameters

def update(w, b, x_train, y_train, learning_rate, number_of_iterarion):
    cost_list = []
    cost_list2 = []
    index = []
    
    # updating(learning) parameters is number_of_iterarion times
    for i in range(number_of_iterarion):
        
        # make forward and backward propagation and find cost and gradients
        cost, gradients = forward_backward_propagation(w, b, x_train, y_train)
        cost_list.append(cost)
        
        # lets update
        w = w - learning_rate * gradients["derivative_weight"]
        b = b - learning_rate * gradients["derivative_bias"]
        if i % 10 == 0:
            cost_list2.append(cost)
            index.append(i)
            print("Cost after iteration %i: %f" %(i, cost))
    
    # we update (learn) parameters weights and bias
    parameters = {"weight": w,"bias": b}
    plt.plot(index, cost_list2)
    plt.xticks(index, rotation='vertical')
    plt.xlabel("Number of Iterarion")
    plt.ylabel("Cost")
    plt.show()
    return parameters, gradients, cost_list

# parameters, gradients, cost_list = update(w, b, x_train, y_train, learning_rate = 0.009, number_of_iterarion=200)
    
# %% prediction

def predict(w,b,x_test):
    
    # x_test is a input for forward propagation
    z = sigmoid(np.dot(w.T, x_test) + b)
    Y_prediction = np.zeros((1, x_test.shape[1]))
    
    # if z is bigger than 0.5, our prediction is sign one (y_head=1),
    # if z is smaller than 0.5, our prediction is sign zero (y_head=0),
    for i in range(z.shape[1]):
        if z[0, i]<= 0.5:
            Y_prediction[0, i] = 0
        else:
            Y_prediction[0, i] = 1

    return Y_prediction

# predict(parameters["weight"],parameters["bias"],x_test)
    
# %% logistic regression

def logistic_regression(x_train, y_train, x_test, y_test, learning_rate,  num_iterations):
    
    # initialize
    dimension =  x_train.shape[0]  # that is 4096
    w,b = initialize_weights_and_bias(dimension)
    
    # do not change learning rate
    parameters, gradients, cost_list = update(w, b, x_train, y_train, learning_rate, num_iterations)
    
    y_prediction_test = predict(parameters["weight"], parameters["bias"], x_test)
    y_prediction_train = predict(parameters["weight"], parameters["bias"], x_train)

    # print train/test errors
    print("train accuracy: {} %".format(100 - np.mean(np.abs(y_prediction_train - y_train)) * 100))
    print("test accuracy: {} %".format(100 - np.mean(np.abs(y_prediction_test - y_test)) * 100))
    
logistic_regression(x_train, y_train, x_test, y_test, learning_rate = 0.01, num_iterations = 300)