# Author:   Ahmad Chalhoub
# Project:  Converting the single-neuron cat classifier into a full single-layer network

import h5py
import numpy as np
import random
import matplotlib.pyplot as plt

# loads in training and testing dataset (images and labels) and the class IDs and returns that information
def load_dataset():
    train_dataset = h5py.File('D:\School\Super_Senior\Deep_Learning\Assignments\Chalhoub_cat_classifier\\train_catvnoncat.h5', "r")
    train_set_x_orig = np.array(train_dataset["train_set_x"][:])
    train_set_y_orig = np.array(train_dataset["train_set_y"][:])
    train_set_y_orig = train_set_y_orig.reshape((1, train_set_y_orig.shape[0]))

    test_dataset = h5py.File('D:\School\Super_Senior\Deep_Learning\Assignments\Chalhoub_cat_classifier\\test_catvnoncat.h5', "r")
    test_set_x_orig = np.array(test_dataset["test_set_x"][:])
    test_set_y_orig = np.array(test_dataset["test_set_y"][:])
    test_set_y_orig = test_set_y_orig.reshape((1, test_set_y_orig.shape[0]))

    classes = np.array(test_dataset["list_classes"][:])

    return train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig, classes

# creates input matrix for training
def create_input_matrix(train_set_x_orig):
    X = 1/255*(np.reshape(train_set_x_orig, (train_set_x_orig.shape[0], train_set_x_orig[-1].flatten().shape[0])).T)

    return X

# initializes the parameters (W matrix and b scalar)
def initialize_parameters(X, hidden_neurons):
    W_1 = np.random.randn(hidden_neurons, X.shape[0])           # (4, 12288)
    b_1 = np.random.randn(hidden_neurons, 1)                    # (4x1)
    W_2 = np.zeros((1, hidden_neurons))                         # (1x4)
    b_2 = np.zeros((1, 1))                                      # (1x1)

    return W_1, b_1, W_2, b_2

# applies relu function on Z_1
def relu(Z_1):
    if Z_1 > 0:
        A_1 = Z_1
    else:
        A_1 = 0

    return A_1

# applies tanh function on Z_1
def tanh(Z_1):
    A_1 = np.tanh(Z_1)
    
    return A_1

# applies sigmoid function on 'Z_2'
def sigmoid(Z_2):
    A_2 = 1/(1 + np.exp(-Z_2))

    return A_2

# goes through the forward propagation steps of training the model
def forward_prop(Y, W_1, b_1, W_2, b_2, X):
    m = X.shape[1]
    Z_1 = np.dot(W_1.T, X) + b_1
    A_1 = tanh(Z_1)
    Z_2 = np.dot(W_2.T, A_1) + b_2                                     
    A_2 = sigmoid(Z_2)
    L = (Y *np.log(A_2)) + (1-Y)*np.log(1-A_2)                                      # loss
    cost = -(1/m)*np.sum(L)                                                     
    accuracy = 1 - cost

    return Z_1, A_1, Z_2, A_2, cost, accuracy

# goes through the backward propagation steps of training the model
def back_prop(Y, X, Z_1, A_1, Z_2, A_2, W_1, b_1, W_2, b_2):
    m = X.shape[1]
    dZ_2 = A_2 - Y
    dW_2 = (1/m)*np.matmul(dZ_2, A_2.T)
    db_2 = (1/m)*np.sum(dZ_2, axis=1)
    g_prime_of_Z_1 = 2                                                 # CALCULATE g' !!!
    dZ_1 = np.matmul(W_2.T, dZ_2) * g_prime_of_Z_1
    dW_1 = (1/m)*np.matmul(dZ_1, X.T)
    db_1 = (1/m)*np.sum(dZ_1, axis=1)

    return dW_1, db_1, dW_2, db_2

# obtains the testing accuracy by using the images in the 'test' dataset to compute accuracy
def prediction(test_set_x_orig, Y_test, W, b):
    m_test =test_set_x_orig.shape[0]
    X_test = 1/255*(np.reshape(test_set_x_orig, (test_set_x_orig.shape[0], test_set_x_orig[-1].flatten().shape[0])).T)
    prediction_values = sigmoid(np.dot(W.T, X_test) + b)
    test_accuracy = 1 - (abs((1/m_test)*np.sum(prediction_values - Y_test)))
    print('test accuracy: ', round(test_accuracy,3)*100, '%')

def main():
    train_set_x_orig, Y_train, test_set_x_orig, Y_test, classes = load_dataset()
    X = create_input_matrix(train_set_x_orig)
    W_1, b_1, W_2, b_2 = initialize_parameters(X, hidden_neurons=4)

    # for-loop to train the network
    for i in range(7000):
        Z_1, A_1, Z_2, A_2, cost, accuracy = forward_prop(Y_train, W_1, b_1, W_2, b_2, X)
        dW_1, db_1, dW_2, db_2 = back_prop(Y_train, X, Z_1, A_1, Z_2, A_2, W_1, b_1, W_2, b_2)
        W_1 -= (0.025*dW_1)                                                 # update and scale weight values
        b_1 -= (0.025*db_1)                                                 # update and scale bias values
        W_2 -= (0.025*dW_2)
        b_2 -= (0.025*db_2)

        if i % 200 ==0:
            print('cost: ', round(cost, 4))
            print('accuracy: ', round(accuracy*100, 4), '%')
    
    print(' ')
    #prediction(test_set_x_orig, Y_test, W, b)                               # compute predictions and accuracy using testing dataset
    print(' ')

if __name__ == '__main__':
    main()