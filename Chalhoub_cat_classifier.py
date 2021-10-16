# Author:   Ahmad Chalhoub
# Project:  Converting the single-neuron cat classifier into a full single-layer network
# Due date: 10/20/2021

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

# initializes the parameters (W and b values)
def initialize_parameters(X, hidden_neurons):
    W_1 = np.random.randn(hidden_neurons, X.shape[0])*0.01           # (4, 12288)
    b_1 = np.random.randn(hidden_neurons, 1)*0.01                    # (4, 1)
    W_2 = np.random.randn(1, hidden_neurons)*0.01                    # (1, 4)
    b_2 = np.random.randn(1, 1)*0.01                                 # (1, 1)

    return W_1, b_1, W_2, b_2

# applies relu function on Z_1
def relu(Z_1):

    return np.maximum(0, Z_1)

# g' when using a relu activation function
def relu_prime(Z_1):

    return np.maximum(0, Z_1)

# applies tanh function on Z_1
def tanh_func(Z_1):
    A_1 = np.tanh(Z_1)
    #A_1 = (np.exp(Z_1) - np.exp(-Z_1)) / (np.exp(Z_1) + np.exp(-Z_1))
    
    return A_1

# g' when using a tanh activation function
def tanh_prime(Z_1):

    return (1 - tanh_func(Z_1)*tanh_func(Z_1))
    
# applies sigmoid function on 'Z_2'
def sigmoid(Z_2):
    A_2 = 1/(1 + np.exp(-Z_2))

    return A_2

# g' when using a sigmoid activation function
def sigmoid_prime(Z_1):

    return (sigmoid(Z_1) * (1 - sigmoid(Z_1)))

# goes through the forward propagation steps of training the model
def forward_prop(Y, W_1, b_1, W_2, b_2, X, activation):
    m = X.shape[1]
    Z_1 = np.dot(W_1, X) + b_1

    if activation == 'tanh':
        A_1 = tanh_func(Z_1)
    elif activation == 'sigmoid':
        A_1 = sigmoid(Z_1)
    elif activation == 'relu':
        A_1 = relu(Z_1)
    
    Z_2 = np.dot(W_2, A_1) + b_2                                     
    A_2 = sigmoid(Z_2)
    L = (Y *np.log(A_2)) + (1-Y)*np.log(1-A_2)
    cost = -(1/m)*np.sum(L)                                                     
    accuracy = 1 - cost

    return Z_1, A_1, A_2, cost, accuracy, activation

# goes through the backward propagation steps of training the model
def back_prop(Y, X, Z_1, A_1, A_2, W_2, activation):
    m = X.shape[1]
    dZ_2 = A_2 - Y
    dW_2 = (1/m)*np.matmul(dZ_2, A_1.T)
    db_2 = (1/m)*np.sum(dZ_2, axis=1)

    if activation == 'tanh':
        Z_1_g_prime = tanh_prime(Z_1)
    elif activation == 'sigmoid':
        Z_1_g_prime = sigmoid_prime(Z_1)
    elif activation == 'relu':
        Z_1_g_prime = relu_prime(Z_1)

    dZ_1 = np.multiply(np.matmul(W_2.T, dZ_2), Z_1_g_prime)
    dW_1 = (1/m)*np.matmul(dZ_1, X.T)
    db_1 = (1/m)*np.sum(dZ_1, axis=1)

    return dW_1, db_1, dW_2, db_2

# obtains the testing accuracy by using the images in the 'test' dataset to compute accuracy
def prediction(test_set_x_orig, Y_test, W_1, b_1, W_2, b_2, activation):
    m_test =test_set_x_orig.shape[0]
    X_test = 1/255*(np.reshape(test_set_x_orig, (test_set_x_orig.shape[0], test_set_x_orig[-1].flatten().shape[0])).T)   

    if activation == 'tanh':
        A_1_test = tanh_func(np.dot(W_1, X_test) + b_1)
    elif activation == 'sigmoid':
        A_1_test = sigmoid(np.dot(W_1, X_test) + b_1)
    elif activation == 'relu':
        A_1_test = relu(np.dot(W_1, X_test) + b_1)

    prediction_values = sigmoid(np.dot(W_2, A_1_test) + b_2)    
    test_accuracy = 1 - ((1/m_test)*np.sum(abs(prediction_values - Y_test)))
    print('test accuracy: ', round(test_accuracy,3)*100, '%')

def main():
    np.random.seed(42)      # always generate the same random values for the weights and biases for consistency

    lr = 0.05              # learning rate

    train_set_x_orig, Y_train, test_set_x_orig, Y_test, classes = load_dataset()
    X = create_input_matrix(train_set_x_orig)
    hidden_neurons = 12
    W_1, b_1, W_2, b_2 = initialize_parameters(X, hidden_neurons)
    
    # for-loop to train the network
    for i in range(5000):
        Z_1, A_1, A_2, cost, accuracy, activation = forward_prop(Y_train, W_1, b_1, W_2, b_2, X, activation='tanh')
        dW_1, db_1, dW_2, db_2 = back_prop(Y_train, X, Z_1, A_1, A_2, W_2, activation)
        db_1 = np.reshape(db_1, (hidden_neurons, 1))
        db_2 = np.reshape(db_2, (1, 1))

        W_1 -= (lr*dW_1)                                              
        b_1 -= (lr*db_1)                                                
        W_2 -= (lr*dW_2)
        b_2 -= (lr*db_2)

        if i % 200 == 0:
            print('Iteration number ', i)
            print('cost: ', round(cost, 4))
            print('accuracy: ', round(accuracy*100, 4), '%')
    
    print(' ')
    prediction(test_set_x_orig, Y_test, W_1, b_1, W_2, b_2, activation)           # compute predictions and accuracy using testing dataset
    print(' ')

if __name__ == '__main__':
    main()