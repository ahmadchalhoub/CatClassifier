# Author:   Ahmad Chalhoub
# Project:  Converting the full single-layer network into a deep network of L layers

import h5py
import numpy as np

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

# applies relu function on Z
def relu(Z):

    return np.maximum(0, Z)

# g' when using a relu activation function
def relu_prime(Z):

    return np.where(Z > 0, 1.0, 0.0)
    
# applies sigmoid function on Z
def sigmoid(Z):
    A = 1/(1 + np.exp(-Z))

    return A

# g' when using a sigmoid activation function
def sigmoid_prime(Z):

    return (sigmoid(Z) * (1 - sigmoid(Z)))

# initializes the parameters (W and b values)
def initialize_parameters(layer_dims):
    weights = {}
    biases = {}
    
    for i in range(1, len(layer_dims)):
        weights["W_{}".format(i)] = np.random.randn(layer_dims[i], layer_dims[i-1]) / np.sqrt(layer_dims[i-1])
        biases["b_{}".format(i)] = np.zeros((layer_dims[i], 1))

    return weights, biases

# performs the linear calculations of forward propagation
def forward_linear(A_prev, W, b):
    Z = np.dot(W, A_prev) + b
    cache = (A_prev, W, b)
    
    return Z, cache

# performs the activation calculations of the forward propagation
def forward_activation(A_prev, W, b, activation):
    Z, linear_cache = forward_linear(A_prev, W, b)

    if activation == 'relu':
        A = relu(Z)
    else:
        A = sigmoid(Z)

    # cache = A_prev, W, b, Z
    cache = (linear_cache, Z)                           

    return A, cache

# performs overall forward propagation steps
def forward_prop(weights, biases, X, layer_dims):
    length = len(layer_dims) - 1
    all_caches = []

    A_prev = X 

    activation = 'relu'
    for i in range(1, length):
        A, cache = forward_activation(A_prev, weights["W_{}".format(i)], biases["b_{}".format(i)], activation)
        all_caches.append(cache)
        A_prev = A
    
    activation = 'sigmoid'
    A_final, cache = forward_activation(A_prev, weights["W_{}".format(length)], biases["b_{}".format(length)], activation)
    all_caches.append(cache)

    return A_final, all_caches

# calculates cost
def calculate_cost(A_final, Y_real):
    m = Y_real.shape[1]
    L = (Y_real *np.log(A_final)) + (1-Y_real)*np.log(1-A_final)
    cost = -(1/m)*np.sum(L)

    return cost

# performs the linear calculations of backward propagation
def backward_linear(dZ, cache):
    A_prev, W, b = cache
    m = A_prev.shape[1]
    dW = (1/m)*np.dot(dZ, A_prev.T)
    db = (1/m)*np.sum(dZ, axis=1)
    dA_prev = np.dot(W.T, dZ)

    return dA_prev, dW, db

# performs the activation calculations of the backward propagation
def backward_activation(dA, activation, cache):
    linear_cache, Z = cache
    if activation == 'relu':
        dZ = dA*relu_prime(Z)
    else:
        dZ = dA*sigmoid_prime(Z)
    
    dA_prev, dW, db = backward_linear(dZ, linear_cache)

    return  dA_prev, dW, db


# performs overall backward propagation steps
def back_prop(X, A_final, Y, layer_dims, cache):
    length = len(layer_dims) - 1

    # initialize backpropagation
    dA = {}
    dW = {}
    db = {}
    dA_final = -(np.divide(Y, A_final) - np.divide(1-Y, 1-A_final))

    # use backward_activation function and the cached elements of the last layer to get dW_L, db_L, and dA_(L-1)
    dA["dA_{}".format(length-1)], dW["dW_{}".format(length)], db["db_{}".format(length)] = backward_activation(dA_final, 'sigmoid', cache[length-1])

    # i = 3 -> 2 -> 1 -> 0 (for L = 5)
    for i in reversed(range(length-1)):
        dA["dA_{}".format(i)], dW["dW_{}".format(i+1)], db["db_{}".format(i+1)] = backward_activation(dA["dA_{}".format(i+1)], 'relu', cache[i])
        db["db_{}".format(i+1)] = np.reshape(db["db_{}".format(i+1)], (db["db_{}".format(i+1)].shape[0], 1))

    return dW, db


# updates the parameters (weights and biases) of the network
def update_parameters(weights, biases, dW, db, lr, layer_dims):

        # loop through weights and biases of all layers and update them
        for i in range(1, len(layer_dims)):
            weights["W_{}".format(i)] -= lr * dW["dW_{}".format(i)]
            biases["b_{}".format(i)] -= lr * db["db_{}".format(i)]

        return weights, biases


# calculate the training & testing accuracies
def prediction(test_set_x_orig, Y_test, X, Y_train, weights, biases, layer_dims):
    X_test = 1/255*(np.reshape(test_set_x_orig, (test_set_x_orig.shape[0], test_set_x_orig[-1].flatten().shape[0])).T)

    # training accuracy
    A_final_predict, caches = forward_prop(weights, biases, X, layer_dims) 
    predictions_test = np.round(A_final_predict)
    testing_accuracy = np.mean(predictions_test==Y_train)
    print('Training accuracy = ', testing_accuracy*100, ' %' )

    # testing accuracy
    A_final_predict, caches = forward_prop(weights, biases, X_test, layer_dims) 
    predictions_test = np.round(A_final_predict)
    testing_accuracy = np.mean(predictions_test==Y_test)
    print('Testing accuracy = ', testing_accuracy*100, ' %' )

    
def main():
    np.random.seed(10)
    lr = 0.02               # learning rate

    # load dataset
    train_set_x_orig, Y_train, test_set_x_orig, Y_test, classes = load_dataset()
    X = create_input_matrix(train_set_x_orig)

    # declare and initialize array with network layers
    layer_dims = (X.shape[0], 22, 10, 7, 5, 3, 1)
        
    print('I am using ', len(layer_dims)-1, ' layers ', layer_dims[1:], ' and a learning rate = ', lr)

    # initialize weights and biases for all layers
    weights, biases = initialize_parameters(layer_dims)

    # training loop
    for i in range(3000):
        A_final, all_caches = forward_prop(weights, biases, X, layer_dims)
        dW, db = back_prop(X, A_final, Y_train, layer_dims, all_caches)
        weights, biases = update_parameters(weights, biases, dW, db, lr, layer_dims)
        if i % 200 == 0:
            cost = calculate_cost(A_final, Y_train)
            print('The cost on iteration ', i, ' is = ', cost)

    # calculate predictions/accuracies of trained model           
    prediction(test_set_x_orig, Y_test, X, Y_train, weights, biases, layer_dims)


if __name__ == '__main__':
    main()
    