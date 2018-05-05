'''
This file implements a multi layer neural network for a multiclass classifier with One Convolution layer and one Max pooling layer
This implementation assumes only one Convolution layer and one Max pooling layer

@name: Achyutha Sreenivasa Bharadwaj
@email: asbhara2@asu.edu
@asuid: 1213094192
CSE - 591 Introduction to Deep Learning in Visual Computing
Mini Project 3
March 2018
Python 3

Accuracy for training set is 40.167 %
Accuracy for testing set is 27.700 %
'''
import numpy as np
from load_dataset import mnist
import matplotlib.pyplot as plt
import pdb
import sys, ast
from sklearn.preprocessing import OneHotEncoder

onehot_encoder = OneHotEncoder(sparse=False)
convolution_params = {}
convolution_gradients = {}

def Max_pooling_forward(Alprev):
    m = Alprev.shape[0]
    f = 2
    Ht = Alprev.shape[1] - f + 1
    Wd = Alprev.shape[2] - f + 1
    Ch = Alprev.shape[3]
    Al = np.zeros((m, Ht, Wd, Ch))
    for i in range(m):
            for h in range(Ht):
                for w in range(Wd):
                    for c in range(Ch):
                        Al[i, h, w, c] = np.max(Alprev[i, h:h+f, w:w+f, c])
    cache = {}
    cache["Alprev"] = Alprev
    return Al, cache

def Max_pooling_backward(dAl, cache):
    f = 2
    Alprev = cache["Alprev"]
    dAlprev = np.zeros(Alprev.shape)
    for i in range(dAl.shape[0]):
            for h in range(dAl.shape[1]):
                for w in range(dAl.shape[2]):
                    for c in range(dAl.shape[3]):
                        a = Alprev[i, h:h+f, w:w+f, c]
                        x, y= np.unravel_index(np.argmax(a, axis=None), a.shape)
                        dAlprev[i, x, y, c] += dAl[i, h, w, c]
    return dAlprev

def convolution_for_a_slice(cslice, W, b):
    z = np.squeeze(np.sum(np.multiply(cslice, W)) + b)
    return z

def convolution_forward_prop():
    W = convolution_params["W"]
    b = convolution_params["b"]
    Alprev = convolution_params["Alprev"]
    m = Alprev.shape[0]
    f = 3
    Ht = Alprev.shape[1] - f + 1
    Wd = Alprev.shape[2] - f + 1
    Ch = 5                 
    Zl = np.zeros((m, Ht, Wd, Ch))
    for i in range(m):
        for h in range(Ht):
            for w in range(Wd):
                for c in range(Ch):                
                    Zl[i, h, w, c] = convolution_for_a_slice(Alprev[i, h:h+f, w:w+f, 0], W[:,:,0,c], b[:,:,0,c])
    return Zl

def convolution_backward_prop(dZl):
    f = 3
    W = convolution_params["W"]
    b = convolution_params["b"]
    Alprev = convolution_params["Alprev"]
    dAlprev = np.zeros(Alprev.shape)    
    dWl = np.zeros(W.shape)
    dbl = np.zeros(b.shape)
    for i in range(dZl.shape[0]):
        for h in range(dZl.shape[1]):
            for w in range(dZl.shape[2]):
                for c in range(dZl.shape[3]): 
                    dAlprev[i, h:h+f, w:w+f, 0] += W[:,:,0,c] * dZl[i, h, w, c]
                    dWl[:,:,0,c] += Alprev[i, h:h+f, w:w+f, 0] * dZl[i, h, w, c]
                    dbl[:,:,0,c] += dZl[i, h, w, c]
    convolution_gradients["dWl"] = dWl
    convolution_gradients["dbl"] = dbl
    convolution_gradients["dAlprev"] = dAlprev

def relu(Z):
    '''
    computes relu activation of Z

    Inputs: 
        Z is a numpy.ndarray (n, m)

    Returns: 
        A is activation. numpy.ndarray (n, m)
        cache is a dictionary with {"Z", Z}
    '''
    A = np.maximum(0,Z)
    cache = {}
    cache["Z"] = Z
    return A, cache

def relu_der(dA, cache):
    '''
    computes derivative of relu activation

    Inputs: 
        dA is the derivative from subsequent layer. numpy.ndarray (n, m)
        cache is a dictionary with {"Z", Z}, where Z was the input 
        to the activation layer during forward propagation

    Returns: 
        dZ is the derivative. numpy.ndarray (n,m)
    '''
    dZ = np.array(dA, copy=True)
    Z = cache["Z"]
    dZ[Z < 0] = 0
    return dZ

def linear(Z):
    '''
    computes linear activation of Z
    This function is implemented for completeness

    Inputs: 
        Z is a numpy.ndarray (n, m)

    Returns: 
        A is activation. numpy.ndarray (n, m)
        cache is a dictionary with {"Z", Z}
    '''
    A = Z
    cache = {}
    return A, cache

def linear_der(dA, cache):
    '''
    computes derivative of linear activation
    This function is implemented for completeness

    Inputs: 
        dA is the derivative from subsequent layer. numpy.ndarray (n, m)
        cache is a dictionary with {"Z", Z}, where Z was the input 
        to the activation layer during forward propagation

    Returns: 
        dZ is the derivative. numpy.ndarray (n,m)
    '''
    dZ = np.array(dA, copy=True)
    return dZ

def softmax_cross_entropy_loss(Z, Y=np.array([])):
    '''
    Computes the softmax activation of the inputs Z
    Estimates the cross entropy loss

    Inputs: 
        Z - numpy.ndarray (n, m)
        Y - numpy.ndarray (1, m) of labels
            when y=[] loss is set to []
    
    Returns:
        A - numpy.ndarray (n, m) of softmax activations
        cache -  a dictionary to store the activations later used to estimate derivatives
        loss - cost of prediction
    '''
    ### CODE DONE 
    loss = 0
    A = np.zeros((Z.shape))
    cache={}

    for i in range(0,Z.shape[1],1):
        maxZ = max(Z[:,i])
        C = np.sum(np.exp(Z[:,i] - maxZ))
        A[:,i] = np.exp(Z[:,i] - maxZ) / C
        if(Y.shape[0]>0):
            loss += (-np.log(A[int(Y[0,i]),i]))
        
    cache["A"] = A    
    return A, cache, float(loss/Z.shape[1])

def softmax_cross_entropy_loss_der(Y, cache):
    '''
    Computes the derivative of softmax activation and cross entropy loss

    Inputs: 
        Y - numpy.ndarray (1, m) of labels
        cache -  a dictionary with cached activations A of size (n,m)

    Returns:
        dZ - numpy.ndarray (n, m) derivative for the previous layer
    '''
    ### CODE DONE 
    Y_bar = onehot_encoder.fit_transform(Y.T)
    A = cache["A"]
    dZ = (A - Y_bar.T)
    return dZ

def initialize_multilayer_weights(net_dims):
    '''
    Initializes the weights of the multilayer network

    Inputs: 
        net_dims - tuple of network dimensions

    Returns:
        dictionary of parameters
    '''
    f = 3
    Ch = 5
    np.random.seed(0)
    numLayers = len(net_dims)
    parameters = {}
    for l in range(numLayers-1):
        parameters["W"+str(l+1)] = np.random.randn(net_dims[l+1], net_dims[l]) * 0.01
        parameters["b"+str(l+1)] = np.random.randn(net_dims[l+1], 1) * 0.01
    convolution_params["W"] = np.random.randn(f, f, 1, Ch)
    convolution_params["b"] = np.random.randn(1, 1, 1, Ch)
    
    #10pix*10pix after convolution layer becomes 8pix*8pix
    #8pix*8pix after max pooling layer becomes 7pix*7pix
    net_dims[0] = 245
    
    return parameters

def linear_forward(A, W, b):
    '''
    Input A propagates through the layer 
    Z = WA + b is the output of this layer. 

    Inputs: 
        A - numpy.ndarray (n,m) the input to the layer
        W - numpy.ndarray (n_out, n) the weights of the layer
        b - numpy.ndarray (n_out, 1) the bias of the layer

    Returns:
        Z = WA + b, where Z is the numpy.ndarray (n_out, m) dimensions
        cache - a dictionary containing the inputs A
    '''
    ### CODE DONE
    Z = np.dot(W, A) + b
    cache = {}
    cache["A"] = A
    return Z, cache

def layer_forward(A_prev, W, b, activation):
    '''
    Input A_prev propagates through the layer and the activation

    Inputs: 
        A_prev - numpy.ndarray (n,m) the input to the layer
        W - numpy.ndarray (n_out, n) the weights of the layer
        b - numpy.ndarray (n_out, 1) the bias of the layer
        activation - is the string that specifies the activation function

    Returns:
        A = g(Z), where Z = WA + b, where Z is the numpy.ndarray (n_out, m) dimensions
        g is the activation function
        cache - a dictionary containing the cache from the linear and the nonlinear propagation
        to be used for derivative
    '''
    Z, lin_cache = linear_forward(A_prev, W, b)
    if activation == "relu":
        A, act_cache = relu(Z)
    elif activation == "linear":
        A, act_cache = linear(Z)
    
    cache = {}
    cache["lin_cache"] = lin_cache
    cache["act_cache"] = act_cache
    return A, cache

def multi_layer_forward(X, parameters):
    '''
    Forward propgation through the layers of the network

    Inputs: 
        X - numpy.ndarray (n,m) with n features and m samples
        parameters - dictionary of network parameters {"W1":[..],"b1":[..],"W2":[..],"b2":[..]...}
    Returns:
        AL - numpy.ndarray (c,m)  - outputs of the last fully connected layer before softmax
            where c is number of categories and m is number of samples in the batch
        caches - a dictionary of associated caches of parameters and network inputs
    '''
    L = len(parameters)//2  
    A = X
    caches = []
    for l in range(1,L):  # since there is no W0 and b0
        A, cache = layer_forward(A, parameters["W"+str(l)], parameters["b"+str(l)], "relu")
        caches.append(cache)

    AL, cache = layer_forward(A, parameters["W"+str(L)], parameters["b"+str(L)], "linear")
    caches.append(cache)
    return AL, caches

def linear_backward(dZ, cache, W, b):
    '''
    Backward prpagation through the linear layer

    Inputs:
        dZ - numpy.ndarray (n,m) derivative dL/dz 
        cache - a dictionary containing the inputs A, for the linear layer
            where Z = WA + b,    
            Z is (n,m); W is (n,p); A is (p,m); b is (n,1)
        W - numpy.ndarray (n,p)
        b - numpy.ndarray (n, 1)

    Returns:
        dA_prev - numpy.ndarray (p,m) the derivative to the previous layer
        dW - numpy.ndarray (n,p) the gradient of W 
        db - numpy.ndarray (n, 1) the gradient of b
    '''
    A_prev = cache["A"]
    ## CODE DONE
    dW = np.dot(dZ, A_prev.T)
    db = np.sum(dZ, axis = 1, keepdims = True)
    dA_prev = np.dot(W.T, dZ)
    return dA_prev, dW, db

def layer_backward(dA, cache, W, b, activation):
    '''
    Backward propagation through the activation and linear layer

    Inputs:
        dA - numpy.ndarray (n,m) the derivative to the previous layer
        cache - dictionary containing the linear_cache and the activation_cache
        activation - activation of the layer
        W - numpy.ndarray (n,p)
        b - numpy.ndarray (n, 1)
    
    Returns:
        dA_prev - numpy.ndarray (p,m) the derivative to the previous layer
        dW - numpy.ndarray (n,p) the gradient of W 
        db - numpy.ndarray (n, 1) the gradient of b
    '''
    lin_cache = cache["lin_cache"]
    act_cache = cache["act_cache"]

    if activation == "relu":
        dZ = relu_der(dA, act_cache)
    elif activation == "linear":
        dZ = linear_der(dA, act_cache)
    dA_prev, dW, db = linear_backward(dZ, lin_cache, W, b)
    return dA_prev, dW, db

def multi_layer_backward(dAL, caches, parameters):
    '''
    Back propgation through the layers of the network (except softmax cross entropy)
    softmax_cross_entropy can be handled separately

    Inputs: 
        dAL - numpy.ndarray (n,m) derivatives from the softmax_cross_entropy layer
        caches - a dictionary of associated caches of parameters and network inputs
        parameters - dictionary of network parameters {"W1":[..],"b1":[..],"W2":[..],"b2":[..]...}

    Returns:
        gradients - dictionary of gradient of network parameters 
            {"dW1":[..],"db1":[..],"dW2":[..],"db2":[..],...}
    '''
    L = len(caches)  # with one hidden layer, L = 2
    gradients = {}
    dA = dAL
    activation = "linear"
    for l in reversed(range(1,L+1)):
        dA, gradients["dW"+str(l)], gradients["db"+str(l)] = \
                    layer_backward(dA, caches[l-1], \
                    parameters["W"+str(l)],parameters["b"+str(l)],\
                    activation)
        activation = "relu"
    convolution_gradients["dAl"] = dA
    return gradients

def classify(X, parameters):
    '''
    Network prediction for inputs X

    Inputs: 
        X - numpy.ndarray (n,m) with n features and m samples
        parameters - dictionary of network parameters 
            {"W1":[..],"b1":[..],"W2":[..],"b2":[..],...}
    Returns:
        YPred - numpy.ndarray (1,m) of predictions
    '''
    ### CODE DONE 
    # Forward propagate X using multi_layer_forward
    # Get predictions using softmax_cross_entropy_loss
    # Estimate the class labels using predictions
    convolution_params["Alprev"] = X
    conv_zl = convolution_forward_prop()
    conv_zl, conv_cache = relu(conv_zl)
    max_pool_Al, max_pool_cache = Max_pooling_forward(conv_zl)
    A0 = max_pool_Al.reshape((max_pool_Al.shape[0], max_pool_Al.shape[1]*max_pool_Al.shape[2]*max_pool_Al.shape[3]))
    Zl, caches = multi_layer_forward(A0.T, parameters)
    Zmax = np.amax(Zl, axis = 0, keepdims = True)
    C = np.sum(np.exp(Zl - Zmax), axis=0, keepdims = True)
    Al = np.exp(Zl - Zmax) / C  
    Ypred = np.argmax(Al, axis = 0)
    return Ypred

def update_parameters(parameters, gradients, epoch, learning_rate, decay_rate=0.0):
    '''
    Updates the network parameters with gradient descent

    Inputs:
        parameters - dictionary of network parameters 
            {"W1":[..],"b1":[..],"W2":[..],"b2":[..],...}
        gradients - dictionary of gradient of network parameters 
            {"dW1":[..],"db1":[..],"dW2":[..],"db2":[..],...}
        epoch - epoch number
        learning_rate - step size for learning
        decay_rate - rate of decay of step size - not necessary - in case you want to use
    '''
    alpha = learning_rate*(1/(1+decay_rate*epoch))
    L = len(parameters)//2
    ### CODE DONE 
    for l in range(1,L):  # since there is no W0 and b0
        parameters["W"+str(l)] -= alpha * gradients["dW"+str(l)]
        parameters["b"+str(l)] -= alpha * gradients["db"+str(l)]
        
    convolution_params["W"] -= alpha * convolution_gradients["dWl"]
    convolution_params["b"] -= alpha * convolution_gradients["dbl"]
    return parameters, alpha

def multi_layer_network(X, Y, net_dims, num_iterations=500, learning_rate=0.2, decay_rate=0.01):
    '''
    Creates the multilayer network and trains the network

    Inputs:
        X - numpy.ndarray (n,m) of training data
        Y - numpy.ndarray (1,m) of training data labels
        net_dims - tuple of layer dimensions
        num_iterations - num of epochs to train
        learning_rate - step size for gradient descent
    
    Returns:
        costs - list of costs over training
        parameters - dictionary of trained network parameters
    '''
    parameters = initialize_multilayer_weights(net_dims)
    convolution_params["Alprev"] = X
    costs = []
    epoch = 0
    for ii in range(num_iterations):
        conv_Zl = convolution_forward_prop()         
        conv_Zl, conv_cache = relu(conv_Zl)
        max_pool_Al, max_pool_cache = Max_pooling_forward(conv_Zl) 
        max_pool_shape = max_pool_Al.shape
        A0 = max_pool_Al.reshape((max_pool_Al.shape[0], max_pool_Al.shape[1]*max_pool_Al.shape[2]*max_pool_Al.shape[3]))
        ### CODE HERE
        # Forward Prop
        ## call to multi_layer_forward to get activations
        ## call to softmax cross entropy loss
        Zl, caches = multi_layer_forward(A0.T, parameters)
        Al, cache_al, loss = softmax_cross_entropy_loss(Zl, Y)

        # Backward Prop
        ## call to softmax cross entropy loss der
        ## call to multi_layer_backward to get gradients
        ## call to update the parameters
        dZl = softmax_cross_entropy_loss_der(Y, cache_al)
        gradients = multi_layer_backward(dZl, caches, parameters)
        max_pool_dAl = convolution_gradients["dAl"].T
        max_pool_dAl = max_pool_dAl.reshape(max_pool_shape)
        conv_dAl = Max_pooling_backward(max_pool_dAl, max_pool_cache)
        conv_dZl = relu_der(conv_dAl, conv_cache)        
        convolution_backward_prop(conv_dZl)
        parameters, alpha = update_parameters(parameters, gradients, epoch, learning_rate, decay_rate)
        costs.append(loss)
        if ii % 10 == 0:
            print("Cost at iteration %i is: %.05f, learning rate: %.05f" %(ii, loss, alpha)) 
            epoch += 1
    
    return costs, parameters

def main():
    '''
    Trains a multilayer network for MNIST digit classification (all 10 digits)
    To create a network with 1 hidden layer of dimensions 800
    Run the progam as:
        python deepMultiClassNetwork_starter.py "[245,800]"
    The network will have the dimensions [245,800,10]
    10 is the number of digits

    '''
    net_dims = [245,500]#ast.literal_eval( sys.argv[1] )
    net_dims.append(10) # Adding the digits layer with dimensionality = 10
    #print("Network dimensions are:" + str(net_dims))

    # getting the subset dataset from MNIST
    train_data, train_label, test_data, test_label = \
            mnist(ntrain=6000,ntest=1000,digit_range=[0,10])
    # initialize learning rate and num_iterations
    learning_rate = 0.2
    num_iterations = 200

    costs, parameters = multi_layer_network(train_data, train_label, net_dims, \
            num_iterations=num_iterations, learning_rate=learning_rate, decay_rate=0.01)
    
    # compute the accuracy for training set and testing set
    train_Pred = classify(train_data, parameters)
    test_Pred = classify(test_data, parameters)

    trAcc = np.sum(train_Pred == train_label)/train_label.shape[1]*100
    teAcc = np.sum(test_Pred == test_label)/test_label.shape[1]*100    
    print("Accuracy for training set is {0:0.3f} %".format(trAcc))
    print("Accuracy for testing set is {0:0.3f} %".format(teAcc))
    
    ### CODE HERE to plot costs
    plt.plot(costs)
    plt.ylabel('cost')
    plt.xlabel('iterations')
    plt.title('Convolution Multi Layer Multi class Classifier with Max pooling')
    plt.savefig('Convolution Multi_Layer_Multi_Class_Classifier_Max_pooling.jpg')

if __name__ == "__main__":
    main()