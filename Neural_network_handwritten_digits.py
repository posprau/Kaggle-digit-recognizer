# -*- coding: utf-8 -*-
"""
Created on Mon Oct  9 16:14:52 2017

@author: Peter Sprau
"""

''' 
Apply what was taught during Machine Learning Online Class on Coursera. 
Use a Neural Network on multi-class classification problem: Recognize 
handwritten digits from 0 to 9. Try different values for regularization and
size of hidden layer. Attempt to improve performance by creating
more training samples from the given sample set using rotation of the digits
and Gaussian blurring.
'''

# import all libraries used throughout the following program
import numpy as np
import csv
import matplotlib.pyplot as plt 
from scipy.optimize import minimize
import time
from scipy.ndimage.filters import gaussian_filter
from scipy.ndimage import rotate

'''
Define all functions used for this project.
'''

## ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++   
def load_data(filename):
    '''
    load_data(filename) uses the csv library to load the data from a csv file.
    Returns the data as a numpy array.
    '''
    # read all the lines from the csv file
    with open(filename,'r') as csvfile:
        data_iter = csv.reader(csvfile, 
                               delimiter = ',', 
                               quotechar = '"')
        data = [data for data in data_iter]
    # turn data into numpy array
    data_array = np.asarray(data)   
    # return the numpy array
    return data_array 	

## ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
def save_data(filename, pred):
    '''
    save_data(filename, pred) uses the csv library to write the predicted labels
	 for the multi-class classification problem into a csv file.
	 Returns a csv file, with header 'ImageId', 'Label' for the two columns, and 
	 each predicted label for every Image of the test set.
	 '''
    
    with open(filename,'w', newline='') as csvfile:
        data_writer = csv.writer(csvfile, 
                               delimiter = ',', 
                               quotechar = '"')
        data_writer.writerow(['ImageId', 'Label'])
        cc = 1
        for elem in pred:
            data_writer.writerow([cc, elem]) 
            cc += 1
            
## ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
def display_data(X):
    '''
	 display_data(X) takes as input the matrix X which contains the images as 
	 row vectors. Reshapes the row vectors into 2D matrices and plots them as
	 a 2D grid of grayscale images.
	 '''
     
    # Set example_width automatically if not passed in
    example_width = int(round( np.sqrt( np.shape(X)[1] ) ) )
    
    # Compute rows, cols
    m,n = np.shape(X)
    example_height = int(n / example_width)
    
    # Compute number of items to display
    display_rows = int(np.floor( np.sqrt(m) ))
    display_cols = int(np.ceil( m / display_rows ))
    
    # Between images padding
    pad = 1
    
    # Setup blank display
    xSize = pad + display_rows * (example_height + pad)
    ySize = pad + display_cols * (example_width + pad)
    display_array = -1* np.ones( (xSize,ySize) )
    
    # Copy each example into a patch on the display array
    curr_ex = 0
    
    for j in range(1, display_rows+1):
    
        for i in range(1, display_cols+1):
            
            if curr_ex >= m: 
                break 
            		            		
            # Get the max value of the current image
            max_val = max( abs( X[curr_ex, :] ) )
                 
            # get the start and end values for the current patch inside the display_array
            rowS = pad + (j - 1) * (example_height + pad)
            rowE = pad + (j - 1) * (example_height + pad) + example_height
            colS = pad + (i - 1) * (example_width + pad)
            colE = pad + (i - 1) * (example_width + pad) + example_width
            
            # reshape the current image from a vector into a matrix and normalize by its maximum value
            currImg = X[curr_ex, :].reshape( (example_height, example_width) ) / max_val
            # insert the current image into the display array
            display_array[ rowS:rowE, colS:colE] = currImg						
                   
            # increase patch counter by 1                 
            curr_ex = curr_ex + 1
    	 
        if curr_ex >= m: 
            break 
    	 
    ##################
    
    # Display the display_array as an image
    plt.figure()
    # set the colormap to gray
    plt.imshow(display_array, cmap = 'gray')
    # Do not show axis
    plt.axis('off')
    plt.show()

## ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
def sigmoid(z):
    '''
    sigmoid(z) Compute sigmoid function of input z (z can be scalar, vector, or matrix).
    Returns the sigmoid of z.
    '''
    # define and compute sigmoid function for z
    g = 1 /(1 + np.exp(-z))
    
	# return the sigmoid evaluated for z
    return g
    
## ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
def sigmoidGradient(z):
    '''
    sigmoidGradient(z) computes the gradient of the sigmoid function evaluated at z
	 Returns the gradient of the sigmoid function.
    '''
    
    # define and compute the gradient of the sigmoid function, make sure that z
    # is an array so that multiplication is elementwise and not matrix multiplication 
    aZ = np.array(z)
    g = sigmoid(aZ) * (1-sigmoid(aZ))
    
    return g

## ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
def randInitWeights(Lin,Lout):    
    '''
    randInitWeights(Lin, Lout) randomly initializes the weights used as 
	 starting point for neural network for a neural network layer with Lin 
	 inputs and Lout outputs.
	 Returns the matrix with random weights.
    '''
    # set eps based on number of inputs and outputs
    eps = np.sqrt(6) / (np.sqrt(Lin+Lout))
	 # create matrix with random weights using random numbers from a uniform distribution
    w = np.random.rand(Lout, Lin+1) *2 *eps - eps

    return w
    
## ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++   
def CostFunc(nn_params, input_layer_size, hidden_layer_size, num_labels, X, y, lamb):
    '''
    CostFunc(nn_params, input_layer_size, hidden_layer_size, num_labels, X, y, lamb)
	 computes the cost function for neural network with two layers.
    '''
    
    # reshape nn_params back into the Theta1 and Theta2 matrices of 2 layer 
    # neural network
    
    dummy1 = nn_params[ 0 : (input_layer_size+1)*hidden_layer_size]
    dummy2 = nn_params[ (input_layer_size+1)*hidden_layer_size : ]
    
    Theta1 = dummy1.reshape( (hidden_layer_size, input_layer_size+1) )
    Theta2 = dummy2.reshape( (num_labels, hidden_layer_size+1) )
    
    
    # Initialize some useful values
    (m, n) = X.shape
    # m: number of training examples
    # n: length of feature vector

    # turn y into a column vector
    y = y.reshape((m,1))
    
    ## implement the computation of the cost function

    # add ones for the bias unit to X
    X = np.concatenate( ( np.ones((m,1)),X) , axis=1)
    
    # write y into a matrix with dim m x k (k = number of labels)
    Y = np.reshape( (y==0), (m,1))
    for i in range(1,num_labels):
        yvec = np.reshape( (y==i), (m,1))
        Y = np.append(Y,yvec, axis=1)
        
    # compute the hidden layer and add ones for the bias unit 
    A2 = sigmoid( X.dot(Theta1.transpose()) )
    A2 = np.concatenate( ( np.ones((m,1)),A2) , axis=1)
	
    # compute the activation in the output layer
    A3 = sigmoid( A2.dot(Theta2.transpose()) )
	
    # compute the cost function without regularization    
    term1 = np.sum( Y * np.log(A3) + (1-Y) * np.log(1-A3) , axis=1)
    J = -1/m * np.sum( term1 )
    
    # add the term for regularization
    term2 = np.sum( np.sum( Theta1[:,1:]**2 ) )
    term3 = np.sum( np.sum( Theta2[:,1:]**2 ) )
    J = J + lamb / (2*m) * ( term2 + term3)
    
    # return the cost function, and use squeeze method to turn into number
    return J.squeeze()
    
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++    
def NNCostFunc(nn_params, input_layer_size, hidden_layer_size, num_labels, X, y, lamb):
    '''
    NNCostFunc(nn_params, input_layer_size, hidden_layer_size, num_labels, X, y, lamb)
	 computes the cost function for neural network with two layers. 
	 Returns the cost and gradient.
    '''
    
    # reshape nn_params back into the Theta1 and Theta2 matrices of 2 layer 
    # neural network
    
    dummy1 = nn_params[ 0 : (input_layer_size+1)*hidden_layer_size]
    dummy2 = nn_params[ (input_layer_size+1)*hidden_layer_size : ]
    
    Theta1 = dummy1.reshape( (hidden_layer_size, input_layer_size+1) )
    Theta2 = dummy2.reshape( (num_labels, hidden_layer_size+1) )
    
    
    # Initialize some useful values
    (m, n) = X.shape
    # m: number of training examples
    # n: length of feature vector

    # turn y into a column vector
    y = y.reshape((m,1))
    
    ## implement the computation of the cost function

    # add ones for the bias unit to X
    X = np.concatenate( ( np.ones((m,1)),X) , axis=1)
    
    # write y into a matrix with dim m x k (k = number of labels)
    Y = np.reshape( (y==0), (m,1))
    for i in range(1,num_labels):
        yvec = np.reshape( (y==i), (m,1))
        Y = np.append(Y,yvec, axis=1)
        
    # compute the hidden layer and add ones for the bias unit 
    A2 = sigmoid( X.dot(Theta1.transpose()) )
    A2 = np.concatenate( ( np.ones((m,1)),A2) , axis=1)
	
    # compute the activation in the output layer
    A3 = sigmoid( A2.dot(Theta2.transpose()) )
	
    # compute the cost function without regularization    
    term1 = np.sum( Y * np.log(A3) + (1-Y) * np.log(1-A3) , axis=1)
    J = -1/m * np.sum( term1 )
    
    # add the term for regularization
    term2 = np.sum( np.sum( Theta1[:,1:]**2 ) )
    term3 = np.sum( np.sum( Theta2[:,1:]**2 ) )
    J = J + lamb / (2*m) * ( term2 + term3)
 
	
	
	#====================================================================
    ## implement backpropagation
    
    # initialize the two matrices Delta1/2 as zero matrices				 
    Delta1 = np.zeros( (hidden_layer_size, input_layer_size + 1) )
    Delta2 = np.zeros( (num_labels, hidden_layer_size + 1) )
    
    for t in range(m):
    	
        # perform one forward propagation for the tth training example
        a1 = np.reshape(X[t, :], (n+1,1) )

        z2 = Theta1.dot(a1)

        a2 = sigmoid( z2 )

        # add a one for the bias unit (did not have to do that for a1 as
        # ones were already added to X
        a2 = np.insert(a2, 0, 1)
        a2 = a2.reshape( (len(a2),1) )

        z3 = Theta2.dot(a2)

        a3 = sigmoid( z3 )

        # perform backward propagation
        d3 = a3 - np.reshape(Y[t, :], (num_labels,1) )

        # add a one for the bias unit again
        z2 = np.insert(z2, 0, 1)
        z2 = z2.reshape( (len(z2),1) )

        d2 = np.dot(Theta2.transpose() , d3) * sigmoidGradient(z2)

        # throw away the first element
        d2 = d2[1:]
        # update the Delta matrices
        Delta2 = Delta2 + d3.dot(a2.transpose()) 
        
        Delta1 = Delta1 + d2.dot(a1.transpose())
        

    # gradient of theta without regularization
    Theta1_grad = 1/m * Delta1
    Theta2_grad = 1/m * Delta2	
    
    # add regularization term to the gradients of theta
    Theta1_grad[:, 1:] = Theta1_grad[:, 1:] + lamb/m * Theta1[:, 1:]
    Theta2_grad[:, 1:] = Theta2_grad[:, 1:] + lamb/m * Theta2[:, 1:]
        
    # =========================================================================
    
    # Unroll gradients
    dummy1 = Theta1_grad.reshape( ((input_layer_size+1)*hidden_layer_size,1) )
    dummy2 = Theta2_grad.reshape( ((hidden_layer_size+1)*num_labels,1) ) 
    grad = np.concatenate( (dummy1,dummy2),axis = 0 )

    # return the cost function, and use squeeze method to turn into number
    return J.squeeze(), grad.flatten() 
	
## ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
def predict(Theta1, Theta2, X):
    '''
    predict(Theta1, Theta2, X) predicts the label of an input given trained 
	 Theta1 and Theta2 obtained from Neural Network.
	 Returns the predicted labels for input features in X.
    '''
    # Get the dimensions of the data
    m,n = np.shape(X)
    # m: number of training examples
    # n: length of feature vector    
	
	# add ones for the bias unit
    X = np.concatenate( (np.ones((m,1)),X), axis = 1 )
   
	# compute output for the layers of neural network using the trained Theta1 and Theta2
    h1 = sigmoid( X.dot(Theta1.transpose()) )
    # add ones for bias unit
    h1 = np.concatenate( (np.ones((m,1)),h1), axis = 1 )
    h2 = sigmoid( h1.dot(Theta2.transpose()) )
    
	# find the indices corresponding to the maximum which is equivalent to the assigned label
    p = np.argmax(h2, axis = 1)
    
    return p

# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
def callbackFunc(theta):
    '''
    callbackFunc(theta) retrieves the theta trained for the current iteration
    and adds it to global variable costmatrix.
    '''
    global costmatrix
    theta = np.reshape(theta, (len(theta),1) )
    costmatrix = np.append(costmatrix, theta, axis=1)
    
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
def plotCostFunc(costmatrix,input_layer_size, hidden_layer_size, num_labels, X, y, lamb):
    '''
    plotCostFunc(costmatrix) plots the value of the cost function vs the 
    number of iterations the optimization algorithm has completed.
    '''
    k, Niter = np.shape(costmatrix)
    n = []
    J = []
    # compute the cost for each iteration
    for i in range(Niter):
        Cost = CostFunc(costmatrix[:,i], input_layer_size, hidden_layer_size, \
                     num_labels, X, y, lamb)
        n.append(i+1)
        J.append(Cost)
        
    plt.figure()
    plt.plot(n, J)
    plt.xlabel("Iterations")
    plt.ylabel("Cost J")
    
## ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
def createMoreSamples(X, y):
    '''
    createMoreSamples(X, y) "adds" samples by generating copies of the images 
    after either a Gaussian blur or a rotation was applied
    '''
    # Set example_width automatically if not passed in
    example_width = int(round( np.sqrt( np.shape(X)[1] ) ) )
    
    # Compute rows, cols
    m,n = np.shape(X)
    example_height = int(n / example_width)
    
    # add zeros for the new training samples to be created
    X = np.concatenate( (X, np.zeros( (2*m,n) ) ) , axis=0)
    # repeat the labels to be used for the new training samples
    y = np.concatenate( (y,y,y), axis=0)
    
    for i in range(m):
            
            # get the current image
            currImg = X[i, 0:n].reshape( (example_height, example_width) ) 
            
            ## rotate randomly by an angle between +(-)40 to +(-)20  degrees
            angle = np.random.randint(20, 40+1)
            angleSign = (np.random.randint(1, 3) - 1.5) / 0.5
            rotateImg = rotate(currImg, angleSign*angle, reshape = False)
            # add rotated image as a new sample
            X[m+i, :] = rotateImg.reshape( (1, n) )
            
            ## gaussian blur
            blurSize = np.random.randint(10, 16)
            gaussianImg = gaussian_filter(currImg, example_height/blurSize)
            # add blurred image as a new sample
            X[2*m+i, :] = gaussianImg.reshape( (1, n) )
             
    # plot the raw, rotated, and blurred image for one example to see the 
    # effect of image manipulation
    plt.figure('raw')
    plt.imshow(currImg, cmap = 'gray')
    
    plt.figure('rotated')
    plt.imshow(rotateImg, cmap = 'gray')
    	
    plt.figure('Gaussian Blur')
    plt.imshow(gaussianImg, cmap = 'gray')    
    
    return X,y

#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++    
def trainNetwork(input_layer_size, hidden_layer_size, num_labels, Xtrain, \
                 ytrain, Xcsv, ycsv, lamb, maxI):
    '''
    trainNetwork(input_layer_size, hidden_layer_size, num_labels, Xtrain, \
                 ytrain, Xcsv, ycsv, lamb, maxI)
    trains the neural network for the given input parameters, and prints the 
    accuracy for the training as well as cross-validation set. It also plots
    the costfunction vs number of iterations. The maxmimum number of iterations
    is set by maxI. Finally, it predicts the labels for the test set, and saves
    the result to a csv file for submission to Kaggle.
    '''
    global costmatrix
    
    # initialize the random weights for training the neural network
    initial_Theta1 = randInitWeights(input_layer_size, hidden_layer_size)
    initial_Theta2 = randInitWeights(hidden_layer_size, num_labels)

    # Unroll parameters into one row vector
    dummy1 = initial_Theta1.reshape( ((input_layer_size+1)*hidden_layer_size,1) )
    dummy2 = initial_Theta2.reshape( ((hidden_layer_size+1)*num_labels,1) ) 
    initial_nn_params = np.concatenate( (dummy1,dummy2),axis = 0 )

    
    # initialize a matrix to store the values of theta1 and theta2 
    costmatrix = initial_nn_params
    
    
    #  Run minimize to obtain the optimal theta
    #  This function will return theta, maxiter value determines how many iterations 
    # will be used to train the network
    nn_params = minimize(NNCostFunc, initial_nn_params, \
                     args = (input_layer_size, hidden_layer_size, num_labels ,\
                             Xtrain,ytrain,lamb), \
                     method = 'CG', jac = True, options = {'maxiter' : maxI}, \
                     callback = callbackFunc)
    
    # plot the cost function vs number of iterations
    plotCostFunc(costmatrix,input_layer_size, hidden_layer_size, num_labels, \
                 Xtrain, ytrain, lamb)
    plt.title(str(hidden_layer_size) + ' hidden layers')

    # reshape nn_params back into the Theta1 and Theta2 matrices of 2 layer 
    # neural network
    dummy1 = nn_params.x[ 0 : (input_layer_size+1)*hidden_layer_size]
    dummy2 = nn_params.x[ (input_layer_size+1)*hidden_layer_size : ]
    
    Theta1 = dummy1.reshape( (hidden_layer_size, input_layer_size+1) )
    
    Theta2 = dummy2.reshape( (num_labels, hidden_layer_size+1) )



    ## visualize what the neural network has "learned"
    display_data(Theta1[:,1:])
    plt.title(str(hidden_layer_size) + ' hidden layers')

    ## compute the predicted labels and compare to the provided labels for the 
    #  training data set
    predtrain = predict(Theta1, Theta2, Xtrain)
    resulttrain = np.mean(np.float64(predtrain == ytrain)) * 100
    print('\nTraining Set Accuracy for ' + str(hidden_layer_size) + \
          ' hidden layers: %f\n'% resulttrain)

    predcsv = predict(Theta1, Theta2, Xcsv)
    resultcsv = np.mean(np.float64(predcsv == ycsv)) * 100
    print('\nCross-validation Set Accuracy for ' + str(hidden_layer_size) + \
          ' hidden layers: %f\n'% resultcsv)
    # =========================================================================
    ## Predict labels for test data set using trained neural network
    # =========================================================================

    ## Load test data set and save prediction to submission file
    testData = load_data('test.csv')  # training data stored in arrays X, y
    XT = np.float64(testData[1:,:])
    predT = predict(Theta1, Theta2, XT)
    save_data('submit_PS_NN_' + str(hidden_layer_size) +'.csv', predT)


## ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
## ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
## ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# start a timer to check which part of the code takes the longest as well as 
# what difference it makes to run the optimization for more iterations.
start_time = time.time()
    
## Setup the parameters for the neural network and data set with exception of
## number of hidden layers
input_layer_size  = 784  # 28x28 Input Images of Digits
num_labels = 10          # 10 labels, from 0 to 9
                          
# Load Training Data
data = load_data('train.csv')  # training data stored in arrays X, y
# throw away the first row which contains string labels for the columns and 
# turn the data into floats
X = np.float64(data[1:,1:])
y = np.float64(data[1:,0])

# get the number of training samples
m = np.shape(X)[0]

# Randomly select 100 data points to display
rand_indices = np.random.permutation(m)
sel = X[rand_indices[0:100], :]
# display the randomly selected set of digits
display_data(sel)

# divide training set into training set and cross-validation (csv) set
# 2 thirds training set, 1/3 csv set
dI = 2*m//3
Xtrain = X[rand_indices[0:dI], :]
Xcsv = X[rand_indices[dI:], :]
ytrain = y[rand_indices[0:dI]]
ycsv = y[rand_indices[dI:]]


# create more training samples by manipulating the provided samples
Xtrain, ytrain = createMoreSamples(Xtrain, ytrain)
## ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# check how much time has passed
print("--- %s seconds ---" % (time.time() - start_time))
print()
## ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++


# train the neural network ====================================================
# different values for regularization to try
lambdaVector = [0.03, 0.1, 0.3, 1, 3, 10]
# set regularization parameter
lamb = lambdaVector[0]


# set the number of hidden layers
hlsVector = [9, 16, 36, 64, 81, 100]

for elem in hlsVector:
    
    hidden_layer_size = elem   # number of hidden units
    # train neural network and display the trained hidden layers. Print out the
    # accuracy for the training set and cross-validation set. maxI is the number
    # of iterations the optimization algorithm will run.
    maxI = 500
    trainNetwork(input_layer_size, hidden_layer_size, num_labels, Xtrain, \
                     ytrain, Xcsv, ycsv, lamb, maxI)
    
    ## ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    # check how much time has passed
    print("--- %s seconds ---" % (time.time() - start_time))
    print()
    ## ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

'''
## 10/22/2017 Note to myself:
## Adding more training samples by manipulating the given samples did not
## improve the performance. 
## -> Next step: Carefully inspect the samples in the training / cross-
## validation set that aren't identified correctly. This could give a hint
## on how to improve performance.
'''
