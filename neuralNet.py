from __future__ import division
from resultsHelper import ResultsHelper
from constants import Constants
import math
import numpy as np
import scipy.sparse

'''
NEURAL NETWORK ANALYSIS
Disclaimer: This code is based on a program I wrote for a research project, consulting the sources listed below.
The code from the sources has been thoroughly modified.

http://www.wildml.com/2015/09/implementing-a-neural-network-from-scratch/
http://iamtrask.github.io/2015/07/12/basic-python-network/
http://ufldl.stanford.edu/tutorial/supervised/MultiLayerNeuralNetworks/
http://axon.cs.byu.edu/papers/Wilson.nn03.batch.pdf
http://cs231n.github.io/neural-networks-case-study/#linear
https://gist.github.com/yusugomori/cf7bce19b8e16d57488a

Neural network:
Input is the raw data: each x-values is a 192D vector of rgb values. They are not processed.
The output is a oneHotIt matrix of the four possible rotations.
Weights are set to small values uniformly distributed around 0.
The output activation function is softmax, while the hidden layer can be sigmoid, tanh, relu, or softmax.

The first two parameters tested for were the dimension of the hidden layer and the hidden layer activation function. 
Having fixed the following parameters:
    Number of iterations = 7*1e6
    Step size for iteration i = 0.01 / (1 + 100*i/iterations) * (1/m), m is number of examples
Testing values ranging from 50 to 300 with increments of 50, for both reLU
--> The results were best between 150 and 200, near the dimensionality of the data, with testing accuracy of 71.79% (best: ReLU nn_hdim = 200)
Using tanh or or reLU as the hidden layer activation function gave test accuracy with a difference less than .5%.

ReLU was chosen for two further tests:
(1) check the effect of decreasing the step size 100 times less. The new function was:
    Step size for iteration i = 0.01 / (1 + i/iterations) * (1/m), m is number of examples
    Number of iterations = 7*1e6
--> Training accuracy increased from 70-72% to 76-77%.
--> Testing accuracy increased from 70-72% to 74-76%.
--> We also tested for smaller hidden layer dimensions: 10 and 25, noticing that the results were no worse than those of higher dimensions.
This result is in line with the fact that adaBoost reached its best performance using only 20 features. We assume there is much redundancy in 
the image data and all learning algorithms work efficiently with the key features.

(2) check the effect of adding many more iterations
    Number of iterations = 1e7:
    Step size for iteration i = 0.01 / (1 + i/iterations) * (1/m), m is number of examples
--> Training accucary increased from 76-77% to 77-78%.
--> Testing accuracy increased from 74-76% to 75-77%

The overall best result of 76.44% testing accuracy was achieved with 1e7 iterations, 250 hidden nodes, and step size function i = 0.01 / (1 + i/iterations) * (1/m).
'''

class NeuralNet:
    def __init__(self, nearestOutputFile, testFile, hiddenCount, processCorpusObj):
        self.confusionMatrix = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
        self.outputFile = nearestOutputFile
        self.testFile = testFile
        self.hiddenCount = int(hiddenCount)
        self.processCorpusObj = processCorpusObj
        self.imageIds = None
        self.vector = None
        self.modelDirectory = 'nnet_model'
        
    def saveModel(self, model, activation_hl, activation_ol, losses, prediction_error):
        w1, b1, w2, b2 = model['w1'], model['b1'], model['w2'], model['b2']
        np.savetxt(self.modelDirectory+'/' + str(self.hiddenCount) + '_w1.txt', w1)
        np.savetxt(self.modelDirectory+'/' + str(self.hiddenCount) + '_w2.txt', w2)
        np.savetxt(self.modelDirectory+'/' + str(self.hiddenCount) + '_b1.txt', b1)
        np.savetxt(self.modelDirectory+'/' + str(self.hiddenCount) + '_b2.txt', b2)
        with open(self.modelDirectory+'/' + str(self.hiddenCount) + "_activations.txt", "w") as text_file:
            text_file.write(activation_hl + " " + activation_ol)
        np.savetxt(self.modelDirectory+'/' + str(self.hiddenCount) + '_losses.txt', losses)
        np.savetxt(self.modelDirectory+'/' + str(self.hiddenCount) + '_prediction_error.txt', prediction_error)
        
    def normalize(self, X):
        return X
#         return np.log(X+1) / np.log(255)
  
    def oneHotIt(self, Y):
        m = Y.shape[0]
        OHX = scipy.sparse.csr_matrix((np.ones(m), (Y, np.array(range(m)))))
        OHX = np.array(OHX.todense())
        return OHX
        
    def oneHotIt_angle(self, imgIds):
        angles = np.array([imgIds[str(i)][1]//90 for i in range(len(imgIds))])
        return self.oneHotIt(angles)
        
    def softmax_batch(self, z):
        return (np.exp(z).T / np.sum(np.exp(z),axis=1)).T

    def softmax(self, z):
        return np.exp(z) / np.sum(np.exp(z))

    def activation_delta(self, a, type):
        if type == "sigmoid":
            return a*(1-a)
        elif type == "tanh":
            return 1 - np.power(a, 2)
        elif type == "relu":
            return 1. * (a > 0)
        elif type == "softmax":
            return 1
        else:
            raise ValueError("unknown activation function")
  
    def activation(self, z, type):
        if type == "sigmoid":
            return 1 / ( 1 + np.exp(-z) )
        elif type == "tanh":
            return np.tanh(z)
        elif type == "relu":
            return z * (z > 0)
        elif type == "softmax":
            return self.softmax(z)
        else:
            raise ValueError("unknown activation function")
        
    def getLoss_training(self, x, y, w1, w2, b1, b2, activation_hl, activation_ol):
        '''calculate loss for all examples combined in a batch'''
        z2 = np.dot(w1,x) + b1 #hidden layer input
        a2 = self.activation(z2, activation_hl) #hidden layer activation
        z3 = np.dot(w2,a2) + b2 #output layer input
        a3 = self.activation(z3, activation_ol) #output value
        err = y - a3
        if activation_ol == "softmax":
            probs = a3
        else:
            probs = self.softmax_batch(a3)
        loss = - np.sum(y * np.log(probs))  #We then find the loss of the probabilities
        '''loss = np.sum(np.abs(err)) / (err.shape[0] * err.shape[1]) #squared error. not desired here?''' 
        preds = np.argmax(probs,axis=0) #class predictions
        predict_error = 0
        for i in range(len(preds)):
            predict_error += 1 if y[preds[i],i] != 1 else 0
        predict_error /= len(preds)
        return loss, predict_error
        
    def build_model(self, X, Y, nn_hdim, activation_hl = "tanh", activation_ol = "softmax", iterations=10000):

        learningRate = 0.01 # learning rate for gradient descent
        reg_lambda = 0.00 # regularization strength
        nn_output_dim = Y.shape[0]
        m_inv = 1/X.shape[1] #inverse of number of examples
        losses = [] #store losses over time for plot
        prediction_errors = []

        #model weights and biases
        w1 = 0.001*np.random.random( (nn_hdim, X.shape[0]) ) - 0.0005
        w2 = 0.001*np.random.random( (nn_output_dim, nn_hdim) ) - 0.0005
        b1 = np.zeros( (nn_hdim, 1) )
        b2 = np.zeros( (nn_output_dim, 1) )
        
        for i in range(iterations):
            #choose a random x, y pair
            ind = np.random.randint(X.shape[1])
            x, y = X[:,[ind]], Y[:,[ind]]
    
            #Forward propagation
            z2 = np.dot(w1,x) + b1 #hidden layer input
            a2 = self.activation(z2, activation_hl) #hidden layer activation
            z3 = np.dot(w2,a2) + b2 #output layer input
            a3 = self.activation(z3, activation_ol) #output value
            err = y - a3
            
            #loss and prediction error calculation over all examples in batch
            if i % 1000 == 0: 
                loss, predict_error = self.getLoss_training(X, Y, w1, w2, b1, b2, activation_hl, activation_ol) 
                #print "iter: ", i, "loss and prediction error for batch: ", loss, predict_error
                losses.append(loss)
                prediction_errors.append(predict_error)

            #Backpropagation
            nl_delta = - err * self.activation_delta(a3, activation_ol) #output layer delta: a(1-a) for sigmoid, 1-a**2 for tanh, 1 for softmax
            l2_delta = np.dot(w2.T, nl_delta) * self.activation_delta(a2, activation_hl)
            dw2 = np.dot(nl_delta, a2.T)
            dw1 = np.dot(l2_delta, x.T)
            db2 = nl_delta
            db1 = l2_delta
        
            #Gradient descent parameter update
            w1 -= m_inv * learningRate * dw1 - reg_lambda * w1
            w2 -= m_inv * learningRate * dw2 - reg_lambda * w2
            b1 -= m_inv * learningRate * db1
            b2 -= m_inv * learningRate * db2
            learningRate = 0.01 / (1 + i/iterations)
        
        model = { 'w1': w1, 'b1': b1, 'w2': w2, 'b2': b2}
        return model, losses, prediction_errors

    def train(self):
        self.imageIds = self.processCorpusObj.getImageIds()
        self.vector = self.processCorpusObj.getVector()
        self.vectorLength = len(self.vector)
        X = self.normalize(np.array(self.vector).T) #192 x N
        y = self.oneHotIt_angle(self.imageIds) #4 x N
        activation_hl, activation_ol = "tanh", "softmax"
        iterations = int(7*1e6)
        model, losses, prediction_error = self.build_model(X, y, self.hiddenCount, activation_hl, activation_ol, iterations = iterations)
        _, predict_error = self.getLoss_training(X, y, model['w1'], model['w2'], model['b1'], model['b2'], activation_hl, activation_ol)
        print "activation functions used: ", activation_hl, activation_ol
        print "Prediction accuracy for training: ", 1 - predict_error
        self.saveModel(model, activation_hl, activation_ol, losses, prediction_error)

