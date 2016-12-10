from __future__ import division
from resultsHelper import ResultsHelper
from constants import Constants
import math
import numpy as np
import scipy.sparse
import matplotlib.pyplot as plt

'''
http://www.wildml.com/2015/09/implementing-a-neural-network-from-scratch/
#http://iamtrask.github.io/2015/07/12/basic-python-network/
http://ufldl.stanford.edu/tutorial/supervised/MultiLayerNeuralNetworks/
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
        
    def saveModel(self, model):
        w1, b1, w2, b2 = model['w1'], model['b1'], model['w2'], model['b2']
        np.savetxt('w1.txt', w1)
        np.savetxt('w2.txt', w2)
        np.savetxt('b1.txt', b1)
        np.savetxt('b2.txt', b2)
    
    def readModel(self):
        w1 = np.loadtxt('w1.txt')
        w2 = np.loadtxt('w2.txt')
        b1 = np.loadtxt('b1.txt')
        b2 = np.loadtxt('b2.txt')
        return { 'w1': w1, 'b1': b1, 'w2': w2, 'b2': b2}
  
    def oneHotIt(self, Y):
        m = Y.shape[0]
        OHX = scipy.sparse.csr_matrix((np.ones(m), (Y, np.array(range(m)))))
        OHX = np.array(OHX.todense())
        return OHX
        
    def oneHotIt_angle(self, imgIds):
        angles = np.array([angle//90 for (id, angle) in imgIds.values()])
        return self.oneHotIt(angles)
        
    def softmax(self, z):
        z -= np.max(z)
        return np.exp(z) / np.sum(np.exp(z))

    def activation_delta(self, a, type):
        if type == "sigmoid":
            return a*(1-a)
        elif type == "tanh":
            return 1 - np.power(a, 2)
        else:
            raise ValueError("unknown activation function")
  
    def activation(self, z, type):
        if type == "sigmoid":
            return 1 / ( 1 + np.exp(-z) )
        elif type == "tanh":
            return np.tanh(z)
        else:
            raise ValueError("unknown activation function")

    def getPreds(self, x, model):
        w1, b1, w2, b2 = model['w1'], model['b1'], model['w2'], model['b2']
        z2 = np.dot(x,w1) + b1 #hidden layer input
        a2 = activation(z2, "tanh") #hidden layer activation
        z3 = np.dot(a2,w2) + b2 #output layer input
        scores = activation(z3, "sigmoid")
        probs = softmax(scores)
        preds = np.argmax(probs,axis=1)
        return preds
        
    def getAccuracy(self, X, y, model):
        preds = getPreds(self, x, model)
        accuracy = sum(preds == y)/len(y)
        return accuracy
        
    def getAccuracy_angle(self, X, imgIds, model):
        y = np.array([angle//90 for (id, angle) in imgIds.values()])
        return self.getAccuracy(self, X, y, model)

    def getLoss_test(w,x,y,lam,counter):
        m = x.shape[0] #First we get the number of training examples
        y_mat = oneHotIt(y) #Next we convert the integer class coding into a one-hot representation
        scores = np.dot(x,w) #Then we compute raw class scores given our input and current weights
        prob = softmax(scores) #Next we perform a softmax on these scores to get their probabilities
        loss = - np.sum(y_mat * np.log(prob)) + (lam/2)*np.sum(w*w) #We then find the loss of the probabilities
        grad = - np.dot(x.T,(y_mat - prob)) + lam*w #And compute the gradient for that loss
        return loss,grad
              
    def getLoss(self, output,y,dim):
        prob = self.softmax(output) #get probability distribution
        return np.sum( np.power(prob - y,2) ) / dim
        
    def build_model(self, X, Y, nn_hdim, iterations=10000):

        prt = False
        learningRate = 0.01 # learning rate for gradient descent
        reg_lambda = 0.00 # regularization strength
        nn_output_dim = Y.shape[0]
        m_inv = 1/X.shape[1] #inverse of number of examples for scaling

        #model weights and biases
        w1 = 0.001*np.random.random( (nn_hdim, X.shape[0]) ) - 0.0005
        w2 = 0.001*np.random.random( (nn_output_dim, nn_hdim) ) - 0.0005
        b1 = np.zeros( (nn_hdim, 1) )
        b2 = np.zeros( (nn_output_dim, 1) )
        
        #gradients of weights and biases
        dw1 = np.zeros( (nn_hdim, X.shape[0]) ) 
        dw2 = np.zeros( (nn_output_dim, nn_hdim) )
        db1 = np.zeros( (nn_hdim, 1) )
        db2 = np.zeros( (nn_output_dim, 1) )
        
        losses = np.zeros(iterations)
        
        for i in range(iterations):
            #choose a random example
            ind = np.random.randint(X.shape[1])
            x, y = X[:,[ind]], Y[:,[ind]]
#             print x.shape, y.shape
    
            #Forward propagation
            if prt:
                print "w1"
                print w1[1:5,1:5]
                print "w2"
                print w2[1:5,1:5]
            z2 = np.dot(w1,x) + b1 #hidden layer input
            if prt:
                print "z2"
                print z2.shape
                print z2[1:5,:]
            a2 = self.activation(z2, "sigmoid") #hidden layer activation
            if prt:
                print "a2"
                print a2.shape
                print a2[1:5,:]
            z3 = np.dot(w2,a2) + b2 #output layer input
            if prt:
                print "z3"
                print z3.shape
                print z3[1:5,:]
            a3 = self.activation(z3, "sigmoid") #output value
            if prt:
                print "a3"
                print a3.shape
            
            err = y - a3
            if prt:
                print "a3"
                print a3[1:5,:]
                print "err"
                print err[1:5,:]
            loss = self.getLoss(a3,y,nn_output_dim) #sum of squared differences
            losses[i] = loss
            if i % 1000 == 0 or i <= 10: print ("i = ", i, "loss = ", loss)

            #Backpropagation
            nl_delta = - err * self.activation_delta(a3, "sigmoid") #output layer delta: a(1-a) for sigmoid, 1-a**2 for tanh
            if prt:
                print "nl"
                print nl_delta
            l2_delta = np.dot(w2.T, nl_delta) * self.activation_delta(a2, "sigmoid") #hidden layer delta #transpose nl_delta?
            if prt:
                print "l2 delta"
                print l2_delta[1:5,:]
            dw2 = np.dot(nl_delta, a2.T)
            dw1 = np.dot(l2_delta, x.T)
#             db2 = np.sum( nl_delta, axis = 0, keepdims = True)
#             db1 = np.sum( l2_delta, axis = 0, keepdims = True)
            db2 = nl_delta
            db1 = l2_delta
        
            #Gradient descent parameter update
            w1 = w1 - learningRate * dw1 - reg_lambda * w1
            w2 = w2 - learningRate * dw2 - reg_lambda * w2
            b1 -= learningRate * db1
            b2 -= learningRate * db2
        
            learningRate *= 0.9999
#             lerningRate = 1 - i*0.00005
        
        model = { 'w1': w1, 'b1': b1, 'w2': w2, 'b2': b2}
        return model, losses

    def classify(self):
        self.imageIds = self.processCorpusObj.getImageIds()
        self.vector = self.processCorpusObj.getVector()
        self.vectorLength = len(self.vector)
        X = np.array(self.vector).T #192 x N
        y = self.oneHotIt_angle(self.imageIds) #4 x N
        
        iterations = 1000000
        model, losses = self.build_model(X, y, self.hiddenCount, iterations = iterations)
        plt.plot(losses)
        self.saveModel(model)
        accuracy = getAccuracy_angle(self, X, self.imageIds, model)
        print ('Average Training Accuracy: ', accuracy)
        quit()

        with open(self.testFile) as document:
            for image in document.read().split(Constants.NEW_LINE):
                if image == Constants.EMPTY_STRING:
                    break 
                imageList = image.split()
                testImg = [int(pixelValue) for pixelValue in imageList[Constants.TWO:]]

                #TODO CLASSIFICATION

                ResultsHelper.updateConfidenceMatrix(int(imageList[Constants.ONE]), predictedOrientation, self.confusionMatrix)
                #ResultsHelper.displayAccuracy(self.confusionMatrix)
                self.outputFile.write(imageList[0] + Constants.SPACE + str(predictedOrientation) + Constants.NEW_LINE)
                #print "Found orientation for: ", imageList[0], ': ', str(predictedOrientation)

    def displayResult(self):
        ResultsHelper.displayAccuracy(self.confusionMatrix)