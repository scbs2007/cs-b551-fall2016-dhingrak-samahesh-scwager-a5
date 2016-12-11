from __future__ import division
from resultsHelper import ResultsHelper
from constants import Constants
import math
import numpy as np
import scipy.sparse

class NeuralNetTest:
    def __init__(self, nearestOutputFile, testFile):
        self.confusionMatrix = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
        self.outputFile = nearestOutputFile
        self.testFile = testFile
    
    def readModel(self):
        w1 = np.loadtxt('w1.txt')
        w2 = np.loadtxt('w2.txt')
        b1 = np.loadtxt('b1.txt')
        b2 = np.loadtxt('b2.txt')
        activations = np.loadtxt('activation_functions.txt')
        return { 'w1': w1, 'b1': b1, 'w2': w2, 'b2': b2}, activations
  
    def activation(self, z, type):
        if type == "sigmoid":
            return 1 / ( 1 + np.exp(-z) )
        elif type == "tanh":
            return np.tanh(z)
        elif type == "softmax":
            return self.softmax(z)
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
        
    def getOrientation(self, x, model, activations):
        w1, b1, w2, b2 = model['w1'], model['b1'], model['w2'], model['b2']
        activation_hl, activation_ol = activations
        z2 = np.dot(x,w1) + b1 #hidden layer input
        a2 = activation(z2, activation_hl) #hidden layer activation
        z3 = np.dot(a2,w2) + b2 #output layer input
        scores = activation(z3, activation_ol)
        probs = softmax(scores)
        pred = np.argmax(probs,axis=1)
        return ( 0 if pred == 0 else Constants.NINETY if pred == 1 else Constants.ONE_EIGHTY if pred == 2 \
                  else Constants.TWO_SEVENTY )
              
    def classify(self):
        model, (activation_hl, activation_ol) = self.readModel()
        
        with open(self.testFile) as document:
            for image in document.read().split(Constants.NEW_LINE):
                if image == Constants.EMPTY_STRING:
                    break 
                imageList = image.split()
                testImg = [int(pixelValue) for pixelValue in imageList[Constants.TWO:]]
                predictedOrientation = self.getOrientation(testImg)
                ResultsHelper.updateConfidenceMatrix(int(imageList[Constants.ONE]), predictedOrientation, self.confusionMatrix)
                self.outputFile.write(imageList[0] + Constants.SPACE + str(predictedOrientation) + Constants.NEW_LINE)

    def displayResult(self):
        ResultsHelper.displayAccuracy(self.confusionMatrix)