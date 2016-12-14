from __future__ import division
from resultsHelper import ResultsHelper
from constants import Constants
import math
import numpy as np
import scipy.sparse

class NeuralNetTest:
    def __init__(self, nearestOutputFile, testFile, hiddenCount, best = False):
        self.confusionMatrix = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
        self.outputFile = nearestOutputFile
        self.testFile = testFile
        self.hiddenCount = int(hiddenCount)
        if best:
            self.modelDirectory = 'nnet_best_model'
        else:
            self.modelDirectory = 'nnet_model'
        
    def readModel(self):
        w1 = np.loadtxt(self.modelDirectory+'/' + str(self.hiddenCount) + '_w1.txt')
        w2 = np.loadtxt(self.modelDirectory+'/' + str(self.hiddenCount) + '_w2.txt')
        b1 = np.loadtxt(self.modelDirectory+'/' + str(self.hiddenCount) + '_b1.txt').reshape((-1,1))
        b2 = np.loadtxt(self.modelDirectory+'/' + str(self.hiddenCount) + '_b2.txt').reshape((-1,1))
        with open(self.modelDirectory+'/' + str(self.hiddenCount) + "_activations.txt", "r") as file:
            activations = file.read().split()
        return { 'w1': w1, 'b1': b1, 'w2': w2, 'b2': b2}, activations
  
    def softmax(self, z):
        return np.exp(z) / np.sum(np.exp(z))
  
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
        
    def getOrientation(self, x, model, activations):
        w1, b1, w2, b2 = model['w1'], model['b1'], model['w2'], model['b2']
        activation_hl, activation_ol = activations
        z2 = np.dot(w1,x) + b1 #hidden layer input
        a2 = self.activation(z2, activation_hl) #hidden layer activation
        z3 = np.dot(w2,a2) + b2 #output layer input
        a3 = self.activation(z3, activation_ol) #output value
        probs = self.softmax(a3)
        pred = np.argmax(probs,axis=0)
        return ( 0 if pred == 0 else Constants.NINETY if pred == 1 else Constants.ONE_EIGHTY if pred == 2 \
                  else Constants.TWO_SEVENTY )
              
    def classify(self):
        model, activations = self.readModel()
        print activations
        
        with open(self.testFile) as document:
            for image in document.read().split(Constants.NEW_LINE):
                if image == Constants.EMPTY_STRING:
                    break 
                imageList = image.split()
                testImg = [int(pixelValue) for pixelValue in imageList[Constants.TWO:]]
                predictedOrientation = self.getOrientation(np.reshape(testImg,(-1,1)),model,activations)
                ResultsHelper.updateConfidenceMatrix(int(imageList[Constants.ONE]), predictedOrientation, self.confusionMatrix)
                self.outputFile.write(imageList[0] + Constants.SPACE + str(predictedOrientation) + Constants.NEW_LINE)

    def displayResult(self):
        ResultsHelper.displayAccuracy(self.confusionMatrix)