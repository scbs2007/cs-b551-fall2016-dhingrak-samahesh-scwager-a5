from __future__ import division
from collections import defaultdict
import math

from resultsHelper import ResultsHelper
from constants import Constants
from adaBoostTrain import AdaBoostTrain

class AdaBoostTest:
    def __init__(self, nearestOutputFile, testFile, adaBoostTrainObj):
        self.confusionMatrix = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
        self.outputFile = nearestOutputFile
        self.testFile = testFile
        self.model = adaBoostTrainObj
        '''
        self.imageIds = None
        self.vector = None
        self.totTrainImages = 0

        # For optimization - populating the values of the following just once:
        self.rangeListImage = []
        self.vectorLength = 0
        '''

    def classify(self):
        '''adaBoostTrain = AdaBoostTrain(self.stumpCount, self.processCorpusObj)
        adaBoostTrain.train() 
        '''
        with open(self.testFile) as document:
            for image in document.read().split(Constants.NEW_LINE):
                if image == Constants.EMPTY_STRING:
                    break 
                imageList = image.split()
                testImg = [int(pixelValue) for pixelValue in imageList[Constants.TWO:]]

                #TODO CLASSIFICATION
                predictedOrientation = 0
                ResultsHelper.updateConfidenceMatrix(int(imageList[Constants.ONE]), predictedOrientation, self.confusionMatrix)
                #ResultsHelper.displayAccuracy(self.confusionMatrix)
                self.outputFile.write(imageList[0] + Constants.SPACE + str(predictedOrientation) + Constants.NEW_LINE)
                #print "Found orientation for: ", imageList[0], ': ', str(predictedOrientation)

    def displayResult(self):
        ResultsHelper.displayAccuracy(self.confusionMatrix)

