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
        self.stumps0 = adaBoostTrainObj.allStumps0
        self.stumps90 = adaBoostTrainObj.allStumps90
        self.stumps180 = adaBoostTrainObj.allStumps180
        self.stumps270 = adaBoostTrainObj.allStumps270
        '''
        self.imageIds = None
        self.vector = None
        self.totTrainImages = 0

        # For optimization - populating the values of the following just once:
        self.rangeListImage = []
        self.vectorLength = 0
        '''

    def getVote(self, testImg, stumps):
        positive = 0
        for eachStump in stumps:
            pix1, pix2, alpha = eachStump.getStumpProperties()
            if testImg[pix1] > testImg[pix2]:
                positive += alpha
        return positive
            
    def getOrientation(self, testImg):
        orientation = 0
        vote = self.getVote(testImg, self.stumps0)
        
        nextVote = self.getVote(testImg, self.stumps90)
        if nextVote > vote:
            vote = nextVote
            orientation = Constants.NINETY 

        nextVote = self.getVote(testImg, self.stumps180)
        if nextVote > vote:
            vote = nextVote
            orientation = Constants.ONE_EIGHTY

        nextVote = self.getVote(testImg, self.stumps270)
        if nextVote > vote:
            vote = nextVote
            orientation = Constants.TWO_SEVENTY
        return orientation
       
    def displayStumps(self):
        print "0: "
        for eachStump in self.stumps0:
            print eachStump.getStumpProperties()
        print "90: "
        for eachStump in self.stumps90:
            print eachStump.getStumpProperties()
        print "180: "
        for eachStump in self.stumps180:
            print eachStump.getStumpProperties()
        print "270: "
        for eachStump in self.stumps270:
            print eachStump.getStumpProperties()

    def classify(self):
        #self.displayStumps()
        with open(self.testFile) as document:
            for image in document.read().split(Constants.NEW_LINE):
                if image == Constants.EMPTY_STRING:
                    break 
                imageList = image.split()
                testImg = [int(pixelValue) for pixelValue in imageList[Constants.TWO:]]
                predictedOrientation = self.getOrientation(testImg)
                ResultsHelper.updateConfidenceMatrix(int(imageList[Constants.ONE]), predictedOrientation, self.confusionMatrix)
                #ResultsHelper.displayAccuracy(self.confusionMatrix)
                self.outputFile.write(imageList[0] + Constants.SPACE + str(predictedOrientation) + Constants.NEW_LINE)
                #if predictedOrientation != int(imageList[Constants.ONE]):
                print "Found orientation for: ", imageList[0], ': ', str(predictedOrientation), "Original orientation (given in Train): ", imageList[Constants.ONE]

    def displayResult(self):
        ResultsHelper.displayAccuracy(self.confusionMatrix)

