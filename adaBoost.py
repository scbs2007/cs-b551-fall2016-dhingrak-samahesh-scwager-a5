from __future__ import division
from resultsHelper import ResultsHelper
from constants import Constants
from heapq import heappush
from collections import defaultdict
from operator import itemgetter
import math

class AdaBoost:
    def __init__(self, nearestOutputFile, testFile, stumpCount, processCorpusObj):
        self.confusionMatrix = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
        self.outputFile = nearestOutputFile
        self.testFile = testFile
        self.stumpCount = stumpCount
        self.processCorpusObj = processCorpusObj
        self.imageIds = None
        self.vector = None
        self.weights = defaultdict(list) # key = index of vector, value = weight # that is stores the indexes of images

        # For optimization - populating the values of the following just once:
        self.rangeListImage = []
        self.vectorLength = 0

    # pos1, pos2 = two column indexes in self.vector whose pixel values need to be compared. classType = one of 0, 90, 180, 270
    def findClassCountsForPositions(self, pos1, pos2, classType, fractionForClassType0, fractionForClassType90, fractionForClassType180, fractionForClassType270): 
        positiveCount = [0, 0, 0, 0] # stores the count of train mages where the orientation is classType and the > condition holds
        negativeCount = [0, 0, 0, 0] # stores the count of train mages where the orientation is not classType and the > condition holds
        for i in self.rangeListImage: #range(self.vectorLength):
            if self.vector[i][pos1] > self.vector[i][pos2]:
                if self.imageIds[str(i)][Constants.ONE] == 0:
                    positiveCount[0] += 1
                elif self.imageIds[str(i)][Constants.ONE] == 90:
                    positiveCount[1] += 1
                elif self.imageIds[str(i)][Constants.ONE] == 180:
                    positiveCount[2] += 1
                else: # self.imageIds[str(i)][Constants.ONE] == 270:
                    positiveCount[3] += 1
                
        fractionForClassType0.append((positiveCount[0] / self.vectorLength, str(pos1) + Constants.DELIMITER + str(pos2)))
        fractionForClassType90.append((positiveCount[1] / self.vectorLength, str(pos1) + Constants.DELIMITER + str(pos2)))
        fractionForClassType180.append((positiveCount[2] / self.vectorLength, str(pos1) + Constants.DELIMITER + str(pos2)))
        fractionForClassType270.append((positiveCount[3] / self.vectorLength, str(pos1) + Constants.DELIMITER + str(pos2)))
        #heappush(fractionForClassType0, (-positiveCount[0] / self.vectorLength, str(pos1) + Constants.DELIMITER + str(pos2)))
        #heappush(fractionForClassType90, (-positiveCount[1] / self.vectorLength, str(pos1) + Constants.DELIMITER + str(pos2)))
        #heappush(fractionForClassType180, (-positiveCount[2] / self.vectorLength, str(pos1) + Constants.DELIMITER + str(pos2)))
        #heappush(fractionForClassType270, (-positiveCount[3] / self.vectorLength, str(pos1) + Constants.DELIMITER + str(pos2)))
        #inserted negative value to get max value instead of min value from heap

    def compareAllPixels(self, classType):
        fractionsForClassType0 = [] # fractionForClassType = heap which stores (count of images where condition satisfies/ total images, pos1'|'pos2)
        fractionsForClassType90 = [] # fractionForClassType = heap which stores (count of images where condition satisfies/ total images, pos1'|'pos2)
        fractionsForClassType180 = [] # fractionForClassType = heap which stores (count of images where condition satisfies/ total images, pos1'|'pos2)
        fractionsForClassType270 = [] # fractionForClassType = heap which stores (count of images where condition satisfies/ total images, pos1'|'pos2)
        for i in range(Constants.IMAGE_LENGTH - 1):
            for j in range(i + 1, Constants.IMAGE_LENGTH):
                self.findClassCountsForPositions(i, j, 0, fractionsForClassType0, fractionsForClassType90, fractionsForClassType180, fractionsForClassType270)
        print "For 0 degree: ", max(fractionsForClassType0, key=itemgetter(0))#fractionsForClassType0[0]
        print "For 90 degree: ", max(fractionsForClassType90, key=itemgetter(0))#fractionsForClassType90[0]
        print "For 180 degree: ", max(fractionsForClassType180, key=itemgetter(0))#fractionsForClassType180[0]
        print "For 270 degree: ", max(fractionsForClassType270, key=itemgetter(0))#fractionsForClassType270[0]
        sys.exit(0)
            
    def setInitialWeights(self):
        self.weights = {str(index): 1/ self.vectorLength for index in self.rangeListImage}

    def classify(self):
        self.imageIds = self.processCorpusObj.getImageIds()
        self.vector = self.processCorpusObj.getVector()
        self.vectorLength = len(self.vector)
        self.rangeListImage = [i for i in range(self.vectorLength)]
        self.setInitialWeights()
        for i in range(len(self.stumpCount)):
            self.compareAllPixels(0)
            self.CalculateError()
            self.saveStump()
            self.updateWeights()

            
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

