#from heapq import nsmallest
from resultsHelper import ResultsHelper
from constants import Constants
import math, sys
#import itertools
#from multiprocessing.dummy import Pool as ThreadPool

class Nearest:
    def __init__(self, nearestOutputFile, testFile, processCorpusObj):
        self.confusionMatrix = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
        self.processCorpusObj = processCorpusObj
        self.outputFile = nearestOutputFile
        self.testFile = testFile
        self.vector = None
        self.imageIds = None
        #self.pool = ThreadPool(5)

        # Optimization - Generating this just once:
        self.rangeImageLength = []

    def findEuclideanDist(self, image1, image2): #images):
        # Optimization - Removed math.sqrt for speed up

        #image1, image2 = images
        #return sum([(image1[index] - image2[index]) * (image1[index] - image2[index]) for index in range(Constants.IMAGE_LENGTH)])
        dist = 0
        for index in self.rangeImageLength:#(Constants.IMAGE_LENGTH):
            dist += (image1[index] - image2[index]) ** 2 
        return dist

    def findNearestImage(self, testImg):
        #euclideanDistances = self.pool.map(self.findEuclideanDist, zip(itertools.repeat(testImg, len(self.vector)), self.vector))
        #euclideanDistances = map(self.findEuclideanDist, zip(itertools.repeat(testImg, len(self.vector)), self.vector)) 
        #euclideanDistances = [self.findEuclideanDist(trainImg, testImg) for trainImg in self.vector]
        #return self.imageIds[str(euclideanDistances.index(min(euclideanDistances)))]

        imageIndex = 0
        minDist = sys.maxint
        for trainImg in self.vector:
            dist = self.findEuclideanDist(trainImg, testImg)
            if dist < minDist:
                minDist = dist
                minIndex = imageIndex
            imageIndex += 1
        return self.imageIds[str(minIndex)]

    def classify(self):
        self.rangeImageLength = [i for i in range(Constants.IMAGE_LENGTH)]
        self.vector = self.processCorpusObj.getVector()
        self.imageIds = self.processCorpusObj.getImageIds()
        with open(self.testFile) as document:
            for image in document.read().split(Constants.NEW_LINE):
                if image == Constants.EMPTY_STRING:
                    break 
                imageList = image.split()
                testImg = [int(pixelValue) for pixelValue in imageList[Constants.TWO:]]
                trainFileId, predictedOrientation = self.findNearestImage(testImg)
                ResultsHelper.updateConfidenceMatrix(int(imageList[Constants.ONE]), predictedOrientation, self.confusionMatrix)
                #ResultsHelper.displayAccuracy(self.confusionMatrix)
                self.outputFile.write(imageList[0] + Constants.SPACE + str(predictedOrientation) + Constants.NEW_LINE)
                print "Found orientation for: ", imageList[0], ': ', str(predictedOrientation), "Original orientation (given in Train): ", imageList[Constants.ONE]
        #self.pool.close()
        #self.pool.join()

    def displayResult(self):
        ResultsHelper.displayAccuracy(self.confusionMatrix)
