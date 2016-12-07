from resultsHelper import ResultsHelper
from constants import Constants
import math

class NeuralNet:
    def __init__(self, nearestOutputFile, testFile, stumpCount, processCorpusObj):
        self.confusionMatrix = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
        self.outputFile = nearestOutputFile
        self.testFile = testFile
        self.stumpCount = stumpCount
        self.processCorpusObj = processCorpusObj
        self.imageIds = None
        self.vector = None
        self.weights = defaultdict(list) # key = weight, value = [indexes of vector] # that is stores the indexes of images

    def classify(self):
        self.trainedModel = self.processCorpusObj.getTrainedModel()
        self.imageIds = self.processCorpusObj.getImageIds()
        self.vector = self.processCorpusObj.getVector()
        self.setInitialWeights()
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

