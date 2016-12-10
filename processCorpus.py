from collections import defaultdict
from constants import Constants

class ProcessCorpus:
    def __init__(self, trainFile):
        self.trainFile = trainFile
        self.vector = [] # stores lists containing pixel values
        self.imageIds = defaultdict(tuple) # stores key = vector index, value = (image id, orientation)
        self.totImages = 0 # total images in the training set

    def creatingVector(self):
        index = 0
        with open(self.trainFile) as document:
            for image in document.read().split(Constants.NEW_LINE):
                if image == Constants.EMPTY_STRING:
                    break
                self.totImages += 1
                imageList = image.split()
                self.imageIds[str(index)] = (imageList[0], int(imageList[Constants.ONE]))
                self.vector.append([int(pixelValue) for pixelValue in imageList[Constants.TWO:]])
                index += Constants.ONE
        #print self.vector
        #print self.imageIds

    def getVector(self):
        return self.vector

    def getImageIds(self):
        return self.imageIds