from __future__ import division
#from heapq import heappush
from collections import defaultdict
from operator import itemgetter
from random import randint
import math, copy

from constants import Constants
from stump import Stump

class AdaBoostTrain:
    def __init__(self, stumpCount, processCorpusObj):
        self.stumpCount = stumpCount
        self.processCorpusObj = processCorpusObj
        self.imageIds = None
        self.vector = None
        self.totTrainImages = 0
        
        # Stores the random pixel indexes as tuples (pixInd1, pixInd2) that are considered for stump construction.
        self.randomCombinations = []

        # Dicts for storing the weights for each image (changes with each iteration of adaboost)
        # key = index of vector, value = weight
        self.weights0 = defaultdict(list) 
        self.weights90 = defaultdict(list) 
        self.weights180 = defaultdict(list) 
        self.weights270 = defaultdict(list) 

        # List which stores tuples: (score for classifier, indexPos1+'|'+indexPos2 for which greatest score was found)
        self.detailsForClassType0 = None
        self.detailsForClassType90 = None
        self.detailsForClassType180 = None
        self.detailsForClassType270 = None
        
        # Lists for saving all self.stumpCount stumps generated. Stores objects of Stump class
        self.allStumps0 = []
        self.allStumps90 = []
        self.allStumps180 = []
        self.allStumps270 = []

        # For optimization - populating the values of the following just once:
        self.rangeListImage = []
        self.vectorLength = 0

    # pos1, pos2 = two column indexes in self.vector whose pixel values need to be compared
    def findClassCountsForPositions(self, pos1, pos2): #, detailsForClassType0, detailsForClassType90, detailsForClassType180, detailsForClassType270): 
        scores = [0, 0, 0, 0] # stores the count of train mages where the orientation is classType and the > condition holds
        #positiveCount = [0, 0, 0, 0] # stores the count of train mages where the orientation is not classType and the > condition holds

        # Store the indexes of the images which were correctly classified
        correctClassifiedIndexes0 = []
        correctClassifiedIndexes90 = []
        correctClassifiedIndexes180 = []
        correctClassifiedIndexes270 = []

        for i in self.rangeListImage:
            if self.vector[i][pos1] > self.vector[i][pos2]:
                if self.imageIds[str(i)][Constants.ONE] == 0:
                    scores[0] += 1 * self.weights0[str(i)]
                    correctClassifiedIndexes0.append(i)
                elif self.imageIds[str(i)][Constants.ONE] == Constants.NINETY:
                    scores[1] += 1 * self.weights90[str(i)]
                    correctClassifiedIndexes90.append(i)
                elif self.imageIds[str(i)][Constants.ONE] == Constants.ONE_EIGHTY:
                    scores[2] += 1 * self.weights180[str(i)]
                    correctClassifiedIndexes180.append(i)
                else: # self.imageIds[str(i)][Constants.ONE] == Constants.TWO_SEVENTY:
                    scores[3] += 1 * self.weights270[str(i)]
                    correctClassifiedIndexes270.append(i)
            else:
                if self.imageIds[str(i)][Constants.ONE] == 0:
                    scores[1] += 1 * self.weights90[str(i)]
                    scores[2] += 1 * self.weights180[str(i)]
                    scores[3] += 1 * self.weights270[str(i)]
                    correctClassifiedIndexes90.append(i)
                    correctClassifiedIndexes180.append(i)
                    correctClassifiedIndexes270.append(i)
                elif self.imageIds[str(i)][Constants.ONE] == Constants.NINETY:
                    scores[0] += 1 * self.weights0[str(i)]
                    scores[2] += 1 * self.weights180[str(i)]
                    scores[3] += 1 * self.weights270[str(i)]
                    correctClassifiedIndexes0.append(i)
                    correctClassifiedIndexes180.append(i)
                    correctClassifiedIndexes270.append(i)
                elif self.imageIds[str(i)][Constants.ONE] == Constants.ONE_EIGHTY:
                    scores[0] += 1 * self.weights0[str(i)]
                    scores[1] += 1 * self.weights90[str(i)]
                    scores[3] += 1 * self.weights270[str(i)]
                    correctClassifiedIndexes0.append(i)
                    correctClassifiedIndexes90.append(i)
                    correctClassifiedIndexes270.append(i)
                else: # self.imageIds[str(i)][Constants.ONE] == Constants.TWO_SEVENTY:
                    scores[0] += 1 * self.weights0[str(i)]
                    scores[1] += 1 * self.weights90[str(i)]
                    scores[2] += 1 * self.weights180[str(i)]
                    correctClassifiedIndexes0.append(i)
                    correctClassifiedIndexes90.append(i)
                    correctClassifiedIndexes180.append(i)
                
        self.detailsForClassType0.append((scores[0], correctClassifiedIndexes0, pos1, pos2))
        self.detailsForClassType90.append((scores[1], correctClassifiedIndexes90, pos1, pos2))
        self.detailsForClassType180.append((scores[2], correctClassifiedIndexes180, pos1, pos2))
        self.detailsForClassType270.append((scores[3], correctClassifiedIndexes270, pos1, pos2))

        #heappush(detailsForClassType0, (-scores[0], correctClassifiedIndexes0, str(pos1) + Constants.DELIMITER + str(pos2)))
        #heappush(detailsForClassType90, (-scores[1], correctClassifiedIndexes90, str(pos1) + Constants.DELIMITER + str(pos2)))
        #heappush(detailsForClassType180, (-scores[2], correctClassifiedIndexes180, str(pos1) + Constants.DELIMITER + str(pos2)))
        #heappush(detailsForClassType270, (-scores[3], correctClassifiedIndexes270, str(pos1) + Constants.DELIMITER + str(pos2)))
        #inserted negative value to get max value instead of min value from heap

    def compareAllPixels(self):
        self.detailsForClassType0 = [] 
        self.detailsForClassType90 = [] 
        self.detailsForClassType180 = [] 
        self.detailsForClassType270 = [] 
        #for i in range(Constants.IMAGE_LENGTH):
        #    for j in range(Constants.IMAGE_LENGTH):
        for ind1, ind2 in self.randomCombinations:
            self.findClassCountsForPositions(ind1, ind2)#, detailsForClassType0, detailsForClassType90, detailsForClassType180, detailsForClassType270)
        '''print "For 0 degree: ", max(self.detailsForClassType0, key=itemgetter(0))#fractionsForClassType0[0]
        print "For 90 degree: ", max(self.detailsForClassType90, key=itemgetter(0))#fractionsForClassType90[0]
        print "For 180 degree: ", max(self.detailsForClassType180, key=itemgetter(0))#fractionsForClassType180[0]
        print "For 270 degree: ", max(self.detailsForClassType270, key=itemgetter(0))#fractionsForClassType270[0]
        sys.exit(0)
        '''
        return max(self.detailsForClassType0, key=itemgetter(0)), max(self.detailsForClassType90, key=itemgetter(0)), \
               max(self.detailsForClassType180, key=itemgetter(0)), max(self.detailsForClassType270, key=itemgetter(0))
            
    def getStump(self, indexes, weight):
        return Stump(indexes[0], indexes[1], weight)
 
    def saveStump(self, best0, best90, best180, best270, alpha0, alpha90, alpha180, alpha270):
        self.allStumps0.append(self.getStump(best0, alpha0))
        self.allStumps90.append(self.getStump(best90, alpha90))
        self.allStumps180.append(self.getStump(best180, alpha180))
        self.allStumps270.append(self.getStump(best270, alpha270))

    def calculateBeta(self, epsilon):
        return epsilon/ (1 - epsilon)
    
    def normalizeWts(self):
        sumTotal = sum(self.weights0.values())
        self.weights0 = {key: value/ sumTotal for key, value in self.weights0.items()}
        
        sumTotal = sum(self.weights90.values())
        self.weights90 = {key: value/ sumTotal for key, value in self.weights90.items()}
        
        sumTotal = sum(self.weights180.values())
        self.weights180 = {key: value/ sumTotal for key, value in self.weights180.items()}
        
        sumTotal = sum(self.weights270.values())
        self.weights270 = {key: value/ sumTotal for key, value in self.weights270.items()}

    def updateWt(self, wts, prevDetails, beta):
        wts = {key: str(int(value) * beta) if int(key) not in set(prevDetails) else value for key, value in wts.items()}

    def updateWtsNextIteration(self, details0, details90, details180, details270, beta0, beta90, beta180, beta270):
        self.updateWt(self.weights0, details0, beta0)
        self.updateWt(self.weights90, details90, beta90)
        self.updateWt(self.weights180, details180, beta180)
        self.updateWt(self.weights270, details270, beta270)
        self.normalizeWts()
    
    def calculateAlpha(self, beta):
        return math.log(1 / beta)

    def train(self):
        self.getDataFromProcessCorpus()
        self.rangeListImage = [i for i in range(self.vectorLength)]
        self.setInitialWeights()
        self.getRandomIndexes()
        for i in range(self.stumpCount):
            best0, best90, best180, best270 = self.compareAllPixels()
            beta0, beta90, beta180, beta270 = map(self.calculateBeta, [best0[0], best90[0], best180[0], best270[0]])
            self.updateWtsNextIteration(best0[1], best90[1], best180[1], best270[1], beta0, beta90, beta180, beta270)
            alpha0, alpha90, alpha180, alpha270 = map(self.calculateAlpha, [beta0, beta90, beta180, beta270]) 
            self.saveStump(best0[2:4], best90[2:4], best180[2:4], best270[2:4], alpha0, alpha90, alpha180, alpha270)

            print "Best: ","\n", best0,"\n", best90, "\n", best180, "\n", best270
            print "Alpha: ", alpha0, alpha90, alpha180, alpha270
            print "Beta: ", beta0, beta90, beta180, beta270 
            sys.exit(0)

    # Generates a list of 1000 random tuples: (pixelIndex1, pixelIndex2). Indexes in the range [0, 191]
    def getRandomIndexes(self):
        for i in range(Constants.THOUSAND):
            ind1 = randint(0, 191)
            ind2 = randint(0, 191)
            # check if indexes are not the same
            while(ind1 == ind2):
                ind1 = randint(0, 191)
                ind2 = randint(0, 191)
            self.randomCombinations.append((ind1, ind2))

    def setInitialWeights(self):
        self.weights0 = {str(index): 1/ self.vectorLength for index in self.rangeListImage}
        self.weights90 = copy.deepcopy(self.weights0)
        self.weights180 = copy.deepcopy(self.weights0)
        self.weights270 = copy.deepcopy(self.weights0)

    def getDataFromProcessCorpus(self):
        self.imageIds = self.processCorpusObj.getImageIds()
        self.vector = self.processCorpusObj.getVector()
        self.vectorLength = len(self.vector)
        self.totTrainImages = self.processCorpusObj.totImages
    
