from __future__ import division
#from heapq import heappush
from collections import defaultdict
from operator import itemgetter
from random import randint
import math, copy
from pprint import pprint

from constants import Constants
from stump import Stump

class AdaBoostTrain:
    def __init__(self, stumpCount, processCorpusObj):
        self.stumpCount = stumpCount
        self.processCorpusObj = processCorpusObj
        self.imageIds = None
        self.vector = None
        self.vectorLength = 0
        
        # set stores the random pixel indexes as tuples (pixInd1, pixInd2) that are considered for stump construction.
        self.randomCombinations = set([])

        # Lists for saving all self.stumpCount stumps generated. Stores objects of Stump class
        self.allStumps0 = []
        self.allStumps90 = []
        self.allStumps180 = []
        self.allStumps270 = []

        # List which stores tuples: (score for classifier, indexes of vectors for correctly classified images, pix1Index, pix2Index)
        # pixel indexes for which greatest score was found. 
        # For every iteration
        self.maxDetails0 = None
        self.maxDetails90 = None
        self.maxDetails180 = None
        self.maxDetails270 = None
        
        # Dicts for storing the weights for each image (changes with each iteration of adaboost)
        # key = index of vector, value = weight
        self.weights0 = defaultdict(list) 
        self.weights90 = defaultdict(list) 
        self.weights180 = defaultdict(list) 
        self.weights270 = defaultdict(list) 


        # For optimization - populating the values of the following just once:
        self.rangeVectorSize = None

    # Find the details for the classifiers with max score on the fly
    def updateMax(self, currScore, currIndexes, currPix1, currPix2, classType):
        if classType == 0:
            if currScore > self.maxDetails0[0]:
                self.maxDetails0 = (currScore, currIndexes, currPix1, currPix2)
        elif classType == Constants.NINETY:
            if currScore > self.maxDetails90[0]:
                self.maxDetails90 = (currScore, currIndexes, currPix1, currPix2)
        elif classType == Constants.ONE_EIGHTY:
            if currScore > self.maxDetails180[0]:
                self.maxDetails180 = (currScore, currIndexes, currPix1, currPix2)
        else:
            if currScore > self.maxDetails270[0]:
                self.maxDetails270 = (currScore, currIndexes, currPix1, currPix2)

    # pos1, pos2 = two column indexes in self.vector whose pixel values need to be compared
    def findDetailsForMaxScorePositions(self, pos1, pos2): 
        # Store the indexes of the images which were correctly classified
        correctClassifiedIndexes0 = []
        correctClassifiedIndexes90 = []
        correctClassifiedIndexes180 = []
        correctClassifiedIndexes270 = []
        
        # stores the count of train mages where the > condition holds
        score0, score90, score180, score270 = 0, 0, 0, 0
        
        for i in self.rangeVectorSize:
            stri = str(i)
            if self.vector[i][pos1] > self.vector[i][pos2]:
                if self.imageIds[stri][Constants.ONE] == 0:
                    score0 += self.weights0[stri]
                    correctClassifiedIndexes0.append(stri)
                elif self.imageIds[stri][Constants.ONE] == Constants.NINETY:
                    score90 += self.weights90[stri]
                    correctClassifiedIndexes90.append(stri)
                elif self.imageIds[stri][Constants.ONE] == Constants.ONE_EIGHTY:
                    score180 += self.weights180[stri]
                    correctClassifiedIndexes180.append(stri)
                else: # self.imageIds[stri][Constants.ONE] == Constants.TWO_SEVENTY:
                    score270 += self.weights270[stri]
                    correctClassifiedIndexes270.append(stri)
            else:
                if self.imageIds[str(i)][Constants.ONE] == 0:
                    score90 += self.weights90[stri]
                    score180 += self.weights180[stri]
                    score270 += self.weights270[stri]
                    correctClassifiedIndexes90.append(stri)
                    correctClassifiedIndexes180.append(stri)
                    correctClassifiedIndexes270.append(stri)
                elif self.imageIds[stri][Constants.ONE] == Constants.NINETY:
                    score0 += self.weights0[stri]
                    score180 += self.weights180[stri]
                    score270 += self.weights270[stri]
                    correctClassifiedIndexes0.append(stri)
                    correctClassifiedIndexes180.append(stri)
                    correctClassifiedIndexes270.append(stri)
                elif self.imageIds[str(i)][Constants.ONE] == Constants.ONE_EIGHTY:
                    score0 += self.weights0[stri]
                    score90 += self.weights90[stri]
                    score270 += self.weights270[stri]
                    correctClassifiedIndexes0.append(stri)
                    correctClassifiedIndexes90.append(stri)
                    correctClassifiedIndexes270.append(stri)
                else: # self.imageIds[str(i)][Constants.ONE] == Constants.TWO_SEVENTY:
                    score0 += self.weights0[stri]
                    score90 += self.weights90[stri]
                    score180 += self.weights180[stri]
                    correctClassifiedIndexes0.append(stri)
                    correctClassifiedIndexes90.append(stri)
                    correctClassifiedIndexes180.append(stri)
        #print score0, score90, score180, score270
        self.updateMax(score0, correctClassifiedIndexes0, pos1, pos2, 0)
        self.updateMax(score90, correctClassifiedIndexes90, pos1, pos2, Constants.NINETY)
        self.updateMax(score180, correctClassifiedIndexes180, pos1, pos2, Constants.ONE_EIGHTY)
        self.updateMax(score270, correctClassifiedIndexes270, pos1, pos2, Constants.TWO_SEVENTY)

    def compareAllPixels(self):
        self.maxDetails0 = (-1, 0, 0, 0)
        self.maxDetails90 = (-1, 0, 0, 0)
        self.maxDetails180 = (-1, 0, 0, 0)
        self.maxDetails270 = (-1, 0, 0, 0)
        for indexes in self.randomCombinations:
            self.findDetailsForMaxScorePositions(*indexes)
        #print self.maxDetails0
        #print self.maxDetails90
        #print self.maxDetails180
        #print self.maxDetails270
        #sys.exit(0)
            
    def getStump(self, indexes, weight):
        return Stump(indexes[0], indexes[Constants.ONE], weight)
 
    def saveStump(self, best0, best90, best180, best270, alpha0, alpha90, alpha180, alpha270):
        self.allStumps0.append(self.getStump(best0, alpha0))
        self.allStumps90.append(self.getStump(best90, alpha90))
        self.allStumps180.append(self.getStump(best180, alpha180))
        self.allStumps270.append(self.getStump(best270, alpha270))

    def calculateBeta(self, score):
        return (Constants.ONE - score)/ score
    
    def normalizeWts(self):
        sumTotal0 = sum(self.weights0.values())
        sumTotal90 = sum(self.weights90.values())
        sumTotal180 = sum(self.weights180.values())
        sumTotal270 = sum(self.weights270.values())
        for i in self.rangeVectorSize:
            self.weights0[str(i)] /= sumTotal0
            self.weights90[str(i)] /= sumTotal90
            self.weights180[str(i)] /= sumTotal180
            self.weights270[str(i)] /= sumTotal270
        '''
        self.weights0 = {key: value/ sumTotal0 for key, value in self.weights0.items()}
        self.weights90 = {key: value/ sumTotal90 for key, value in self.weights90.items()}        
        self.weights180 = {key: value/ sumTotal180 for key, value in self.weights180.items()}        
        self.weights270 = {key: value/ sumTotal270 for key, value in self.weights270.items()}
        '''

    '''def updateWt(self, wts, correctlyClassified, beta):
        correctSet = set(correctlyClassified)
        print "In update wts: ", wts, correctSet, beta
        return {key: value * beta if int(key) not in correctSet else value for key, value in wts.items()}
    '''
    
    def updateWtsNextIteration(self, correctlyClassified0, correctlyClassified90, correctlyClassified180, correctlyClassified270, beta0, beta90, beta180, beta270):
        correctlyClassified0 = set(correctlyClassified0)
        correctlyClassified90 = set(correctlyClassified90)
        correctlyClassified180 = set(correctlyClassified180)
        correctlyClassified270 = set(correctlyClassified270)

        for i in self.rangeVectorSize:
            stri = str(i)
            
            currWeight0 = self.weights0[stri]
            self.weights0[stri] = currWeight0 * beta0 if stri in correctlyClassified0 else currWeight0
            
            currWeight90 = self.weights90[stri]
            self.weights90[stri] = currWeight90 * beta90 if stri in correctlyClassified90 else currWeight90
            
            currWeight180 = self.weights180[stri]
            self.weights180[stri] = currWeight180 * beta180 if stri in correctlyClassified180 else currWeight180
            
            currWeight270 = self.weights270[stri]
            self.weights270[stri] = currWeight270 * beta270 if stri in correctlyClassified270 else currWeight270
            
        self.normalizeWts()
        '''
        print "Updates270: ", 
        pprint(self.weights270)
        
        self.weights0 = self.updateWt(self.weights0, correctlyClassified0, beta0)
        #print "Updated0: ", 
        #pprint(self.weights0)
        self.weights90 = self.updateWt(self.weights90, correctlyClassified90, beta90)
        #print "Updated90: ", 
        #pprint(self.weights90)
        self.weights180 = self.updateWt(self.weights180, correctlyClassified180, beta180)
        #print "Updated180: ", 
        #pprint(self.weights180)
        self.weights270 = self.updateWt(self.weights270, correctlyClassified270, beta270)
        #print "Updated270: ", 
        #pprint(self.weights270)
        '''
    
    def calculateAlpha(self, beta):
        return math.log(Constants.ONE / beta)

    def train(self):
        self.getDataFromProcessCorpus()
        self.rangeVectorSize = [i for i in xrange(self.vectorLength)]
        self.setInitialWeights()
        #print self.weights0
        #print self.weights90
        #print self.weights180
        #print self.weights270
        self.getRandomIndexes()
        for i in xrange(self.stumpCount):
            self.compareAllPixels()
            #print "Found Best pixel indexes."
            beta0, beta90, beta180, beta270 = map(self.calculateBeta, [self.maxDetails0[0], self.maxDetails90[0], self.maxDetails180[0], self.maxDetails270[0]])
            #print "Found beta Values.", beta0, beta90, beta180, beta270
            alpha0, alpha90, alpha180, alpha270 = map(self.calculateAlpha, [beta0, beta90, beta180, beta270])
            #print "Calculated Alpha Values."
            self.saveStump(self.maxDetails0[Constants.TWO:Constants.FOUR], self.maxDetails90[Constants.TWO:Constants.FOUR], \
                      self.maxDetails180[Constants.TWO:Constants.FOUR], self.maxDetails270[Constants.TWO:Constants.FOUR], alpha0, alpha90, alpha180, alpha270)
            #print "Saved Stump."
            
            self.updateWtsNextIteration(self.maxDetails0[Constants.ONE], self.maxDetails90[Constants.ONE], \
                                       self.maxDetails180[Constants.ONE], self.maxDetails270[Constants.ONE], beta0, beta90, beta180, beta270)
            #print "Updated weights for next iteration."
            #print self.weights0, "\n"
            #print self.weights90, "\n"
            #print self.weights180, "\n"
            #print self.weights270, "\n"
            
            #print "Alpha: ", alpha0, alpha90, alpha180, alpha270
            #print "Beta: ", beta0, beta90, beta180, beta270 
        #sys.exit(0)

    # Generates a list of 1000 random tuples: (pixelIndex1, pixelIndex2). Indexes in the range [0, 191]
    def getRandomIndexes(self):
        for i in xrange(Constants.THOUSAND):
            ind1 = randint(0, Constants.ONE_NINETYONE)
            ind2 = randint(0, Constants.ONE_NINETYONE)
            # check if indexes are not the same, or if an index pair has already been generated
            while(ind1 == ind2 or (ind1, ind2) in self.randomCombinations):
                ind1 = randint(0, Constants.ONE_NINETYONE)
                ind2 = randint(0, Constants.ONE_NINETYONE)
            self.randomCombinations.add((ind1, ind2))
        #print "Generated 1000 random index pairs."

    def setInitialWeights(self):
        self.weights0 = {str(index): Constants.ONE/ self.vectorLength for index in self.rangeVectorSize}
        self.weights90 = copy.deepcopy(self.weights0)
        self.weights180 = copy.deepcopy(self.weights0)
        self.weights270 = copy.deepcopy(self.weights0)

    def getDataFromProcessCorpus(self):
        self.imageIds = self.processCorpusObj.getImageIds()
        self.vector = self.processCorpusObj.getVector()
        self.vectorLength = self.processCorpusObj.totImages
    
