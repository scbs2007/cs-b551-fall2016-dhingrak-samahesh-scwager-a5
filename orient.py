from processCorpus import ProcessCorpus
from constants import Constants
from nearest import Nearest
from adaBoost import AdaBoost
from neuralNet import NeuralNet
from bestClassifier import BestClassifier
import sys

trainFile, testFile, classifierType = sys.argv[Constants.ONE:Constants.FOUR]
classifiers = set([Constants.NEAREST, Constants.ADABOOST, Constants.NNET, Constants.BEST])

if classifierType not in classifiers:
    print "Incorrect classifier type specified: ", classifierType
else:
    outputFile = open(classifierType + Constants.OUTPUT_FILE, Constants.WRITE)
    print "Reading File..."
    processCorpus = ProcessCorpus(trainFile)
    print "Training model..."

    if classifierType == Constants.NEAREST:
        processCorpus.creatingVector() 
        print "Trained."
        print "Started classifying..."
        nearestObj = Nearest(outputFile, testFile, processCorpus)
        nearestObj.classify()
        nearestObj.displayResult()
    
    elif classifierType == Constants.ADABOOST:
        processCorpus.creatingVector() 
        stumpCount = sys.argv[Constants.FOUR]
        adaBoostObj = AdaBoost(outputFile, testFile, stumpCount, processCorpus)
        adaBoostObj.classify()
        adaBoostObj.displayResult()

    elif classifierType == Constants.NNET:
        hiddenCount = sys.argv[Constants.FOUR]
        neuralNetObj = NeuralNet(outputFile, testFile, hiddenCount, processCorpus)
        neuralNetObj.classify()
        neuralNetObj.displayResult()

    elif classifierType == Constants.BEST:
        pass
    outputFile.close()
    
