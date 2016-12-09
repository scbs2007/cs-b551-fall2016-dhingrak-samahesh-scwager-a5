from processCorpus import ProcessCorpus
from constants import Constants
from nearest import Nearest
from adaBoostTrain import AdaBoostTrain
from adaBoostTest import AdaBoostTest
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
        stumpCount = int(sys.argv[Constants.FOUR])
        trainObj = AdaBoostTrain(stumpCount, processCorpus)
        trainObj.train()
        print "Trained."
        print trainObj.allStumps0[0].pixelIndex1, trainObj.allStumps0[0].pixelIndex2, trainObj.allStumps0[0].alpha
        print "Started classifying..."
        adaBoostObj = AdaBoostTest(outputFile, testFile, trainObj)
        adaBoostObj.classify()
        adaBoostObj.displayResult()

    elif classifierType == Constants.NNET:
        processCorpus.creatingVector() 
        hiddenCount = sys.argv[Constants.FOUR]
        neuralNetObj = NeuralNet(outputFile, testFile, hiddenCount, processCorpus)
        neuralNetObj.classify()
        neuralNetObj.displayResult()

    elif classifierType == Constants.BEST:
        pass
    outputFile.close()
    
