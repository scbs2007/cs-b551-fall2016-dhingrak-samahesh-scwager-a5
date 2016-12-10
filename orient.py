from processCorpus import ProcessCorpus
from constants import Constants
from nearest import Nearest
from adaBoostTrain import AdaBoostTrain
from adaBoostTest import AdaBoostTest
from neuralNet import NeuralNet
from bestClassifier import BestClassifier
import sys, pickle

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
        stumpCount = int(sys.argv[Constants.FOUR])
        processCorpus.creatingVector() 
        trainingObj = AdaBoostTrain(stumpCount, processCorpus)
        trainingObj.train()
        print "Trained."
        pickle.dump(trainingObj, open('model' + str(stumpCount), "wb"))
        '''
        trainingObj = pickle.load(open('model' + str(stumpCount), "rb"))
        print "Started classifying..."
        testObj = AdaBoostTest(outputFile, testFile, trainingObj)
        testObj.classify()
        testObj.displayResult()
        '''
    elif classifierType == Constants.NNET:
        processCorpus.creatingVector() 
        hiddenCount = sys.argv[Constants.FOUR]
        neuralNetObj = NeuralNet(outputFile, testFile, hiddenCount, processCorpus)
        neuralNetObj.classify()
        neuralNetObj.displayResult()

    elif classifierType == Constants.BEST:
        pass
    outputFile.close()
    
