'''
We have stored all images' pixels as a list and have stored each of those lists in another list which represents our vector. Also have kept the information of each 
image id and given orientation in a dictionary - imageIds which has key as the earlier said vector's index and value as a tuple representing a tuple 
(image id, given orientation)

Our program has been divided into many files:
1. orient.py is where the main execution starts.
2. processCorpus.py deals with the reading of all the records given for the training of the models. This is where the aforesaid vector and imageIds dictionary 
gets created.
3. nearest.py is where the code for 1 nearest neighbor resides. 
4. adaBoostTrain.py is the file which deals with the training of the decision stumps. This is where the adaboost algorithmm has been implemented.
5. adaBoostTest.py is where we test the given test images on the trained adaboost model.
6. neuralNet.py is the file where the Neural Net has been implemented
7. neuralNetTest.py is where the testing of the nnet on the given test images takes place.

1-Nearest: This implementation is straight forward. For each given test image we calculate its euclidean distance from each training image, and find the smallest
distance among all. That image's orientation is the one we assign to the test image.

Result:

Accuracy Percentage:  67.48
Confidence Matrix: 
	0	90	180	270
0	153	21	36	24	

90	18	146	18	30	

180	37	26	145	13	

270	21	31	19	166

AdaBoost: We have formalized usage of ada boost as:
For each iteration of Adaboost we build 4 stumps - one each for the 4 degrees: 0, 90, 180, 270. 
- As the Adaboost algorithm says: We first give 1/ (total number of images) in the training data as the weight to each of the train images.
  [For simplicity the below has been written just for 0 degree. The same applies for all the other 3 degrees.]
- Now, For 0 degree classifier we build '0 vs all' decision stumps in the following way:
- We pick up 1000 random pairs of pixels from the given 192 image vector.
- Now, for each of those 1000 random pairs we:
- - - - Iterate over all the given training images and find the score which is the sum of 0Yes + ~0Yes where
        0Yes = sum of weight of images having 0 degree orientation and having pixel1 value > pixel2 value
        ~0Yes = sum of weights of images having non 0 degree orientations and having pixel1 value < pixel2 value
- From among all the 1000 scores we then find the max score S and those pixels which gave the max score; and those are the pixels we use in the first stump for the 0 
degree classifier.
- We now find the beta value, which is = (1 - S)/ (S)
- Then we update the weights of the vectors which were correctl classified by multiplying them with beta.
- Next we find the alpha value which represents the confidence vote of the found stump in the overall decision.
- This completes the first iteration of ada boost.
And the above process is repeated for the stump count number entered by the user.

We experimented with parameters for Adaboost as explained in Report.docx and found the best accuracy for the model with 75 stumps where each stump was found
after considering 1000 random pixels pairs.

Result:

Accuracy Percentage:  70.2
Confidence Matrix: 
	0	90	180	270
0	169	34	24	12	

90	21	172	9	22	

180	40	31	150	15	

270	29	38	6	171

Neural network:
Input is the raw data: each x-values is a 192D vector of rgb values. They are not processed.
The output is a oneHotIt matrix of the four possible rotations.
Weights are set to small values uniformly distributed around 0.
The output activation function is softmax, while the hidden layer can be sigmoid, tanh, relu, or softmax.

The first two parameters tested for were the dimension of the hidden layer and the hidden layer activation function. 
Having fixed the following parameters:
--> Number of iterations = 7*1e6
--> Step size for iteration i = 0.01 / (1 + 100*i/iterations) * (1/m), m is number of examples
Testing values ranging from 50 to 300 with increments of 50, for both reLU
--> The results were best between 150 and 200, near the dimensionality of the data, with testing accuracy of 71.79% (best: ReLU nn_hdim = 200)
Using tanh or or reLU as the hidden layer activation function gave test accuracy with a difference less than .5%.

ReLU was chosen for two further tests:
(1) check the effect of decreasing the step size 100 times less. The new function was:
    Step size for iteration i = 0.01 / (1 + i/iterations) * (1/m), m is number of examples
    Number of iterations = 7*1e6
--> Training accuracy increased from 70-72% to 76-77%.
--> Testing accuracy increased from 70-72% to 74-76%.
--> We also tested for smaller hidden layer dimensions: 10 and 25, noticing that the results were no worse than those of higher dimensions.
This result is in line with the fact that adaBoost reached its best performance using only 20 features. We assume there is much redundancy in 
the image data and all learning algorithms work efficiently with the key features.

(2) check the effect of adding many more iterations
    Number of iterations = 1e7:
    Step size for iteration i = 0.01 / (1 + i/iterations) * (1/m), m is number of examples
--> Training accucary increased from 76-77% to 77-78%.
--> Testing accuracy increased from 74-76% to 75-77%

The overall best result of 76.44% testing accuracy was achieved with 1e7 iterations, 250 hidden nodes, and step size function i = 0.01 / (1 + i/iterations) * (1/m)

Result:

Accuracy Percentage:  76.44
Confidence Matrix: 
	0	90	180	270
0	175	12	33	14

90 	12 	167 	13 	20

180 	33 	16 	162 	10

270 	13 	25 	12 	187 


Best:
After having experimented with a lot of features for both adaboost and neural net we found the best accuracy using the features stated above for nnet.
when the program is run for "best" option the model is loaded from the model file. This is the file where after you run the nnet option the model for nnet is stored.

'''

from processCorpus import ProcessCorpus
from constants import Constants
from nearest import Nearest
from adaBoostTrain import AdaBoostTrain
from adaBoostTest import AdaBoostTest
from neuralNet import NeuralNet
from neuralNetTest import NeuralNetTest
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
        #pickle.dump(trainingObj, open('../cv_set4_750/model' + str(stumpCount), "wb"))        
        #trainingObj = pickle.load(open('model' + str(stumpCount), "rb"))
        print "Started classifying..."
        testObj = AdaBoostTest(outputFile, testFile, trainingObj)
        testObj.classify()
        testObj.displayResult()
        
    elif classifierType == Constants.NNET:
        processCorpus.creatingVector() 
        hiddenCount = sys.argv[Constants.FOUR]
        neuralNetObj = NeuralNet(outputFile, testFile, hiddenCount, processCorpus)
        neuralNetObj.train()
#         neuralNetObj.displayResult()
        testObj = NeuralNetTest(outputFile, testFile)

    elif classifierType == Constants.BEST:
        pass
    outputFile.close()
    
