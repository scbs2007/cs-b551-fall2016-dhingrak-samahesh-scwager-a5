# Splits file into 90% training and 10% test
# For cross validation

import sys, os
from random import randint

trainFile = "train-data.txt"
# 5 fold cross validation

for i in range(5):
    stri = str(i)
    if not os.path.exists("../validationSet" + stri):
        os.makedirs("../validationSet" + stri)
    train90 = open("../validationSet" + stri + "/trainFile" + stri, 'w')
    test10 = open("../validationSet" + stri + "/testFile" + stri, 'w')

    with open(trainFile) as document:
        for image in document.read().split('\n'):
            if image == '':
                break
            if 0 <= randint(0,10) <= 9:
                train90.write(image + '\n')
            else:
                test10.write(image + '\n')
    
    train90.close()
    test10.close()
