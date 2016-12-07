'''
#!/usr/bin/python -tt
# -*- coding: utf-8 -*-
import codecs, sys
'''
from constants import Constants

class ResultsHelper:
    @staticmethod
    def displayAccuracy(matrix):
        #UTF8Writer = codecs.getwriter(Constants.UTF8)
        #sys.stdout = UTF8Writer(sys.stdout)
        print "\nAccuracy Percentage: ", round(sum([matrix[i][i] for i in range(Constants.CONFIDENCE_MATRIX_SIZE)]) * Constants.HUNDRED_FLOAT / \
                              sum([matrix[i][j] for i in range(Constants.CONFIDENCE_MATRIX_SIZE) for j in range(Constants.CONFIDENCE_MATRIX_SIZE)]), Constants.TWO)
        print "Confidence Matrix: "
	print Constants.CONFIDENCE_HEADER.encode(Constants.UTF8, Constants.REPLACE)
        for i in range(Constants.CONFIDENCE_MATRIX_SIZE):
            print str(i * Constants.NINETY) + Constants.DEGREE_TAB.encode(Constants.UTF8, Constants.REPLACE), 
            for j in range(Constants.CONFIDENCE_MATRIX_SIZE):
                print str(matrix[i][j]) + Constants.TAB, 
            print Constants.NEW_LINE

    @staticmethod
    def updateConfidenceMatrix(actualOrientation, predictedOrientation, matrix):
        #print actualOrientation, predictedOrientation
        matrix[actualOrientation / Constants.NINETY][predictedOrientation / Constants.NINETY] += Constants.ONE
