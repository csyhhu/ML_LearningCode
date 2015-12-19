'''
Codes come from Machine Leaning in Action
Modified by Shangyu Chen in Dec.17th, 2015
'''
from numpy import *
import matplotlib.pyplot as pyplot
from logRegression import *

def loadData():
	train_x = []
	train_y = []
	fileIn = open('./testSet.txt')
	for line in fileIn.readlines():
		lineArr = line.strip().split()
		train_x.append([1.0, float(lineArr[0]), float(lineArr[1])]) #add a 1.0 in order to make linear model compact
		train_y.append(float(lineArr[2]))
	return mat(train_x), mat(train_y).transpose()

print 'Step 1: Load Data'
train_x, train_y = loadData()
#print train_x
#print shape(train_x)

print 'Step 2: Training'
opts = {'alpha':0.01, 'maxIter':200, 'optimizeType':'smoothStocGradDescent'}
optimalWeights = trainLogRegress(train_x, train_y, opts)
#print optimalWeights

print 'Step 3: Testing'
accuracy = testLogRegression(optimalWeights, train_x, train_y)
print accuracy

print 'Step 4: Show the result'
showLogRegression(optimalWeights, train_x, train_y)