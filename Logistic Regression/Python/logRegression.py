from numpy import *
import matplotlib.pyplot as plt

def sigmoid(inx):
	return 1.0 / (1 + exp(-inx))

def trainLogRegress(train_x, train_y, opts):

	numSamples, numFeatures = shape(train_x) #100, 3
	alpha = opts['alpha']
	maxIter = opts['maxIter']
	weights = zeros((numFeatures, 1)) #Here we can choose zeros or ones

	if opts['optimizeType'] == 'gradDescent':
		for k in range(maxIter):
			output = sigmoid(train_x * weights) # This step calculate all the samples, output is an N*1 vector
			error = train_y - output
			weights = weights + alpha * train_x.transpose() * error
	elif opts['optimizeType'] == 'stochgradDescent':
		'''
		This method calculate all the samples for just one time
		'''
		for k in range(numSamples):
			output = sigmoid(train_x[k,:] * weights) # Output is a number
			error = train_y[k,0] - output
			weights = weights + alpha * train_x[k,:].transpose() * error
	elif opts['optimizeType'] == 'smoothStocGradDescent': 
		'''
		This method will randomly choose a sample to calculate
		'''
		for k in range(maxIter):
			dataIndex = range(numSamples)
			for i in range(numSamples):
				alpha = 4/(1.0+k+i)+0.01
				randIndex = int(random.uniform(0,len(dataIndex)))
				output = sigmoid(train_x[randIndex,:] * weights)
				error = train_y[randIndex,0] - output
				weights = weights + alpha * train_x[randIndex,:].transpose() * error
				del(dataIndex[randIndex])

	return weights

def testLogRegression(weights, train_x, train_y):

	numSamples, numFeatures = shape(train_x)
	matchCount = 0
	for i in range(numSamples):
		predict = sigmoid(train_x[i,:] * weights)[0, 0] > 0.5
		if predict == bool(train_y[i,0]):
			matchCount += 1
	accuracy = float(matchCount) / numSamples
	return accuracy

def showLogRegression(weights, train_x, train_y):

	numSamples, numFeatures = shape(train_x)

	for i in range(numSamples):
		if int(train_y[i,0] == 0):
			plt.plot(train_x[i, 1], train_x[i, 2], 'or')
		else:
			plt.plot(train_x[i, 1], train_x[i, 2], 'ob')

	# draw the classify line  
	min_x = min(train_x[:, 1])[0, 0]  
	max_x = max(train_x[:, 1])[0, 0]  
	weights = weights.getA()  # convert mat to array  
	y_min_x = float(-weights[0] - weights[1] * min_x) / weights[2]  
	y_max_x = float(-weights[0] - weights[1] * max_x) / weights[2]  
	plt.plot([min_x, max_x], [y_min_x, y_max_x], '-g')  
	plt.xlabel('X1'); plt.ylabel('X2')  
	plt.show()