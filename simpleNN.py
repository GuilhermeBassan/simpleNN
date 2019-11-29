# -*- coding: utf-8 -*-
"""
Simple NN - calculating XOR function

Created on Fri Nov 29 19:55:59 2019

@author: guilherme
"""
import numpy as np

# Creates a function for the sigmoid activation
def sigmoid(sum):
    return 1 / (1 + np.exp(sum))

# Creates a function for the derivative
def sigmoidDelta(sig):
    return sig * (1 - sig)

# Array with the input vectors
xorInput = np.array([[0, 0],
                     [0, 1],
                     [1, 0],
                     [1, 1]])

# Array with the output constants
xorOutput = np.array([[0],
                      [1],
                      [1],
                      [0]])

# Array with the random generated weights for the
# hidden layer input
weightsInput = np.array([[-0.424, -0.704, -0.961],
                        [0.358, -0.577, -0.469]])

# Array with the random generated weights for the
# hidden layer output
weightsOutput = np.array([[-0.017], 
                          [-0.893],
                          [0.148]])

epochs = 1000000 # Number of training epochs
learningRate = 0.3 # Learning rate
momentum = 1 # Momentum of scan

# The neural network - it adjusts the 
# weights for the determined number of epochs
for j in range(epochs):
    entryLayer = xorInput
    
    sumSynapsis0 = np.dot(xorInput, weightsInput) # Dot product between 
                                                 # input and weights
    hiddenLayer = sigmoid(sumSynapsis0) # Generating hidden layer values
    
    sumSynapsis1 = np.dot(hiddenLayer, weightsOutput)
    outputLayer = sigmoid(sumSynapsis1)
    
    outputLayerError = xorOutput - outputLayer
    absMean = np.mean(np.abs(outputLayerError))
    print("Erro: " + str(absMean))
    
    outputDeriv = sigmoidDelta(outputLayer)
    outputDelta = outputLayerError * outputDeriv
    
    transposeWeights1 = weightsOutput.T
    outputDeltaXWeights = outputDelta.dot(transposeWeights1)
    hiddenLayerDelta = outputDeltaXWeights * sigmoidDelta(hiddenLayer)
    
    transpodeHiddenLayer = hiddenLayer.T
    newWeightsOutput = transpodeHiddenLayer.dot(hiddenLayerDelta)
    weigtsOutput = ((weightsOutput * momentum) + 
                    (newWeightsOutput * learningRate))
    
    transposeEntryLayer = entryLayer.T
    newWeightsInput = transposeEntryLayer.dot(hiddenLayerDelta)
    weightsInput = ((weightsInput * momentum) + 
                    (newWeightsInput * learningRate))
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    