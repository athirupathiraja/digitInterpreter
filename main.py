#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 13 17:34:31 2021

@author: athirupathiraja
"""

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

data = pd.read_csv('/Users/athirupathiraja/Downloads/train.csv')

data = np.array(data)
m, n = data.shape
np.random.shuffle(data)

dataTest = data[0:1000].T
YTest = dataTest[0]
XTest = dataTest[1:n]

dataTrain = data[1000:m].T
YTrain = dataTrain[0]
XTrain = dataTrain[1:n]
XTrain = XTrain / 255


def initParams():
    W1 = np.random.rand(10, 784) - 0.5
    b1 = np.random.rand(10, 1) - 0.5
    W2 = np.random.rand(10, 10) - 0.5
    b2 = np.random.rand(10, 1) - 0.5
    return W1, b1, W2, b2


def ReLU(Z):
    return np.maximum(0, Z)


def tanH(Z):
    return (np.exp(Z) - np.exp(-Z)) / (np.exp(Z) + np.exp(-Z))


def sigmoid(Z):
    return 1 / (1 + np.exp(-Z))


def softmax(Z):
    return np.exp(Z) / sum(np.exp(Z))


def forwardProp(W1, b1, W2, b2, X):
    Z1 = W1.dot(X) + b1
    A1 = ReLU(Z1)
    Z2 = W2.dot(A1) + b2
    A2 = softmax(Z2)
    return Z1, A1, Z2, A2


def oneHot(Y):
    oneHotY = np.zeros((Y.size, Y.max() + 1))
    oneHotY[np.arange(Y.size), Y] = 1
    oneHotY = oneHotY.T
    return oneHotY


def derivativeReLU(Z):
    return Z > 0


def derivativeTanH(Z):
    return 1 - tanH(Z) * tanH(Z)


def derivativeSigmoid(Z):
    return sigmoid(Z) * (1 - sigmoid(Z))


def backwardProp(Z1, A1, Z2, A2, W2, X, Y):
    m = Y.size
    oneHotY = oneHot(Y)
    dZ2 = A2 - oneHotY
    dW2 = 1 / m * dZ2.dot(A1.T)
    db2 = 1 / m * np.sum(dZ2)
    dZ1 = W2.T.dot(dZ2) * derivativeReLU(Z1)
    dW1 = 1 / m * dZ1.dot(X.T)
    db1 = 1 / m * np.sum(dZ1)
    return dW2, db2, dW1, db1


def updateParams(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha):
    W1 = W1 - alpha * dW1
    b1 = b1 - alpha * db1
    W2 = W2 - alpha * dW2
    b2 = b2 - alpha * db2
    return W1, b1, W2, b2


def getPredictions(A2):
    return np.argmax(A2, 0)


def getAccuracy(predictions, Y):
    print(predictions, Y)
    return np.sum(predictions == Y) / Y.size


def gradientDescent(X, Y, iterations, alpha):
    W1, b1, W2, b2 = initParams()
    for i in range(iterations):
        Z1, A1, Z2, A2 = forwardProp(W1, b1, W2, b2, X)
        dW2, db2, dW1, db1 = backwardProp(Z1, A1, Z2, A2, W2, X, Y)
        W1, b1, W2, b2 = updateParams(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha)
        if (i % 10 == 0):
            print("Iterations: ", i)
            predictions = getPredictions(A2)
            print("Accuracy: ", getAccuracy(predictions, Y))
    return W1, b1, W2, b2


W1, b1, W2, b2 = gradientDescent(XTrain, YTrain, 1000, 0.2)


def makePredictions(X, W1, b1, W2, b2):
    _, _, _, A2 = forwardProp(W1, b1, W2, b2, X)
    predictions = getPredictions(A2)
    return predictions


def testPrediction(index, W1, b1, W2, b2):
    current_image = XTrain[:, index, None]
    prediction = makePredictions(XTrain[:, index, None], W1, b1, W2, b2)
    label = YTrain[index]
    print("Prediction: ", prediction)
    print("Label: ", label)

    current_image = current_image.reshape((28, 28)) * 255
    plt.gray()
    plt.imshow(current_image, interpolation='nearest')
    plt.show()


testPrediction(37, W1, b1, W2, b2)
