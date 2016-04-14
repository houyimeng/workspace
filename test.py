# -*- coding: utf-8 -*-
"""
Created on Tue Apr 12 16:47:16 2016

@author: ThinkPad
"""

from ELM.genData2 import genData2
import matplotlib.pylab as plt
from model.ELM import ELM
from numpy import zeros, reshape

obj = genData2()
elm = ELM(784, 7840, 24, 'lite', 'dec')

dataset = obj.Dataset
label0 = obj.Label

N = dataset.shape[0]
labelset = zeros((N, 24))
for i in range(N):
    labelset[i, label0[i]-10] = 1   
    
threshold = 3000
train_data = dataset[:threshold,:]
test_data = dataset[threshold:,:]
train_label = labelset[:threshold,:]
test_label = labelset[threshold:,:]

elm.trainModel(train_data, train_label)
elm.testModel(test_data, test_label)

plt.figure()
plt.imshow(reshape(dataset[10,:], (28,28)))


