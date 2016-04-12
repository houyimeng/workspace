# -*- coding: utf-8 -*-
"""
Created on Tue Mar 15 17:52:11 2016

@author: zz
"""

#test ELM classification performance of different pre-p methods

from ELM import ELM
from loadImgData import loadImgData
from numpy import vstack, savetxt, loadtxt, reshape, hstack
from prep import *
import matplotlib.pylab as plt

'''
train_data, train_label, test_data, test_label = loadImgData('greyscale', ratio=0.9)
dataset = vstack((train_data, test_data))
labelset = vstack((train_label, test_label))
savetxt('dataset',dataset)
savetxt('labelset', labelset)
'''

dataset0 = loadtxt('dataset')
labelset = loadtxt('labelset')
dataset = zeros((dataset0.shape))
N = dataset0.shape[0]

feature_dim = 12
label_dim = 10
elm = ELM(feature_dim, feature_dim*10, label_dim, 'lite', 'dec')

#dataset0 = normalizeData(dataset0)

dataset = zeros((N, 12))
for i in range(N):
    print "Current iter = ", i, "of", N
    example = dataset0[i,:]
    temp = reshape(example, (28, 28))
    temp_out1 = cooc(temp, 0)
    temp_out2 = cooc(temp, 45)
    temp_out3 = cooc(temp, 90)      
    dataset[i,:4] = featurer(temp_out1)
    dataset[i,4:8] = featurer(temp_out2)
    dataset[i,8:] = featurer(temp_out3)       


threshold = 6000
train_data = dataset[:threshold,:]
test_data = dataset[threshold:,:]
train_label = labelset[:threshold,:]
test_label = labelset[threshold:,:]

elm.trainModel(train_data, train_label)
elm.testModel(test_data, test_label)

plt.figure()
plt.imshow(reshape(dataset0[10,:], (28,28)))

'''

for i in range(N):
    print "Current iter = ", i, "of", N
    temp_before = reshape(dataset0[i,:], (28,28))
    temp_aft = openning(temp_before)
    dataset[i,:] = temp_aft.flatten()
'''