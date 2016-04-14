# -*- coding: utf-8 -*-
"""
Created on Mon Apr 11 10:20:13 2016

@author: ThinkPad
"""

from zhao.ELM import ELM
from loadImgFile import loadImgFile
from numpy import zeros
from ELM.prep import *
import matplotlib.pylab as plt

qwe = loadImgFile()
dataset, labels = qwe.load(14000)
labelset = zeros(( len(labels), 2 ))

for i in range(len(labels)):
    if labels[i] == -1:
        labelset[i,0] = 1
    else:
        labelset[i,1] = 1

elm = ELM(28*28, 28*28*10, 2, 'lite', 'dec')

threshold = 13500
train_data = dataset[:threshold,:]
test_data = dataset[threshold:,:]
train_label = labelset[:threshold,:]
test_label = labelset[threshold:,:]

elm.trainModel(train_data, train_label)
elm.testModel(test_data, test_label)
elm.save('C:\\dataspace\\weights\\binary')
