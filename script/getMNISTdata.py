# -*- coding: utf-8 -*-
"""
Created on Wed Mar 23 11:11:17 2016

@author: houyimeng
"""

from ELMtoolbox.MNISTDataset import MNISTDataset
from numpy import zeros, argmax, savetxt, where

dataset = MNISTDataset('MNIST')
datamatrix = zeros((10000, 784))
datalabel = zeros(10000)

for j in range(10000):
    print j
    label, img = dataset.getTestingItem(j)
    datamatrix[j,:] = img
    datalabel[j] = label.argmax()
    
meanval = datamatrix.mean(0)
varval = datamatrix.std(0)
varval[where(varval == 0)] = 1
newdatamatrix = (datamatrix - meanval)/varval


savetxt('normtestdata', newdatamatrix)
savetxt('normtestlabel', datalabel)

     
    