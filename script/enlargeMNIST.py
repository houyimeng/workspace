# -*- coding: utf-8 -*-
"""
Created on Mon Mar 21 13:40:54 2016

@author: houyimeng
"""

from ELMtoolbox.MNISTDataset import MNISTDataset
from ELMtoolbox.zeropadding import zeropadding
from numpy import reshape, zeros, vstack, savetxt
import matplotlib.pylab as plt

numtr = 60000
numte = 10000
dataset = MNISTDataset('MNIST')
padsize = 2

newdataset = zeros((numtr+numte, (28+padsize*2)**2))
newlabel = zeros((numtr+numte))

for i in range(numtr):
    print i
    temp = dataset.getTrainingItem(i)
    imgin = reshape(temp[1], (28,28))
    imgout = zeropadding(imgin, padsize, padsize)
    imgvec = imgout.flatten()
    newdataset[i,:] = imgvec
    newlabel[i] = temp[0].argmax()

for j in range(numte):
    print numtr+j
    temp = dataset.getTestingItem(j)
    imgin = reshape(temp[1], (28,28))
    imgout = zeropadding(imgin, padsize, padsize)
    imgvec = imgout.flatten()
    newdataset[numtr+j,:] = imgvec
    newlabel[numtr+j] = temp[0].argmax()

savetxt('MNISTenlarged', newdataset)
savetxt('MNISTenlargedLabel', newlabel)

plt.figure(1)
plt.imshow(reshape(newdataset[9000,:],(32,32)))
