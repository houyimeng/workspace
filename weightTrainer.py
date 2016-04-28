# -*- coding: utf-8 -*-
"""
Created on Wed Mar 09 12:01:19 2016

@author: ThinkPad
"""

from ELM.ELM import ELM
from ELM.distCal import *
from numpy import reshape, uint8, sqrt
from ELM.genIMG import genIMG
import time
import cv2
import matplotlib.pylab as plt

# training
obj = genIMG(1.166666)
dataset = obj.data
label = obj.label
num = dataset.shape[0]
dim = int(sqrt(dataset.shape[1]))
numTr = int(num*0.99)
numTe = int(num*0.01)
numLabel = 34
result = []

hog = cv2.HOGDescriptor((dim, dim), (8,8), (4,4), (4,4), 9)
hist = hog.compute( uint8(reshape(dataset[2,:], (dim,dim))) )
ELMobj = ELM(hist.size, hist.size*10, numLabel)

plt.figure()
plt.imshow( reshape(dataset[0,:], (dim,dim)))

tic = time.time()
for i in range(numTr):
    print "Current training iteration is", i+1, "of", numTr
    temp_img = uint8(reshape(dataset[i,:], (dim, dim)))
    hist = hog.compute( temp_img )
    ELMobj.train(hist, reshape(label[i,:], (numLabel,1)))
    
for j in range(numTe):
    print "Current testing iteration is", j+1, "of", numTe
    temp_img = uint8(reshape(dataset[j,:], (dim, dim)))
    hist = hog.compute( temp_img )
    labelhat0 = ELMobj.recall(hist)
    labeltrue = distCal2( reshape(label[j,:], (numLabel,1)) )
    labelhat = distCal2( labelhat0 )    
    result.append(labeltrue == labelhat)
        
Acc = result.count(True)/ float(numTe)

toc = time.time()
print "Time Elapsed =", toc-tic
print "The classification accuracy is:", Acc

ELMobj.save('C:\\dataspace\\weights\\harbour34_24_basic')


# training
obj = genIMG(1.0)
dataset = obj.data
label = obj.label
num = dataset.shape[0]
dim = int(sqrt(dataset.shape[1]))
numTr = int(num*0.99)
numTe = int(num*0.01)
numLabel = 34
result = []

hog = cv2.HOGDescriptor((dim, dim), (8,8), (4,4), (4,4), 9)
hist = hog.compute( uint8(reshape(dataset[2,:], (dim,dim))) )
ELMobj = ELM(hist.size, hist.size*10, numLabel)

plt.figure()
plt.imshow( reshape(dataset[0,:], (dim,dim)))

tic = time.time()
for i in range(numTr):
    print "Current training iteration is", i+1, "of", numTr
    temp_img = uint8(reshape(dataset[i,:], (dim, dim)))
    hist = hog.compute( temp_img )
    ELMobj.train(hist, reshape(label[i,:], (numLabel,1)))
    
for j in range(numTe):
    print "Current testing iteration is", j+1, "of", numTe
    temp_img = uint8(reshape(dataset[j,:], (dim, dim)))
    hist = hog.compute( temp_img )
    labelhat0 = ELMobj.recall(hist)
    labeltrue = distCal2( reshape(label[j,:], (numLabel,1)) )
    labelhat = distCal2( labelhat0 )    
    result.append(labeltrue == labelhat)
        
Acc = result.count(True)/ float(numTe)

toc = time.time()
print "Time Elapsed =", toc-tic
print "The classification accuracy is:", Acc

ELMobj.save('C:\\dataspace\\weights\\harbour34_28_basic')


    

       