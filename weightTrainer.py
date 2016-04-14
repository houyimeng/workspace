# -*- coding: utf-8 -*-
"""
Created on Wed Mar 09 12:01:19 2016

@author: ThinkPad
"""

from ELM.ELM import ELM
from ELM.distCal import distCal
from numpy import reshape, uint8, arange, sqrt
from ELM.genIMG import genIMG
import time
import cv2
import matplotlib.pylab as plt

# training
obj = genIMG(labelrange = arange(-1,34), scalefactor = 1.6)
dataset = obj.data
label = obj.label
num = dataset.shape[0]
dim = int(sqrt(dataset.shape[1]))
numTr = int(num*0.99)
numTe = int(num*0.01)
numLabel = 35
result = []

hog = cv2.HOGDescriptor((dim, dim), (6,6), (3,3), (3,3), 9)
hist = hog.compute( uint8(reshape(dataset[2,:], (dim,dim))) )
ELMobj = ELM(hist.size, hist.size*10, numLabel)

tic = time.time()
for i in range(numTr):
    if i%1000 == 0:
        print "Current training iteration is", i+1, "of", numTr
    temp_img = uint8(reshape(dataset[i,:], (dim, dim)))
    hist = hog.compute( temp_img )
    ELMobj.train(hist, reshape(label[i,:], (numLabel,1)))
    
for j in range(numTe):
    if j%1000 == 0:
        print "Current testing iteration is", j+1, "of", numTe
    temp_img = uint8(reshape(dataset[j,:], (dim, dim)))
    hist = hog.compute( temp_img )
    labelhat0 = ELMobj.recall(hist)
    labeltrue = distCal( reshape(label[j,:], (numLabel,1)) )
    labelhat = distCal( labelhat0 )    
    result.append(labeltrue == labelhat)
        
Acc = result.count(True)/ float(numTe)

toc = time.time()
print "Time Elapsed =", toc-tic
print "The classification accuracy is:", Acc

ELMobj.save('C:\\dataspace\\weights\\harbour35_1818')
plt.figure()
plt.imshow( reshape(dataset[2,:], (dim,dim)))


    

       