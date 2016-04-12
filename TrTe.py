# -*- coding: utf-8 -*-
"""
Created on Wed Mar 09 12:01:19 2016

@author: ThinkPad
"""

from ELM.ELM import ELM
from ELM.distCal import distCal
from numpy import reshape
from ELM.genData import genData
from ELM.normalize import normalize
import time


# training
obj = genData('binary')
dataset, label = obj.getdata
num = dataset.shape[0]
numTr = int(num*0.9)
numTe = int(num*0.1)
numLabel = 10
#dataset = normalize(dataset)
result = []
ELMobj = ELM(18*18, 18*18*10, numLabel)

tic = time.time()
for i in range(numTr):
    if i%1000 == 0:
        print "Current training iteration is", i+1, "of", numTr
    ELMobj.train(dataset[i,:], reshape(label[i,:], (numLabel,1)))
    
for j in range(numTe):
    if j%1000 == 0:
        print "Current testing iteration is", j+1, "of", numTe
    labelhat0 = ELMobj.recall(dataset[j,:])
    labeltrue = distCal( reshape(label[j,:], (numLabel,1)) )
    labelhat = distCal(labelhat0)    
    result.append(labeltrue == labelhat)
        
Acc = result.count(True)/ float(numTe)

toc = time.time()
print "Time Elapsed =", toc-tic
print "The classification accuracy is:", Acc

ELMobj.save('C:\\ELMframework\\w8\\koutu18')
    

    

       