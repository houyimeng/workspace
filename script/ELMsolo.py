# -*- coding: utf-8 -*-
"""
Created on Wed Mar 09 12:01:19 2016

@author: ThinkPad
"""

from multiprocessing import pool
from ELM.ELM import ELM
from ELM.MNISTDataset import MNISTDataset
import time

tic = time.time()

# training
numTr = 6000
numTe = 1000
dataset = MNISTDataset("MNIST")
result = []

ELMobj = ELM(28*28, 28*28*10, 10)


for i in range(numTr):
    if i%1000 == 0:
        print "Current training iteration is", i+1, "of", numTr
    labelTr, itemTr = dataset.getTrainingItem(i)
    ELMobj.train(itemTr, labelTr)

for j in range(numTe):
    if j%1000 == 0:
        print "Current training iteration is", j+1, "of", numTe
    labelTe, itemTe = dataset.getTestingItem(j)
    labelhat0 = ELMobj.recall(itemTe)
    labeltrue = labelTe.argmax()
    labelhat = labelhat0.argmax()    
    result.append(labeltrue == labelhat)
        
Acc = result.count(True)/ float(numTe)

toc = time.time()
print "Time Elapsed =", toc-tic
print "The classification accuracy is:", Acc

        
    

    

       