# -*- coding: utf-8 -*-
"""
Created on Wed Mar 16 16:21:54 2016

@author: ThinkPad
"""

from ELM.ELM import ELM
from distCal import distCal
from numpy import loadtxt, reshape, zeros
import matplotlib.pylab as plt
import time

data_tr = loadtxt('dataset_tr')
label_tr = loadtxt('dataset_tr_label')
data_te = loadtxt('dataset_te')
label_te = loadtxt('dataset_te_label')


# training
numTr = 220
numTe = 10
result = []
labelmatrix = zeros((numTe, 10))

ELMobj = ELM(28*28, 28*28*5, 10, 'basic', 'dec')

tic = time.time()
for i in range(numTr):
    print "Current training iteration is", i+1, "of", numTr
    temp0 = data_tr[i,:]
    temp1 = reshape(label_tr[i,:], (10,1))
    ELMobj.train(temp0, temp1)
    

for j in range(numTe):
    print "Current testing iteration is", j+1, "of", numTe

    labelhat0 = ELMobj.recall(data_te[j,:])
    labeltrue = distCal(label_te[j,:])
    labelhat = distCal(labelhat0)    
    result.append(labeltrue == labelhat)
    labelmatrix[j,:] = reshape(labelhat0, 10)
        
Acc = result.count(True)/ float(numTe)

toc = time.time()
print "Time Elapsed =", toc-tic
print "The classification accuracy is:", Acc

plt.figure(1)
plt.imshow(labelmatrix)

#counts = array([22, 118, 142, 38, 68, 54, 55, 43, 51, 39])
#plt.figure(2)
#plt.plot(counts)


#temp = data_tr.tolist()


