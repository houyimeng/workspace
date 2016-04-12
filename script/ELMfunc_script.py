# -*- coding: utf-8 -*-
"""
Created on Thu Mar 10 09:45:54 2016

@author: ThinkPad
"""

from ELM.MNISTDataset import MNISTDataset
from ELMfunc import ELMtrain, ELMrecall
from ELM.randbits import randbits
from numpy import zeros, random, loadtxt, arange
from ELM.ELM import ELM
import matplotlib.pylab as plt
from distCal import distCal


dataset = MNISTDataset("MNIST")
testdata = loadtxt('newimgdata')

Insize = 28*28
Hidsize = 28*28*10
Outsize = 10
method = 'lite'
datatype = 'dec'

#anELM = ELM(Insize, Hidsize, Outsize, method, datatype)
#anELM.load('WellTrainedELMWeights')


#RanW = anELM.getRanWeight()
#LinW = anELM.getLinWeight
mat = []

outputhatM = zeros((10,10))

label_tr = zeros(10)
label_te = arange(0, 10)

if datatype == 'bin':
    RanW = randbits(Hidsize, Insize)  
elif datatype == 'dec':
    RanW = random.rand(Hidsize, Insize) - 0.5      
LinW = zeros((Outsize, Hidsize))


for i in range(9,-1,-1):
    print i
    item_tr = testdata[i,:].flatten()
    label_tr = zeros((10,1))
    label_tr[i] = 1 
    RanW, LinW = ELMtrain(item_tr, label_tr, RanW, LinW, Insize, Hidsize, Outsize, method, datatype)

    
for j in range(9,-1,-1):
    print j     
    temp = testdata[j,:].flatten()       
    outputhat = ELMrecall(temp, RanW, LinW, Insize)
    outputhatM[j,:] = outputhat.flatten()
    if ( distCal(outputhat) == label_te[j] ): 
        print "find one"
        mat.append(0)

acc = len(mat)/float(10)
print acc

plt.figure(1)
plt.imshow(outputhatM)
plt.figure(2)
plt.plot(outputhat)




