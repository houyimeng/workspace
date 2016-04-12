# -*- coding: utf-8 -*-
"""
Created on Thu Mar 24 14:17:12 2016

@author: ThinkPad
"""

from ELMtoolbox.ELM import ELM
from ELMtoolbox.MNISTDataset import MNISTDataset
from numpy import zeros
from ELMtoolbox.distCal import distCal
import math

anELM = ELM(28*28, 28*28*20, 10, 'basic', 'dec')
anELM.load('WellTrainedELMWeights')
dataset = MNISTDataset('MNIST')

result = []
numTe = 10000

for j in range(numTe):
    print "Current testing iteration is", j+1, "of", numTe
    labelTe, itemTe = dataset.getTestingItem(j)
    labelhat0 = anELM.recall(itemTe)
    labeltrue = distCal(labelTe)
    labelhat = distCal(labelhat0)    
    result.append(labeltrue == labelhat)
        
Acc = result.count(True)/ float(numTe)
