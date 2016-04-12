# -*- coding: utf-8 -*-
"""
Created on Thu Mar 17 17:45:17 2016

@author: houyimeng
"""
from ImageToolbox.synthesisImage import synthesisImage
import matplotlib.pylab as plt
from numpy import reshape 
import random

path = 'C:\\ELMframework\\harbour\\lida\\'

img_size = (18,20)
replicates = 1
noiser = 5

data_tr = synthesisImage(path, img_size, replicates, noiser)

plt.figure(1)
idx = random.randint(0, replicates*10-1)
plt.imshow( reshape(data_tr[idx,:], (20, 18)))

