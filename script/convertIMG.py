# -*- coding: utf-8 -*-
"""
Created on Tue Mar 15 15:53:52 2016

@author: ThinkPad
"""

from PIL import Image
from numpy import zeros, array, reshape, savetxt, mean, arange, eye
import matplotlib.pylab as plt
import os, sys


data = []
data_label = []

for i in range(10):
    path = 'C:\\ELMframework\\Ningbo\\'+str(i)+'\\'
    dirs = os.listdir(path)
    
    counts = 0
    for item in dirs:
    
        img = Image.open(path+item).convert("L")
        newimg = img.resize((28, 28))
        meanval = mean(mean(newimg.getdata()))
        temp =  array ((newimg.getdata()- meanval)/float(255) > 0, dtype =int)
        
        if counts <= 21:
            data.append(temp)
            data_label.append(i)
        counts += 1


num = 90
plt.figure(1)
plt.imshow(reshape(data[num],(28,28)))
print 'Label is ', data_label[num]


dataset = zeros((len(data), 28*28))
dataset_label = zeros((len(data), 10))

for i in range(len(data)):
    dataset[i,:] = reshape( array(data[1]), 28*28)
    temp = zeros(10)
    temp[data_label[i]] = 1
    dataset_label[i,:] = temp

