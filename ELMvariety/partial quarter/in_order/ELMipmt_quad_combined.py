# -*- coding: utf-8 -*-
"""
Created on Thu Jan 28 09:56:18 2016

@author: houyimeng
"""
# quarter ELM
# the output of each ELM are combined

from MNISTdataset import MNISTdataset
from ELM import *
import numpy
import time

tic = time.time()

# init
size_input = 14*14
size_output = 10
Factor = 5

N_train = 6000
N_test = 1000
genWeights_type = 'bin' # 'bin' for binary weights and 'dec' for float weights
OPIUM_type = 'basic' #'basic' or 'lite'

elist = []

img00 = zeros((14,14))
img01 = zeros((14,14))
img10 = zeros((14,14))
img11 = zeros((14,14))

N_sim = 2
class_errors = zeros((N_sim, 1))
Accuracy = zeros((N_sim, 1))

MNIST = MNISTdataset("data")

for ii in range(N_sim):
    
    print "#### Current No. of Simulation: %d of %d ####"%(ii+1, N_sim)
    print "..."
    
    # init class
    q1 = ELM(size_input, size_output, Factor, OPIUM_type, genWeights_type)
    q2 = ELM(size_input, size_output, Factor, OPIUM_type, genWeights_type)
    q3 = ELM(size_input, size_output, Factor, OPIUM_type, genWeights_type)
    q4 = ELM(size_input, size_output, Factor, OPIUM_type, genWeights_type)

    for i in range(N_train):
        label_train, item_train = MNIST.GetTrainingItem(i)
        item_train = reshape(item_train,(28,28))
        
        # the way that divide the image       
        for j in range(14):
            for k in range(14): 
                
                img00[k][j] = item_train[k*2][j*2]
                img01[k][j] = item_train[1+k*2][j*2]
                img10[k][j] = item_train[k*2][1+j*2]
                img11[k][j] = item_train[1+k*2][1+j*2]
                
        outq1 = q1.train(img00, label_train)
        outq2 = q2.train(img01, label_train)
        outq3 = q3.train(img10, label_train)
        outq4 = q4.train(img11, label_train)
        
        if (i+1)%100 == 0:        
            print "Current # of training iter is %d of %d"%(i+1, N_train)
            
    print ">>> Training complete <<<"
            
    for i in range(N_test):
                
        label_test, item_test = MNIST.GetTestingItem(i)
        item_test = reshape(item_test,(28,28))
        
        for j in range(14):
            for k in range(14):
                img00[k][j] = item_test[k*2][j*2]
                img01[k][j] = item_test[1+k*2][j*2]
                img10[k][j] = item_test[k*2][1+j*2]
                img11[k][j] = item_test[1+k*2][1+j*2]
                
        output_hat1 = q1.recall(img00)
        output_hat2 = q2.recall(img01)
        output_hat3 = q3.recall(img10)
        output_hat4 = q4.recall(img11)
        
        output_hat_matrix = numpy.hstack((output_hat1, output_hat2, output_hat3, output_hat4))
        
        output_hat = output_hat_matrix.mean(1)
        
        output_hat_max = output_hat.argmax()   
        output_max = label_test.argmax()

        if (output_hat_max!= output_max):
            class_errors[ii] +=1
            elist.append(i)
        
        if (i+1)%100 == 0:
            print "Current # of testing iter is %d of %d"%(i+1, N_test)

    elist.append(-1)
    print ">>> Testing complete <<<"

for l in range(N_sim):
    Accuracy[l] = 1-class_errors[l]/float(N_test)

toc = time.time()
timeElapsed = toc-tic

mean_acc = Accuracy.mean()

print ""
print "TimeElapsed = %6.2f seconds "%timeElapsed
print "The Accuracy is :",'\n', Accuracy 
print "The mean Accuracy is :", mean_acc  

#q2.save('q2.pickle')
#qq = ELM(size_input, size_output, Factor, OPIUM_type, genWeights_type)
#qq.load('q2.pickle')