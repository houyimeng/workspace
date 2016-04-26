# -*- coding: utf-8 -*-
"""
Created on Thu Mar 10 11:56:41 2016

@author: ThinkPad
"""

from numpy import tanh, dot, reshape, eye
from OPIUM import *

def train(train_item, train_label, RandomWeight, LinearWeight, Insize, Hidsize, Outsize, method, datatype):
       
    # training
    train_item = reshape(train_item, (Insize, 1))
    activation = reshape(dot(RandomWeight, train_item), (Hidsize, 1)) # calculate activation
    activation = tanh(activation)
    train_output_hat = dot(LinearWeight, activation) # observed output
    e = train_label - train_output_hat # deviation
        
    if method == 'basic':
        Theta = eye(Hidsize)
        OPIUM(activation, e, LinearWeight, Theta)
    elif method == 'lite':
        Theta = 1
        OPIUMl(activation, e, LinearWeight, Theta)
    
    return RandomWeight, LinearWeight    
    
    # testing    
def recall(test_item, RandomWeight, LinearWeight, Insize):
     test_item = reshape(test_item, (Insize,1))
     activation = tanh(dot(RandomWeight, test_item))
     Output_hat = dot(LinearWeight, activation)
     
     return Output_hat
        