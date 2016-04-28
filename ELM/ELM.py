# -*- coding: utf-8 -*-
"""
Created on Tue Jan 19 14:12:06 2016

@author: houyimeng
"""

""" building an ELM framework """

from AbstractNetwork import AbstractNetwork 
from numpy import zeros, tanh, dot, reshape, random
from OPIUM import *
from randbits import randbits
import pickle

class ELM(AbstractNetwork):
    
    def __init__(self, InSize, HidSize, OutSize, OPIUM_type = 'basic', genWeights_type = 'dec'):
        
        AbstractNetwork.__init__(self, InSize, HidSize, OutSize)
        self.genWeights_type = genWeights_type # type of generating weights
        
        if genWeights_type == 'bin':
            self.RandomWeight = randbits(self.HidSize, self.InSize)
        elif genWeights_type == 'dec':
            self.RandomWeight = random.rand(self.HidSize, self.InSize) - 0.5
        else:
            raise Exception("Invalid type of generating random weights")
        
        if OPIUM_type == 'lite':
            self.theta = 1
        elif OPIUM_type == 'basic': 
            self.theta = eye(self.HidSize)
                
        self.LinearWeight = zeros((self.OutSize, self.HidSize))
        self.Test_output_hat = zeros((self.OutSize, 1))
           
    def train(self, train_item, train_label): # training an ELM

        train_item = reshape(train_item, (self.InSize, 1))
        activation = reshape(dot(self.RandomWeight, train_item), (self.HidSize,1)) # calculate activation
        activation = tanh(activation)
        train_output_hat = dot(self.LinearWeight, activation) # observed output
        e = train_label - train_output_hat # deviation        
        OPIUM(activation, e, self.LinearWeight, self.theta)
        
        return train_output_hat
        
    def recall(self, test_item): # testing examples with trained ELM
  
        test_item = reshape(test_item, (self.InSize,1))
        activation = tanh(dot(self.RandomWeight, test_item))
        self.Test_output_hat = dot(self.LinearWeight, activation)
        return self.Test_output_hat    

    @property
    def getLinWeight(self):
        return self.LinearWeight
    
    def getRanWeight(self):
        return self.RandomWeight
        
    def save(self, FilePath):

        data_dict = { 'LinearWeight':self.LinearWeight,\
                      'RandomWeight':self.RandomWeight}
                    
        with open(FilePath, 'wb') as f:
            pickle.dump(data_dict , f)
        
    def load(self, FilePath):
        
        with open(FilePath, 'rb') as f:
            data_dict = pickle.load(f)

        self.LinearWeight = data_dict['LinearWeight']
        self.RandomWeight = data_dict['RandomWeight']
                                                                                                       
        