# -*- coding: utf-8 -*-
"""
Created on Wed Mar 30 09:29:55 2016

@author: houyimeng
"""

from numpy import where

def normalize(data_input):
    # each row of data_input is an example
    
    temp_mean = data_input.mean(0)
    temp_std = data_input.std(0)
    temp_std[where(temp_std == 0)] = 1   
    data_out = (data_input-temp_mean)/temp_std
    
    return data_out
