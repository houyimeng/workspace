# -*- coding: utf-8 -*-
"""
Created on Tue Apr 05 18:07:59 2016

@author: ThinkPad
"""

# dilation and erosion
from numpy import zeros, array, ones
from zeropadding import zeropadding

def dilation(inputdata):
    
    flt_arm = 1
    inputmatrix = zeropadding(inputdata, flt_arm, flt_arm, -1)
    outputmatrix = zeros(inputmatrix.shape)
    m, n = inputmatrix.shape
    win = ones((flt_arm+1, flt_arm+1))
        
    for row in range(flt_arm, m-flt_arm):
        for col in range(flt_arm, n-flt_arm):
            
            data_window_ed = win*inputmatrix[row-flt_arm:row+flt_arm+1, col-flt_arm:col+flt_arm+1]
            if data_window_ed.flatten.sum(0) == 9:
                outputmatrix[row-flt_arm:row+flt_arm+1, col-flt_arm:col+flt_arm+1] = 1
            else:
                continue
            
    return outputmatrix

def erosion(inputdata):
    
    flt_arm = 1
    inputmatrix = zeropadding(inputdata, flt_arm, flt_arm, -1)
    outputmatrix = zeros(inputmatrix.shape)
    m, n = inputmatrix.shape
    win = ones((flt_arm+1, flt_arm+1))
        
    for row in range(flt_arm, m-flt_arm):
        for col in range(flt_arm, n-flt_arm):
            
            data_window_ed = win*inputmatrix[row-flt_arm:row+flt_arm+1, col-flt_arm:col+flt_arm+1]
            if data_window_ed.flatten.sum(0) == 9:
                outputmatrix[row-flt_arm:row+flt_arm+1, col-flt_arm:col+flt_arm+1] = 1
            else:
                continue
            
    return outputmatrix