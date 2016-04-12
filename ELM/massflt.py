# -*- coding: utf-8 -*-
"""
Created on Fri Apr 01 16:12:26 2016

@author: ThinkPad
"""
from numpy import ones
from zeropadding import zeropadding

def massflt(inputdata, threshold = 20):
    
    flt_arm = 2
    inputmatrix = zeropadding(inputdata, flt_arm, flt_arm, -1)
    m, n = inputmatrix.shape
    outdata = -1*ones(inputdata.shape)
        
    for row in range(flt_arm, m-flt_arm):
        for col in range(flt_arm, n-flt_arm):
            data_window = inputmatrix[row-flt_arm:row+flt_arm+1, col-flt_arm:col+flt_arm+1].flatten()
            
            temp = data_window.tolist().count(-1)
            if temp <= threshold:
                outdata[row-flt_arm, col-flt_arm] = inputmatrix[row-flt_arm, col-flt_arm]
            else:
                continue
                
    return outdata
            
        