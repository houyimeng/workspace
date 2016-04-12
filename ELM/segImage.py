# -*- coding: utf-8 -*-
"""
Created on Wed Mar 09 17:53:49 2016

@author: ThinkPad
"""

# divide input data(image) into quarters
from numpy import zeros,sqrt

def segImage(entireData, num):
    
    row = entireData.shape[0]
    col = entireData.shape[1]
        
    outputMatrix = zeros((num, row*col/num))
    
    rowseg = round(row/sqrt(num))
    colseg = round(col/sqrt(num))
    
    for i in range(int(sqrt(num))):
        for j in range(int(sqrt(num))):
            
            temp = entireData[i*rowseg:(i+1)*rowseg, j*colseg:(j+1)*colseg]
            outputMatrix[i*int(sqrt(num))+j,:] = temp.flatten()
            
    
    return outputMatrix
    