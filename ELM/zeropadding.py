# -*- coding: utf-8 -*-
"""
Created on Tue Mar  1 15:31:44 2016

@author: houyimeng
"""
from numpy import pad

# padding zeros around the original image so that make life easier for filtering

def zeropadding(matrix, padsize1, padsize2, constant_val = 0):
    leftPad,rightPad,topPad,bottomPad = padsize1, padsize1, padsize2, padsize2
    pads = ((leftPad,rightPad),(topPad,bottomPad))
    return pad(matrix, pads, 'constant', constant_values = constant_val)