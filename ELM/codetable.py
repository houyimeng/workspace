# -*- coding: utf-8 -*-
"""
Created on Thu Apr 14 16:22:59 2016

@author: ThinkPad
"""
from numpy import arange, where

def find(lst, tar):
    return [i for i, x in enumerate(lst) if x == tar]

def codetable(inputlabel):
    truelabel = ['0','1','2','3','4','5','6','7','8','9',\
    'A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','R','S','T','U','V','W','X','Y']
    number = arange(0, 34)   
    idx = (inputlabel == number).tolist().index(True)
    return truelabel[idx]
    