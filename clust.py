# -*- coding: utf-8 -*-
"""
Created on Thu Apr 14 13:38:45 2016

@author: ThinkPad
"""

from numpy import loadtxt, zeros, sqrt, where, array

halfSize = 14
datamatrix = loadtxt('datamatrix')

center = zeros(datamatrix.shape[1])
center[0] = 1 

n = 2  
for i in range(1,datamatrix.shape[1]):
    coor = [datamatrix[0,i], datamatrix[1,i]]
    dist = zeros(i)

    for j in range(0, i):
        dist[j] = sqrt((coor[0] - datamatrix[0, j])**2 + (coor[1] - datamatrix[1, j])**2)
    print min(dist), dist.argmin()
    if min(dist) <= halfSize:
        center[i] =  center[dist.argmin()]
    else:
        center[i] = n
        n += 1
n -= 1        


def find(lst, tar):
    return [i for i, x in enumerate(lst) if x == tar]
    
    
result = find(center, 4)