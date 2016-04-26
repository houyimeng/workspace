# -*- coding: utf-8 -*-
"""
Created on Fri Apr 08 10:15:52 2016

@author: ThinkPad
"""

from numpy import zeros, log2, dot, uint8, ones
import cv2

def cooc(inputmtx, angle, greylevel=16, norm_flag=1):
    a,b = inputmtx.shape
    inputmatrix = zeros(inputmtx.shape)
    cocurrentmtx = zeros((256/greylevel, 256/greylevel))
    
    for x in range(inputmtx.shape[0]):
        for y in range(inputmtx.shape[1]):            
            inputmatrix[x,y] = round(inputmtx[x,y]/greylevel )   

    if angle == 0:
        for i in range(a):
            for j in range(b-1):
                leftitem = inputmatrix[i,j]
                rightitem = inputmatrix[i,j+1]
                cocurrentmtx[leftitem-1,rightitem-1] += 1
                cocurrentmtx[rightitem-1,leftitem-1] += 1
    elif angle == 45:
        for i in range(a-1):
            for j in range(1,b):
                upperrightitem = inputmatrix[i,j]
                lowerleftitem = inputmatrix[i+1,j-1]
                cocurrentmtx[upperrightitem-1,lowerleftitem-1] += 1
                cocurrentmtx[lowerleftitem-1,upperrightitem-1] += 1
    elif angle == 90:
        for i in range(a-1):
            for j in range(b):
                upperitem = inputmatrix[i,j]
                loweritem = inputmatrix[i+1,j]
                cocurrentmtx[upperitem-1,loweritem-1] += 1
                cocurrentmtx[loweritem-1,upperitem-1] += 1 
    elif angle == 135:
        for i in range(a-1):
            for j in range(b-1):
                upperleftitem = inputmatrix[i,j]
                lowerrightitem = inputmatrix[i+1,j+1]
                cocurrentmtx[upperleftitem-1,lowerrightitem-1] += 1
                cocurrentmtx[lowerrightitem-1,upperleftitem-1] += 1                     
    if norm_flag == 1:
        return cocurrentmtx/float(sum(sum(cocurrentmtx)))
    else:
        return cocurrentmtx
        
def featurer(coocmtx):
    # coocmtx is cooccurent matrix
    a,b = coocmtx.shape
    mu_a = coocmtx.mean(1)
    mu_b = coocmtx.mean(0)
    sigma_a = coocmtx.std(1)
    sigma_b = coocmtx.std(0)    
    con_val, ent_val, corr_val = 0,0,0
    
    asm_val = sum(sum(coocmtx*coocmtx))
    for i in range(a):
        for j in range(b):
            con_val += coocmtx[i,j]*((i-j)**2)            
            if coocmtx[i,j] !=0:
                ent_val += coocmtx[i,j]*log2(coocmtx[i,j])
            corr_val += i*j*coocmtx[i,j]
            
    corr_val = (corr_val-dot(mu_a,mu_b))/dot(sigma_a,sigma_b)
    ent_val = -ent_val
    
    return asm_val, con_val, ent_val, corr_val
    
def openning(inputmatrix, wind_size = (3,3)):
    
    kernel = ones(wind_size, uint8)
    erosion = cv2.erode(inputmatrix, kernel)
    dilation = cv2.dilate(erosion, kernel)
    return dilation
    
def closing(inputmatrix, wind_size = (3,3)):
    
    kernel = ones(wind_size, uint8)
    dilation = cv2.dilate(inputmatrix, kernel)
    erosion = cv2.erode(dilation, kernel)
    return erosion
    
def normalize(data):
    data_mean = data.mean(axis = 0)
    data_std = data.std(axis = 0) 
    data_std[data_std == 0] = 1   
    return (data - data_mean)/data_std    

    
    
