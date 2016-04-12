# -*- coding: utf-8 -*-
"""
Created on Thu Mar 17 09:16:02 2016

@author: ThinkPad
"""

from PIL import Image
from numpy import zeros, reshape
import matplotlib.pylab as plt
import os
from ranvec import ranvec
import ImageFilter

def synthesisImage(path, img_size, replicates, noiser):
    
    dirs = os.listdir(path)
    tot_num = img_size[0]*img_size[1]
    n = 0
    outputdata = zeros((replicates*len(dirs), tot_num))
    datamatrix = zeros((len(dirs), tot_num))
    
    # read images from directory and preprocessing it 
    for item in dirs:
    
        img = Image.open(path+item).convert("L")
        # resize the image
        img = img.resize(img_size)        
        # filter the image
        img = img.filter(ImageFilter.SMOOTH)        
        # get content
        datamatrix[n,:] = reshape(img.getdata(), tot_num) /float(255)
        n += 1
    
    for i in range(len(dirs)):
        for j in range(replicates):
            
            temp = datamatrix[i,:]
            randidx1 = ranvec(noiser, 0, tot_num)
            randidx2 = ranvec(noiser, 0, tot_num)    
            temp[randidx1] = 1
            temp[randidx2] = 0
            
            outputdata[i*replicates+j,:] = temp
            
    return outputdata

