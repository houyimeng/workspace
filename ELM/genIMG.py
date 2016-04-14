# -*- coding: utf-8 -*-
"""
Created on Tue Apr 12 16:35:23 2016

@author: ThinkPad
"""

from numpy import random, array, reshape, hstack, arange, zeros
import cv2
import os

class genIMG(object):
    
    def __init__(self, folder='greyscale', labelrange = arange(-1,34), scalefactor = 1.4):
        
        self.numLabel = len(labelrange)
        self.folder = folder
        self.itemlist = labelrange

        print ">>> Loading Data<<<"    
        database0 = []
        labelbase0 = []
        for i in self.itemlist:
            path = 'C:\\dataspace\\IMGdata\\'+self.folder+'\\'+str(i)+'\\'
            dirs = os.listdir(path)
            for item in dirs:                 
                img = cv2.imread(path+item,0)
                img = cv2.resize(img, (0,0), fx=1/scalefactor, fy=1/scalefactor)
                database0.append( img.flatten() )
                labelbase0.append( i )
        print ">>> Load complete <<<"
        
        database0 = array(database0, dtype='int')
        labelbase0 = array(labelbase0)
        labelbase = zeros((database0.shape[0], self.numLabel))
        for j in range(database0.shape[0]):
            if labelbase0[j] == -1:
                temp = self.numLabel-1
            else:
                temp = labelbase0[j]                
            labelbase[j, temp] = 1
        
        data_con = hstack((database0, labelbase))
        random.shuffle(data_con)
        self.dataset = data_con[:, :database0.shape[1]]
        self.labelset = data_con[:, database0.shape[1]:]     
        
        self.numSum = self.dataset.shape[0]        
        shuffle_idx = random.permutation(self.numSum)
        self.dataset = self.dataset[shuffle_idx]
        self.labelset = self.labelset[shuffle_idx]

    @property
    def data(self):
        return self.dataset
    @property
    def label(self):
        return self.labelset
    
 
        

    
