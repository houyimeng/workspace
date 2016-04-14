# -*- coding: utf-8 -*-
"""
Created on Mon Apr 11 10:49:47 2016

@author: ThinkPad
"""

from numpy import array, random, arange, hstack, reshape
import matplotlib.pylab as plt
import cv2
import os

class loadImgFile(object):
    
    def __init__(self, folder='greyscale', labelrange = arange(-1,10)):
        
        self.train_data = []
        self.train_label = []
        self.test_data = []
        self.test_label = []
        self.folder = folder
        self.itemlist = labelrange
        self.database = None
        self.labelbase = None
    
    def load(self):
        print ">>> Loading Data<<<"    
        database0 = []
        labelbase0 = []
        for i in self.itemlist:
            path = 'C:\\dataspace\\IMGdata\\'+self.folder+'\\'+str(i)+'\\'
            dirs = os.listdir(path)
            for item in dirs:                 
                img = cv2.imread(path+item,0)             
                database0.append( img.flatten() )
                labelbase0.append( i )
        print ">>> Load complete <<<"
        
        database0 = array(database0, dtype='int')
        labelbase0 = reshape(array(labelbase0), (database0.shape[0], 1))
        data_con = hstack((database0, labelbase0))
        random.shuffle(data_con)
        self.database = data_con[:, :database0.shape[1]]
        self.labelbase = data_con[:, database0.shape[1]:]        
                       
        return self.database, self.labelbase
    