# -*- coding: utf-8 -*-

"""
Created on Mon Mar  7 09:49:31 2016

@author: houyimeng
"""

# this file contains the convolving section of ELM

from ELM import ELM
from MNISTCanvasImage import MNISTCanvasImage
import matplotlib.pylab as plt 
from numpy import zeros, array
from zeropadding import zeropadding
import random


class ELMrecogPic_conv(object):
    
    def __init__(self):
        
        self.patcharm = 28
        self.newELM = ELM(InSize = self.patcharm**2, HidSize=(self.patcharm**2)*10, OutSize=11)
        self.newELM.load('HYBRID') 
        #self.newELM.load('rfciwelm')
   
    def runtest(self, numDigits, method = 'line', canvasSize0 = [], paddingMode = 0, edgeSize = 0):
        
        fltSize = 28
        halfSize = 14
        
        if method == 'rand':
            
            imgcanvas = MNISTCanvasImage('MNIST', canvasSize0)    
            self.testcanvas0, labeltrue = imgcanvas.generateImage(numDigits)            
            
        elif method == 'line':   
            
            testcanvas_temp, labeltrue, canvasSize_info = lineMNIST(numDigits, edgeSize)
            
            if paddingMode == 1:
                
                self.testcanvas0 = zeros(canvasSize0)
                idx1 = random.randint(0, canvasSize0[0]-canvasSize_info[0]-1)
                idx2 = random.randint(0, canvasSize0[1]-canvasSize_info[1]-1)
                self.testcanvas0[idx1:idx1+canvasSize_info[0], idx2:idx2+canvasSize_info[1]] = testcanvas_temp
            
            else:
                self.testcanvas0 = testcanvas_temp
                canvasSize0 = canvasSize_info        
        
        ## zeropadding
        testcanvas = zeropadding(self.testcanvas0, halfSize, halfSize)    
        canvasSize = array(canvasSize0) + fltSize        
        
        ## convolving part
        tot_count = (canvasSize[0]-fltSize)*(canvasSize[1]-fltSize)
        labelmatrix = zeros((canvasSize[0]-fltSize, canvasSize[1]-fltSize))
        
        print "> Convolving in progress ..."
        for y in range(halfSize, canvasSize[1]-halfSize):
            for x in range(halfSize, canvasSize[0]-halfSize):

                if ((y-halfSize)*(canvasSize[0]-fltSize)+x-halfSize+1)%500 == 0:
                    print "> Current convolving index is %d ( %d in total )..."\
                    %((y-halfSize)*(canvasSize[0]-fltSize)+x-halfSize+1, tot_count)
                    
                item_temp = testcanvas[x-halfSize:x+halfSize, y-halfSize:y+halfSize]
                labelhatpatch = self.newELM.recall(item_temp)
                p = distCal(labelhatpatch)
                #labelmatrix[(canvasSize[0]-fltSize)*(y-halfSize)+x-halfSize, :] = reshape(labelhatpatch,10)
                labelmatrix[x-halfSize, y-halfSize] = p
                
        return testcanvas, labelmatrix, labeltrue
                

