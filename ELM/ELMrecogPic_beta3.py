
from ELM import ELM
import matplotlib.pylab as plt 
from numpy import zeros, where, vstack, array, ones, argsort, absolute, mean, reshape
from fragImg import fragImg
import math
import cv2
import os

class ELMrecogPic_beta3(object):
    
    def __init__(self, lane = 'v'):
        
        self.patcharm = 28
        self.Outsize = 2 
        self.newELM = ELM(self.patcharm**2, (self.patcharm**2)*10, self.Outsize)
        self.newELM.load('C:\\ELMframework\\w8\\binary')
        #self.newELM.load('C:\\ELMframework\\w8\\MNIdata')            
        
        self.truecolormatrix =  None
        self.hatcolormatrix =  None
        self.testcanvas0 = None
        self.labelmatrix0 = None
        self.labelmatrix = None
        self.Lane = lane
   
    def runtest(self):
    
        path = 'C:\\comingdata\\others\\'
        dirs = os.listdir(path)
        for item in dirs: 
            img = cv2.imread(path+item,0)
            #img = img[:int(img.shape[0]*3/4), int(img.shape[1]*1/3):int(img.shape[1]*2/3)]
            #img = cv2.resize(img,(0,0), fx=0.8, fy=0.8)
            self.testcanvas = img
        print img.shape[0], img.shape[1]
        canvasSize = [self.testcanvas.shape[0], self.testcanvas.shape[1]]
        
        self.labelmatrix = zeros((math.ceil(canvasSize[0]/float(self.patcharm)), math.ceil(canvasSize[1]/float(self.patcharm)) ))
        frag_testcanvas = fragImg(self.testcanvas, (self.patcharm, self.patcharm))
        print frag_testcanvas.shape  
        
        for i in range(len(frag_testcanvas)):
            labelpatch = self.newELM.recall( frag_testcanvas[i,:] )
            self.labelmatrix[ i/math.ceil(canvasSize[1]/float(self.patcharm)), i%math.ceil(canvasSize[1]/float(self.patcharm))] = labelpatch.argmax()
        
        return self.labelmatrix, frag_testcanvas
