# -*- coding: utf-8 -*-
"""
Created on Tue Apr 05 15:33:54 2016

@author: ThinkPad
"""

from ELM import ELM
import matplotlib.pylab as plt 
from numpy import zeros, where, vstack, array, ones, argsort, absolute, mean
from kmean import kmean
from zeropadding import zeropadding
from lineIMAGE import lineIMAGE
from lineMNIST import lineMNIST
from distCal import distCal

class ELMrecogPic_beta2(object):
    
    def __init__(self, lane = 'v'):
        
        self.patcharm = 28
        self.Outsize = 10
        self.newELM = ELM(self.patcharm**2, (self.patcharm**2)*10, self.Outsize)
        self.newELM.load('C:\\dataspace\\weights\\')            
        
        self.truecolormatrix =  None
        self.hatcolormatrix =  None
        self.testcanvas0 = None
        self.labelmatrix = None
        self.raw_labelmatrix = None
        self.Lane = lane
   
    def runtest(self, numDigits):
        
        fltSize = 28
        halfSize = 14
        datasource = 'binary' # 'synthesis' or others
        
        if datasource == 'MNIST':            
            self.testcanvas0, labeltrue, canvasSize0 = lineMNIST(numDigits)         
        else:
            self.testcanvas0, labeltrue, canvasSize0 = lineIMAGE(numDigits, datasource)

        ## zeropadding
        testcanvas = zeropadding(self.testcanvas0, halfSize, halfSize)    
        canvasSize = array(canvasSize0) + fltSize
        
        ## convolving part
        tot_count = (canvasSize[0]-fltSize)*(canvasSize[1]-fltSize)
        self.labelmatrix = zeros((canvasSize[0]-fltSize, canvasSize[1]-fltSize))
        self.raw_labelmatrix = zeros((canvasSize[0]-fltSize, canvasSize[1]-fltSize))
        
        print "> Convolving in progress ..."
        for x in range(halfSize, canvasSize[0]-halfSize):
            for y in range(halfSize, canvasSize[1]-halfSize):
                
                if ( (x-halfSize)*(canvasSize[1]-fltSize)+y-halfSize) %500 == 0:
                    print "> Current convolving index is %d ( %d in total )..."%((x-halfSize)*(canvasSize[1]-fltSize)+y-halfSize+1, tot_count)
                item_temp = testcanvas[x-halfSize:x+halfSize, y-halfSize:y+halfSize]
                labelhatpatch = self.newELM.recall(item_temp) 
                    
                self.labelmatrix[x-halfSize,y-halfSize] = distCal(labelhatpatch)               
        
        ## clustering            
        clust_temp = where(self.labelmatrix != -1)    
        datamatrix = vstack((clust_temp[0], clust_temp[1]))
        Con, Iter = True, 0
        separateWidth = 21        
        
        while Con and Iter < (numDigits * 10):
            print "> Clustering entry No.",Iter+1, "..."
            
            clusters, centroids = kmean(datamatrix.T, numDigits)
            Con = False
            
            for i in range(len(centroids)):
                for j in range(len(centroids)):
                    
                    # bound detection 
                    if i != j and absolute(centroids[i][0] - centroids[j][0]) <= separateWidth and  \
                                  absolute(centroids[i][1] - centroids[j][1]) <= separateWidth :
                        Con = True

            Iter += 1
        if Iter < numDigits * 10:        
            print "Clustering Succeed!"
        else:
            print "Clustering failed, latest entry picked"
        
        centroids = array(centroids)
        labelhat0 = zeros(numDigits)
        countnum = zeros((self.Outsize, numDigits))

        # decicsion making   
        if clusters != []:
            for i in range(len(clusters)):
                for j in range(len(clusters[i])):
                    temp = clusters[i][j]
                    distfactor = absolute(temp[0]-centroids[i,0]) + absolute(temp[1]-centroids[i,1]) + 0.01 # euclidean distance
                    countnum [self.labelmatrix[temp[0],temp[1]],i] += 1/distfactor # equalization

                labelhat0[i] = countnum[:,i].argmax()
        else:
            print "No data generated"
            labelhat0 = -1
                
        if self.Lane == 'h':
            centroidindex = argsort(centroids[:,1])
        elif self.Lane == 'v':
            centroidindex = argsort(centroids[:,0])
            
        labelhat = array(labelhat0[centroidindex], dtype='int')
        
        self.truecolormatrix =  (-1)*ones((canvasSize[0]-fltSize, canvasSize[1]-fltSize))
        self.hatcolormatrix  =  (-1)*ones((canvasSize[0]-fltSize, canvasSize[1]-fltSize))
        displaysize = 2        
        # visualize the centroids and their labels
        for i in range(len(centroids)):
            self.truecolormatrix [round(centroids[i][0])-displaysize:round(centroids[i][0])+displaysize, \
                     round(centroids[i][1])-displaysize:round(centroids[i][1])+displaysize ] = labeltrue[i]
                     
        for i in range(len(centroids)):
            self.hatcolormatrix [round(centroids[i][0])-displaysize:round(centroids[i][0])+displaysize, \
                     round(centroids[i][1])-displaysize:round(centroids[i][1])+displaysize ] = labelhat[i]        
        
        print "True labels are:", labeltrue   
        print "Hat  labels are:", labelhat 
        
        return self.labelmatrix        
        
    def visualize(self):
        
        if self.Lane == 'h':
            fig = plt.figure()
            fig.add_subplot(4,1,1)
            plt.imshow(self.testcanvas0)
            fig.add_subplot(4,1,2)
            plt.imshow(self.labelmatrix0)  
            fig.add_subplot(4,1,3)
            plt.imshow(self.truematrix)  
            fig.add_subplot(4,1,4)
            plt.imshow(self.hatcolormatrix)
            
        elif self.Lane == 'v':
            fig = plt.figure()
            fig.add_subplot(1,4,1)
            plt.imshow(self.testcanvas0)
            fig.add_subplot(1,4,2)
            plt.imshow(self.labelmatrix)  
            fig.add_subplot(1,4,3)
            plt.imshow(self.truecolormatrix)  
            fig.add_subplot(1,4,4)
            plt.imshow(self.hatcolormatrix)                              
        