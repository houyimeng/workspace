# -*- coding: utf-8 -*-
"""
Created on Fri Apr  1 09:12:43 2016

@author: houyimeng
"""

from ELM import ELM
import matplotlib.pylab as plt 
from numpy import zeros, where, vstack, array, ones, argsort, absolute, mean, reshape
from kmean import kmean
from zeropadding import zeropadding
from lineIMAGE import lineIMAGE
from lineMNIST import lineMNIST
from distCal import distCal
from massflt import massflt
from PIL import Image
import os

class ELMrecogPic_beta(object):
    
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
   
    def runtest(self, numDigits):
        
        fltSize = self.patcharm
        halfSize = fltSize/2
        datasource = 'others' # 'synthesis' or others
        
        if datasource == 'MNIST':            
            self.testcanvas0, labeltrue, canvasSize0 = lineMNIST(numDigits) 
        elif datasource == 'others':
            labeltrue = zeros((numDigits))
            path = 'C:\\comingdata\\others\\'
            dirs = os.listdir(path)
            for item in dirs: 
                img = Image.open(path+item)
                img = img.crop(( int(img.size[0]/5), 1, int(img.size[0]/3), int(img.size[1]/2)))
                SIZE = ( img.size[0], img.size[1] )
                #SIZE = ( int(img.size[0]*1.6), int(img.size[1]*1.6) )
                #img = img.resize(SIZE, Image.BILINEAR)
                temp_canvas0 = array(img.getdata())/255
                print temp_canvas0.shape
                self.testcanvas0 = reshape( temp_canvas0, (SIZE[1], SIZE[0]))
            canvasSize0 = array([SIZE[1], SIZE[0]])
        else:
            self.testcanvas0, labeltrue, canvasSize0 = lineIMAGE(numDigits, source=datasource)

        ## zeropadding
        testcanvas = zeropadding(self.testcanvas0, halfSize, halfSize)    
        canvasSize = array(canvasSize0) + fltSize
        
        ## convolving part
        tot_count = (canvasSize[0]-fltSize)*(canvasSize[1]-fltSize)
        self.labelmatrix0 = zeros((canvasSize[0]-fltSize, canvasSize[1]-fltSize))
        self.labelmatrix = zeros((canvasSize[0]-fltSize, canvasSize[1]-fltSize))
        
        print "> Convolving in progress ..."
        for x in range(halfSize, canvasSize[0]-halfSize):
            for y in range(halfSize, canvasSize[1]-halfSize):
                
                if ( (x-halfSize)*(canvasSize[1]-fltSize)+y-halfSize) %500 == 0:
                    print "> Current convolving index is %d ( %d in total )..."%((x-halfSize)*(canvasSize[1]-fltSize)+y-halfSize+1, tot_count)
                item_temp = testcanvas[x-halfSize:x+halfSize, y-halfSize:y+halfSize]
                labelhatpatch = self.newELM.recall(item_temp)                     
                self.labelmatrix0[x-halfSize,y-halfSize] = distCal(labelhatpatch)               
                
        #self.labelmatrix = massflt(self.labelmatrix0)
        self.labelmatrix = self.labelmatrix0
        
        ## clustering            
        clust_temp = where(self.labelmatrix != -1)    
        datamatrix = vstack((clust_temp[0], clust_temp[1]))
        Con, Iter = True, 0
        separateWidth = self.patcharm*3/4        
        
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
            plt.imshow(self.labelmatrix)  
            fig.add_subplot(4,1,4)
            plt.imshow(self.hatcolormatrix)
            
        elif self.Lane == 'v':
            fig = plt.figure()
            fig.add_subplot(1,4,1)
            plt.imshow(self.testcanvas0)
            fig.add_subplot(1,4,2)
            plt.imshow(self.labelmatrix0)  
            fig.add_subplot(1,4,3)
            plt.imshow(self.labelmatrix)  
            fig.add_subplot(1,4,4)
            plt.imshow(self.hatcolormatrix)                              
        