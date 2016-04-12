# -*- coding: utf-8 -*-
"""
Created on Thu Mar 31 17:15:09 2016

@author: houyimeng
"""

from ELM import ELM
import matplotlib.pylab as plt 
from numpy import zeros, where, vstack, array, reshape, ones, argsort, absolute, random, mean
from kmean import kmean
from ImageToolbox.zeropadding import zeropadding
from distCal import distCal
from PIL import Image
import os

class ELMrecogPic_ningbo_(object):
    
    def __init__(self):
        
        self.patcharm = 28
        self.Outsize = 11 
        self.newELM = ELM(self.patcharm**2, (self.patcharm**2)*10, self.Outsize)
        self.newELM.load('SYNdata')          
        
        self.truecolormatrix =  None
        self.hatcolormatrix =  None
        self.testcanvas0 = None
        self.labelmatrix = None
    def train(self):
        
        basetrpath = '/home/houyimeng/Binary/'
        imglist = []  
        labelist = []
        categry = len(os.listdir(basetrpath))
        n = 0
        for i in os.listdir(basetrpath):
            path = '/home/houyimeng/Binary/'+str(i)+'/'
            dirs = os.listdir(path)
            for item in dirs: 
                img = Image.open(path+item) 
                imglist.append( array(img.getdata()))
                labelist.append( i )
                n += 1
                
        print labelist
        print imglist
        trdata = zeros((self.patcharm**2, n))
        trlabel = zeros((categry, n))
        randidx = random.choice(n, n, replace=False)
        for i in randidx:
            trdata[:,i] = imglist[i].flatten()
            temp_la = zeros( categry )
            temp_la[labelist[i]] = 1
            trlabel[:,i] = temp_la
            
    def runtest(self, numDigits):
        
        fltSize, halfSize, displaysize = 28, 14, 3
      
        path = '/home/houyimeng/testcanvas/'
        tedirs = os.listdir(path)
        extcanvas = []
        
        for item in tedirs:
            temp_img = Image.open( path+item )
            temp_img = temp_img.crop(( int(temp_img.size[0]/2), 1, temp_img.size[0]-1, temp_img.size[1]-1))
            SIZE = ( int(temp_img.size[0]*1.6), int(temp_img.size[1]*1.6) )
            temp_img = temp_img.resize( SIZE, Image.BILINEAR)
            extcanvas.append( array( (temp_img.getdata()-mean(temp_img.getdata()))>0 , dtype='int')) 
        
        self.testcanvas0 = reshape( extcanvas[0], (SIZE[1], SIZE[0]))

        canvasSize0 = [self.testcanvas0.shape[0], self.testcanvas0.shape[1]]
        labeltrue = zeros(numDigits)
            
        ## zeropadding
        testcanvas = zeropadding(self.testcanvas0, halfSize, halfSize)    
        canvasSize = array(canvasSize0) + fltSize        
                   
        ## convolving part
        tot_count = (canvasSize[0]-fltSize)*(canvasSize[1]-fltSize)
        self.labelmatrix = zeros((canvasSize[0]-fltSize, canvasSize[1]-fltSize))
        
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
        
        while Con and Iter < (numDigits * 10):
            print "> Clustering entry No.",Iter+1, "..."
            
            clusters, centroids = kmean(datamatrix.T, numDigits)
            Con = False
            
            for i in range(len(centroids)):
                for j in range(len(centroids)):
                    
                    # bound detection 
                    if i != j and absolute(centroids[i][0] - centroids[j][0]) <= 21 and  \
                                  absolute(centroids[i][1] - centroids[j][1]) <= 21 :
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
                
        centroidindex = argsort(centroids[:,1])          
        labelhat = array(labelhat0[centroidindex], dtype='int')
        
        self.truecolormatrix =  (-1)*ones((canvasSize[0]-fltSize, canvasSize[1]-fltSize))
        self.hatcolormatrix  =  (-1)*ones((canvasSize[0]-fltSize, canvasSize[1]-fltSize))
        
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
        
        plt.figure(1)        
        plt.imshow(self.testcanvas0)
        plt.figure(2)
        plt.imshow(self.labelmatrix)
        plt.figure(3)
        plt.imshow(self.truecolormatrix)
        plt.figure(4)
        plt.imshow(self.hatcolormatrix)