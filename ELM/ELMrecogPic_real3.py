# -*- coding: utf-8 -*-
"""
Created on Mon Apr 18 10:14:06 2016

@author: ThinkPad
"""


from ELM import ELM
import matplotlib.pylab as plt 
from numpy import array, uint8, ones, where, vstack, zeros, sqrt, absolute, hstack
from zeropadding import zeropadding
from distCal import distCal
from codetable import codetable
import cv2
import os

def find(lst, tar):
    return [i for i, x in enumerate(lst) if x == tar]

class ELMrecogPic_real3(object):
    
    def __init__(self):
        
        self.patcharm = [576, 900, 1296] #1296 #900
        self.Outsize = 35
        self.dim = [20, 24, 28]
        self.newELM_20 = ELM(self.patcharm[0], self.patcharm[0]*10, self.Outsize)
        self.newELM_24 = ELM(self.patcharm[1], self.patcharm[1]*10, self.Outsize)        
        self.newELM_28 = ELM(self.patcharm[2], self.patcharm[2]*10, self.Outsize)
        self.newELM_20.load('C:\\dataspace\\weights\\harbour35_20')    
        self.newELM_28.load('C:\\dataspace\\weights\\harbour35_28')                
        self.hog_20 = cv2.HOGDescriptor((self.dim[0], self.dim[0]), (8,8), (4,4), (4,4), 9)
        self.hog_24 = cv2.HOGDescriptor((self.dim[1], self.dim[1]), (8,8), (4,4), (4,4), 9)
        self.hog_28 = cv2.HOGDescriptor((self.dim[2], self.dim[2]), (8,8), (4,4), (4,4), 9)
   
    def runtest(self):
        
        # load tested image        
        #crop_rg = [0.4, 0.5, 0.1, 0.75] 
        crop_rg = [0.4, 0.5, 0.1, 0.75]
        scaleFactor = 0.6               
        path = 'C:\\dataspace\\IMGdata\\others\\'
        dirs = os.listdir(path)
        for item in dirs: 
            img = cv2.imread(path+item, 0)
            img = img[int(img.shape[0]*crop_rg[2]):int(img.shape[0]*crop_rg[3]), int(img.shape[1]*crop_rg[0]):int(img.shape[1]*crop_rg[1])]
            #self.testcanvas = img
            self.testcanvas = cv2.resize(img, (0,0), fx=scaleFactor, fy=scaleFactor)
            print "The size of the testcanvas:", self.testcanvas.shape
            cv2.imwrite('greyscale.png', img)
        
###############################################################################
        
        fltSize_20 = self.dim[0]
        halfSize_20 = self.dim[0]/2
        
        ## zeropadding
        testcanvas = zeropadding(self.testcanvas, halfSize_20, halfSize_20)    
        canvasSize = array(self.testcanvas.shape) + fltSize_20   
        ## convolving part 1
        tot_count = (canvasSize[0]-fltSize_20)*(canvasSize[1]-fltSize_20)
        self.labelmatrix_20 = -1*ones(( canvasSize[0]-fltSize_20, canvasSize[1]-fltSize_20))
        
        print "> Convolving part 1 in progress ..., WindowSize = (20,20)"
        for y in range(halfSize_20, canvasSize[0]-halfSize_20):
            stepsize = 3
            for x in range(halfSize_20, canvasSize[1]-halfSize_20):
                #print x,y
                if ((y-halfSize_20)*(canvasSize[1]-fltSize_20)+x-halfSize_20+1)%2000 == 0:
                    print "> Current convolving index is %d ( %d in total )..."\
                    %((y-halfSize_20)*(canvasSize[1]-fltSize_20)+x-halfSize_20+1, tot_count)
                    
                if x%stepsize == 0:  
                    item_temp = testcanvas[y-halfSize_20:y+halfSize_20, x-halfSize_20:x+halfSize_20]
                    temp_img = uint8(item_temp)
                    hist = self.hog_20.compute(temp_img)
                    labelhatpatch = self.newELM_20.recall(hist)
                    label_val = distCal(labelhatpatch)
                    self.labelmatrix_20[y-halfSize_20, x-halfSize_20] = label_val
                    if label_val != -1:
                        stepsize = 3
                        for i in range(-stepsize+1, 0):
                            for j in range(-stepsize+1, 0):
                                item_sec = testcanvas[y-halfSize_20+i:y+halfSize_20+i, x-halfSize_20+j:x+halfSize_20+j]
                                temp_sec = uint8(item_sec)
                                hist_sec = self.hog_20.compute(temp_sec)
                                labelhatpatch_sec = self.newELM_20.recall(hist_sec)
                                self.labelmatrix_20[y-halfSize_20+i, x-halfSize_20+j] = distCal(labelhatpatch_sec)
                    else:
                        if stepsize < 6:
                            stepsize += 2
                        else:
                            stepsize = 6
###############################################################################
        ## convolving part 2
        fltSize_24 = self.dim[1]
        halfSize_24 = self.dim[1]/2
        
        ## zeropadding
        testcanvas = zeropadding(self.testcanvas, halfSize_24, halfSize_24)    
        canvasSize = array(self.testcanvas.shape) + fltSize_24   
        ## convolving part 1
        tot_count = (canvasSize[0]-fltSize_24)*(canvasSize[1]-fltSize_24)
        self.labelmatrix_24 = -1*ones(( canvasSize[0]-fltSize_24, canvasSize[1]-fltSize_24))
        
        print "> Convolving part 2 in progress ..., WindowSize = (24,24)"
        for y in range(halfSize_24, canvasSize[0]-halfSize_24):
            stepsize = 3
            for x in range(halfSize_24, canvasSize[1]-halfSize_24):
                #print x,y
                if ((y-halfSize_24)*(canvasSize[1]-fltSize_24)+x-halfSize_24+1)%2000 == 0:
                    print "> Current convolving index is %d ( %d in total )..."\
                    %((y-halfSize_24)*(canvasSize[1]-fltSize_24)+x-halfSize_24+1, tot_count)
                    
                if x%stepsize == 0:  
                    item_temp = testcanvas[y-halfSize_24:y+halfSize_24, x-halfSize_24:x+halfSize_24]
                    temp_img = uint8(item_temp)
                    hist = self.hog_24.compute(temp_img)
                    labelhatpatch = self.newELM_24.recall(hist)
                    label_val = distCal(labelhatpatch)
                    self.labelmatrix_24[y-halfSize_24, x-halfSize_24] = label_val
                    if label_val != -1:
                        stepsize = 3
                        for i in range(-stepsize+1, 0):
                            for j in range(-stepsize+1, 0):
                                item_sec = testcanvas[y-halfSize_24+i:y+halfSize_24+i, x-halfSize_24+j:x+halfSize_24+j]
                                temp_sec = uint8(item_sec)
                                hist_sec = self.hog_24.compute(temp_sec)
                                labelhatpatch_sec = self.newELM_24.recall(hist_sec)
                                self.labelmatrix_24[y-halfSize_24+i, x-halfSize_24+j] = distCal(labelhatpatch_sec)
                    else:
                        if stepsize < 6:
                            stepsize += 2
                        else:
                            stepsize = 6
                            
###############################################################################
        ## convolving part 3
        fltSize_28 = self.dim[2]
        halfSize_28 = self.dim[2]/2
        
        ## zeropadding
        testcanvas = zeropadding(self.testcanvas, halfSize_28, halfSize_28)    
        canvasSize = array(self.testcanvas.shape) + fltSize_28   
        ## convolving part 1
        tot_count = (canvasSize[0]-fltSize_28)*(canvasSize[1]-fltSize_28)
        self.labelmatrix_28 = -1*ones(( canvasSize[0]-fltSize_28, canvasSize[1]-fltSize_28))
        
        print "> Convolving part 3 in progress ..., WindowSize = (28,28)"
        for y in range(halfSize_28, canvasSize[0]-halfSize_28):
            stepsize = 3
            for x in range(halfSize_28, canvasSize[1]-halfSize_28):
                #print x,y
                if ((y-halfSize_28)*(canvasSize[1]-fltSize_28)+x-halfSize_28+1)%2000 == 0:
                    print "> Current convolving index is %d ( %d in total )..."\
                    %((y-halfSize_28)*(canvasSize[1]-fltSize_28)+x-halfSize_28+1, tot_count)
                    
                if x%stepsize == 0:  
                    item_temp = testcanvas[y-halfSize_28:y+halfSize_28, x-halfSize_28:x+halfSize_28]
                    temp_img = uint8(item_temp)
                    hist = self.hog_28.compute(temp_img)
                    labelhatpatch = self.newELM_28.recall(hist)
                    label_val = distCal(labelhatpatch)
                    self.labelmatrix_28[y-halfSize_28, x-halfSize_28] = label_val
                    if label_val != -1:
                        stepsize = 3
                        for i in range(-stepsize+1, 0):
                            for j in range(-stepsize+1, 0):
                                item_sec = testcanvas[y-halfSize_28+i:y+halfSize_28+i, x-halfSize_28+j:x+halfSize_28+j]
                                temp_sec = uint8(item_sec)
                                hist_sec = self.hog_28.compute(temp_sec)
                                labelhatpatch_sec = self.newELM_28.recall(hist_sec)
                                self.labelmatrix_28[y-halfSize_28+i, x-halfSize_28+j] = distCal(labelhatpatch_sec)
                    else:
                        if stepsize < 6:
                            stepsize += 2
                        else:
                            stepsize = 6
        
################################################################################
        # get several possible centers
                            
        clust_temp_20 = where(self.labelmatrix_20 != -1)
        clust_temp_24 = where(self.labelmatrix_24 != -1)
        clust_temp_28 = where(self.labelmatrix_28 != -1) 
        clust_temp_x = hstack((clust_temp_20[0], clust_temp_24[0], clust_temp_28[0]))
        clust_temp_y = hstack((clust_temp_20[1], clust_temp_24[1], clust_temp_28[1]))
        data_coor = vstack((clust_temp_x, clust_temp_y))
            
        center = ones(data_coor.shape[1])
        
        numDigits = 2  
        for i in range(1, data_coor.shape[1]):
            coor = [data_coor[0,i], data_coor[1,i]]
            dist = zeros(i)
        
            for j in range(0, i):
                dist[j] = sqrt((coor[0] - data_coor[0, j])**2 + (coor[1] - data_coor[1, j])**2)
            if min(dist) <= halfSize_20:
                center[i] =  center[dist.argmin()]
            else:
                center[i] = numDigits
                numDigits += 1       
        
##############################################################################
                 
        labelhat = zeros(numDigits)
        centroids = zeros((numDigits, 2))
        
        for i in range(1, numDigits):
            
            centroid_x, centroid_y = 0,0
            clusters_idx = find(center, i)
            for j in clusters_idx:
                centroid_x += data_coor[0,j]
                centroid_y += data_coor[1,j]
            centroids[i-1,:] = array([centroid_x/len(clusters_idx), centroid_y/len(clusters_idx)])
            countnum = zeros(self.Outsize-1)
            # decicsion making   
            for k in clusters_idx:
                distfactor = absolute(data_coor[0, k]-centroids[i-1, 0]) + absolute(data_coor[1, k]-centroids[i-1, 1]) + 0.01 # euclidean distance
                
                if k < len(clust_temp_20[0]):
                    countnum[self.labelmatrix_20[data_coor[0, k], data_coor[1, k]]] += 1/distfactor # equalization #1
                elif len(clust_temp_20[0]) <= k < len(clust_temp_20[0]) + len(clust_temp_24[0]): 
                    countnum[self.labelmatrix_24[data_coor[0, k], data_coor[1, k]]] += 1/distfactor # equalization #1                    
                else:
                    countnum[self.labelmatrix_28[data_coor[0, k], data_coor[1, k]]] += 1/distfactor # equalization #1                    
            labelhat[i] = countnum.argmax()
        
            print "Number/letter :", i ,", Coordinates =", "(", int(centroids[i-1, 0]), ',', int(centroids[i-1, 1]), "), Prediction ->", codetable(int(labelhat[i]))         
        
        return self.labelmatrix_20, self.labelmatrix_24, self.labelmatrix_28, self.testcanvas
                
#############################################################################
        
    def visualize(self):
        plt.figure(1)
        plt.imshow(self.testcanvas)
        plt.figure(2)
        plt.imshow(self.labelmatrix_20)
        plt.figure(3)
        plt.imshow(self.labelmatrix_24)
        plt.figure(4)
        plt.imshow(self.labelmatrix_28)
