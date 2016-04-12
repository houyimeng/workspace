"""
Created on Mon Mar  7 09:49:31 2016

@author: houyimeng
"""

from ELM import ELM
from MNISTCanvasImage import MNISTCanvasImage
import matplotlib.pylab as plt 
from numpy import zeros, where, vstack, array, mean, ones, argsort, absolute
from kmean import kmean
from zeropadding import zeropadding
from lineMNIST import lineMNIST
from distCal import distCal
import random
import math

def softmax(input_raw):

    output0 = zeros(len(input_raw))
    for idx in range(len(input_raw)):
        output0[idx] = 1/(1+math.exp(-input_raw[idx])) 
    output0 = output0/sum(output0)
    print output0
    output1 = output0 >= 0.1

    if output1.tolist().count(True) >= 1:
        return output1.argmax()
    elif (output1 >= 0.1).tolist().count(True) == 0:
        return -1

def fltdot(input_raw):
    
    average_activation = 0.1
    peak_activation = 0.2    
    
    if  peak_activation * max(input_raw) >= mean(input_raw) >= average_activation:
        output_raw = distCal(input_raw)
    else:
        output_raw = -1
    
    return output_raw

class ELMrecogPic(object):
    
    def __init__(self, lane = 'v'):
        
        self.patcharm = 28
        self.outSize = 10
        self.newELM = ELM(self.patcharm**2, (self.patcharm**2)*20, self.outSize)
        self.newELM.load('WellTrainedELMWeights')          
        
        #self.eqfactor = loadtxt('MNISTstatistics_factor.dat')
        
        self.truecolormatrix =  None
        self.hatcolormatrix =  None
        self.testcanvas0 = None
        self.labelmatrix = None
        self.raw_labelmatrix = None
        self.Lane = lane
   
    def runtest(self, numDigits, method = 'line', canvasSize0 = [], paddingMode = 0, edgeSize = 0):
        
        fltSize = 28
        halfSize = 14
        displaysize = 5
        DigitLane = 10
        norm_flag = 0
        
        if method == 'rand':
            
            imgcanvas = MNISTCanvasImage('MNIST', canvasSize0)    
            self.testcanvas0, labeltrue = imgcanvas.generateImage(numDigits)            
            
        elif method == 'line':   
            
            testcanvas_temp, labeltrue, canvasSize_info = lineMNIST(numDigits, edgeSize, DigitLane, self.Lane, norm_flag)
            
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
        self.labelmatrix = zeros((canvasSize[0]-fltSize, canvasSize[1]-fltSize))
        self.raw_labelmatrix = zeros((canvasSize[0]-fltSize, canvasSize[1]-fltSize))
        
        print "> Convolving in progress ..."
        for x in range(halfSize, canvasSize[0]-halfSize):
            for y in range(halfSize, canvasSize[1]-halfSize):
                
                if ( (x-halfSize)*(canvasSize[1]-fltSize)+y-halfSize) %500 == 0:
                    print "> Current convolving index is %d ( %d in total )..."%((x-halfSize)*(canvasSize[1]-fltSize)+y-halfSize+1, tot_count)
                item_temp = testcanvas[x-halfSize:x+halfSize, y-halfSize:y+halfSize]
                labelhatpatch = self.newELM.recall(item_temp)
                self.raw_labelmatrix[x-halfSize,y-halfSize] = labelhatpatch.argmax() 
                self.labelmatrix[x-halfSize,y-halfSize] = fltdot(labelhatpatch)
                
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
                    if i != j and absolute(centroids[i][0] - centroids[j][0]) <= halfSize and  \
                                  absolute(centroids[i][1] - centroids[j][1]) <= halfSize :
                        Con = True

            Iter += 1
        if Iter < numDigits * 10:        
            print "Clustering Succeed!"
        else:
            print "Clustering failed, latest entry picked"
        
        centroids = array(centroids)
        labelhat0 = zeros(numDigits)
        countnum = zeros((10,numDigits))

        # decicsion making   
        if clusters != []:
            for i in range(len(clusters)):
                for j in range(len(clusters[i])):
                    temp = clusters[i][j]
                    distfactor = absolute(temp[0]-centroids[i,0]) + absolute(temp[1]-centroids[i,1]) + 0.01 # euclidean distance
                    countnum [self.labelmatrix[temp[0],temp[1]],i] += 1/distfactor#/self.eqfactor[self.labelmatrix[temp[0],temp[1]]] # equalization

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
        
        # visualize the centroids and their labels
        for i in range(len(centroids)):
            self.truecolormatrix [round(centroids[i][0])-displaysize:round(centroids[i][0])+displaysize, \
                     round(centroids[i][1])-displaysize:round(centroids[i][1])+displaysize ] = labeltrue[i]
                     
        for i in range(len(centroids)):
            self.hatcolormatrix [round(centroids[i][0])-displaysize:round(centroids[i][0])+displaysize, \
                     round(centroids[i][1])-displaysize:round(centroids[i][1])+displaysize ] = labelhat[i]        
        
        print "True labels are:", labeltrue   
        print "Hat  labels are:", labelhat 
        
        '''
        if (labeltrue == labelhat).tolist().count(False) == len(labeltrue):
            return 0
        else:
            return (labeltrue == labelhat).tolist().count(False)
        '''
        
    def visualize(self):
        
        if self.Lane == 'h':
            fig = plt.figure()
            fig.add_subplot(5,1,1)
            plt.imshow(self.testcanvas0)
            fig.add_subplot(5,1,2)
            plt.imshow(self.raw_labelmatrix)
            fig.add_subplot(5,1,3)
            plt.imshow(self.labelmatrix)  
            fig.add_subplot(5,1,4)
            plt.imshow(self.truecolormatrix)  
            fig.add_subplot(5,1,5)
            plt.imshow(self.hatcolormatrix)
            
        elif self.Lane == 'v':
            fig = plt.figure()
            fig.add_subplot(1,5,1)
            plt.imshow(self.testcanvas0)
            fig.add_subplot(1,5,2)
            plt.imshow(self.raw_labelmatrix)
            fig.add_subplot(1,5,3)
            plt.imshow(self.labelmatrix)  
            fig.add_subplot(1,5,4)
            plt.imshow(self.truecolormatrix)  
            fig.add_subplot(1,5,5)
            plt.imshow(self.hatcolormatrix)  
        
