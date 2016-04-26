"""
Created on Tue Mar 22 14:50:01 2016

@author: houyimeng
"""


from ELM import ELM
from MNISTCanvasImage import MNISTCanvasImage
import matplotlib.pylab as plt 
from numpy import zeros, where, vstack, loadtxt, array, mean, ones, argsort, absolute
from kmean import kmean
from distCal import distCal
from zeropadding import zeropadding
from lineMNIST import lineMNIST
import math
import random

class ELMrecogPic_multiscale(object):
    
    def __init__(self, Num, Method, IMGsize = (100, 100)):
        
        self.patcharm = (28, 28)
        self.newELM = ELM(InSize = self.patcharm[0]**2, HidSize=(self.patcharm[0]**2)*20, OutSize=10, \
                           OPIUM_type='basic', genWeights_type ='dec')
        self.newELM.load('WellTrainedELMWeights')
        #self.newELM.load('rfciwelm')
        
        self.eqfactor = loadtxt('MNISTstatistics_factor.dat')
        
        self.truecolormatrix =  None
        self.hatcolormatrix =  None
        self.testcanvas0 = None
        self.labelmatrix = None
        self.label_max = None
        self.label_mean = None
        self.numDigits = Num
        self.labeltrue = zeros(Num)
        self.labelhat = zeros(Num)
        self.method = Method
        self.canvasSize0 = IMGsize
        self.entrylist = []
        
        self.ave_act = [0.09, 0.1]
        self.peak_act = [0.3, 0.25, 0.2]       
        self.entries = len(self.ave_act)* len(self.peak_act)
        
    def convolve(self):
        
        stepSize = 1
        edgeSize = 0
        paddingMode = 0
        
        halfSize = [int(self.patcharm[0]/2), int(self.patcharm[1]/2)]
        
        if self.method == 'rand':
            
            imgcanvas = MNISTCanvasImage('MNIST', self.canvasSize0)    
            self.testcanvas0, self.labeltrue = imgcanvas.generateImage(self.numDigits)            
            
        elif self.method == 'line':   
            
            testcanvas_temp, self.labeltrue, canvasSize_info = lineMNIST(self.numDigits, edgeSize)
            
            if paddingMode == 1:
                
                self.testcanvas0 = zeros(self.canvasSize0)
                idx1 = random.randint(0, self.canvasSize0[0]-canvasSize_info[0]-1)
                idx2 = random.randint(0, self.canvasSize0[1]-canvasSize_info[1]-1)
                self.testcanvas0[idx1:idx1+canvasSize_info[0], idx2:idx2+canvasSize_info[1]] = testcanvas_temp
            
            else:
                self.testcanvas0 = testcanvas_temp
                self.canvasSize0 = canvasSize_info        
        
        ## zeropadding
        testcanvas = zeropadding(self.testcanvas0, halfSize[0], halfSize[1])    
        canvasSize = array(self.canvasSize0) + array(self.patcharm)        
        
        ## convolving part
        tot_count = (canvasSize[0]-halfSize[0]*2)*(canvasSize[1]-halfSize[1]*2)
        
        self.labelmatrix = zeros(((canvasSize[0]-halfSize[0]*2)/stepSize, (canvasSize[1]-halfSize[1]*2)/stepSize))
        self.label_max = zeros(((canvasSize[0]-halfSize[0]*2)/stepSize, (canvasSize[1]-halfSize[1]*2)/stepSize))
        self.label_mean = zeros(((canvasSize[0]-halfSize[0]*2)/stepSize, (canvasSize[1]-halfSize[1]*2)/stepSize))
        
        print "> Convolving in progress ..."
        for x in range(halfSize[0], canvasSize[0]-halfSize[0]):
            for y in range(halfSize[1], canvasSize[1]-halfSize[1]):
                
                
                if ( (x-halfSize[0])*(canvasSize[1]-halfSize[1]*2)+y-halfSize[1]+1 ) %500 == 0:
                    print "> Current convolving index is %d ( %d in total )..."%\
                   ( (x-halfSize[0])*(canvasSize[1]-halfSize[1]*2)+y-halfSize[1]+1, tot_count)
                
                if x%stepSize == 0 and y%stepSize == 0:
                    
                    item_temp = testcanvas[x-halfSize[0]:x+halfSize[0], y-halfSize[1]:y+halfSize[1]]
                    labelhatpatch = self.newELM.recall(item_temp)                                      
                    self.labelmatrix[(x-halfSize[0])/stepSize, (y-halfSize[1])/stepSize] = distCal(labelhatpatch)
                    self.label_max[(x-halfSize[0])/stepSize, (y-halfSize[1])/stepSize] = max(labelhatpatch)
                    self.label_mean[(x-halfSize[0])/stepSize, (y-halfSize[1])/stepSize] = mean(labelhatpatch)

    def testOne(self, scale_idx):

        _labelmatrix_temp = zeros(self.labelmatrix.shape)
        _labelmatrix_temp = self.labelmatrix
        
        labelhat0 = zeros(self.numDigits)
        countnum = zeros((10, self.numDigits))
        
        print "Current entry :", scale_idx+1
        for idx in range(_labelmatrix_temp.shape[0]):
            for idy in range(_labelmatrix_temp.shape[1]):
                if self.peak_act[int(scale_idx/len(self.ave_act))] * self.label_max[idx, idy] \
                    >= self.label_mean[idx, idy] >= self.ave_act[scale_idx%len(self.ave_act)]:
                    _labelmatrix_temp[idx,idy] = _labelmatrix_temp[idx,idy]
                else:
                    _labelmatrix_temp[idx,idy] = -1      
                
        ## clustering            
        clust_temp = where(_labelmatrix_temp != -1)    
        datamatrix = vstack((clust_temp[0], clust_temp[1]))
        
        Con, Iter = True, 0
        
        while Con and Iter < (self.numDigits * 10):
            print "> Clustering entry No.",Iter+1, "..."
            
            clusters, centroids = kmean(datamatrix.T, self.numDigits)
            Con = False
            
            for i in range(len(centroids)):
                for j in range(len(centroids)):
                    
                    # bound detection 
                    if i != j and absolute(centroids[i][0] - centroids[j][0]) <= int(self.patcharm[0]/2) and  \
                                  absolute(centroids[i][1] - centroids[j][1]) <= int(self.patcharm[1]/2) :
                        Con = True

            Iter += 1
        if Iter < self.numDigits * 10: 
            self.entrylist.append(scale_idx)
            print "Clustering Succeed!"
        else:
            print "Clustering failed, latest entry picked"
            
        centroids = array(centroids)
        
        # decision making   
        if clusters != []:
            for i in range(len(clusters)):
                for j in range(len(clusters[i])):
                    temp = clusters[i][j]
                    distfactor = (temp[0]-centroids[i,0])**2+(temp[1]-centroids[i,1])**2 + 0.01 # euclidean distance
                    countnum [_labelmatrix_temp[temp[0],temp[1]],i] += 1/distfactor/self.eqfactor[self.labelmatrix[temp[0],temp[1]]] # equalization

                labelhat0[i] = countnum[:,i].argmax()
        else:
            print "No labels are tapped"
            labelhat0 = -1*ones(10)
            
        return labelhat0              

    def testLoop(self): 
        
        result_multi = zeros((self.entries, self.numDigits))
        for ety in range(self.entries):
            result_multi[ety,:] = self.testOne(ety)
        
        # voting
        for i1 in range(self.numDigits):           
            combining_temp = zeros(10)
            
            for j1 in self.entrylist:                 
                combining_temp[result_multi[j1,i1]] += 1
                print combining_temp
            self.labelhat[i1] = combining_temp.argmax()           

        print 'True label:', self.labeltrue
        print 'Hat label:', array(self.labelhat,dtype='int')
                
    def visualize(self):
        
        plt.figure(1)        
        plt.imshow(self.testcanvas0)
        plt.figure(2)
        plt.imshow(self.labelmatrix)
        plt.figure(3)
        plt.imshow(self.truecolormatrix)
        plt.figure(4)
        plt.imshow(self.hatcolormatrix)
