
from ELM import ELM
import matplotlib.pylab as plt 
from numpy import zeros, array, uint8, ones
from zeropadding import zeropadding
from distCal import distCal
import cv2
import os
import math

class ELMrecogPic_real(object):
    
    def __init__(self):

        self.patcharm = 900 #576 #1296
        self.Outsize = 35
        self.newELM = ELM(self.patcharm, self.patcharm*10, self.Outsize)
        self.newELM.load('C:\\dataspace\\weights\\harbour35_1818')          
        #self.hog = cv2.HOGDescriptor((20,20), (8,8), (4,4), (4,4), 9)
        self.hog = cv2.HOGDescriptor((18,18), (6,6), (3,3), (3,3), 9)
   
    def runtest(self, lane = 'h'):
        
        # load tested image        
        crop_rg = [0.4, 0.6, 0.4, 0.7]                
        path = 'C:\\dataspace\\IMGdata\\others\\'
        dirs = os.listdir(path)
        for item in dirs: 
            img = cv2.imread(path+item, 0)
            img = img[int(img.shape[0]*crop_rg[2]):int(img.shape[0]*crop_rg[3]), int(img.shape[1]*crop_rg[0]):int(img.shape[1]*crop_rg[1])]
            #self.testcanvas = img
            self.testcanvas = cv2.resize(img, (0,0), fx=0.8, fy=0.8)
            print "The size of the testcanvas:", self.testcanvas.shape
            #self.testcanvas = img
            cv2.imwrite('greyscale.png', img)

        fltSize = 20
        halfSize = fltSize/2
        stepsize = 2
        
        ## zeropadding
        testcanvas = zeropadding(self.testcanvas, halfSize, halfSize)    
        canvasSize = array(self.testcanvas.shape) + fltSize        

        ## convolving part
        tot_count = (canvasSize[0]-fltSize)*(canvasSize[1]-fltSize)
        self.labelmatrix = -1*ones(( canvasSize[0]-fltSize, canvasSize[1]-fltSize ))
        
        if lane == 'v':
            print "> Convolving in progress ..."
            for y in range(halfSize, canvasSize[0]-halfSize):
                stepsize = 2
                for x in range(halfSize, canvasSize[1]-halfSize):
                    
                    if ((y-halfSize)*(canvasSize[1]-fltSize)+x-halfSize+1)%2000 == 0:
                        print "> Current convolving index is %d ( %d in total )..."\
                        %((y-halfSize)*(canvasSize[1]-fltSize)+x-halfSize+1, tot_count)
                        
                    if x%stepsize == 0:  
                        item_temp = testcanvas[y-halfSize:y+halfSize, x-halfSize:x+halfSize]
                        temp_img = temp_img = uint8(item_temp)
                        hist = self.hog.compute(temp_img)
                        labelhatpatch = self.newELM.recall(hist)
                        label_val = distCal(labelhatpatch)
                        self.labelmatrix[y-halfSize, x-halfSize] = label_val
                        if label_val != -1:
                            stepsize = 2
                            for i in range(-stepsize+1, stepsize):
                                for j in range(-stepsize+1, stepsize):
                                    item_sec = testcanvas[y-halfSize+i:y+halfSize+i, x-halfSize+j:x+halfSize+j]
                                    temp_sec = uint8(item_sec)
                                    hist_sec = self.hog.compute(temp_sec)
                                    labelhatpatch_sec = self.newELM.recall(hist_sec)
                                    self.labelmatrix[y-halfSize+i, x-halfSize+j] = distCal(labelhatpatch_sec)
                        else:
                            if stepsize <6:
                                stepsize += 2
                            else:
                                stepsize = 6
        elif lane == 'h':

            print "> Convolving in progress ..."
            for y in range(halfSize, canvasSize[1]-halfSize):
                stepsize = 2
                for x in range(halfSize, canvasSize[0]-halfSize):
                    
                    if ((y-halfSize)*(canvasSize[0]-fltSize)+x-halfSize+1)%2000 == 0:
                        print "> Current convolving index is %d ( %d in total )..."\
                        %((y-halfSize)*(canvasSize[0]-fltSize)+x-halfSize+1, tot_count)
                        
                    if x%stepsize == 0:  
                        item_temp = testcanvas[x-halfSize:x+halfSize, y-halfSize:y+halfSize]
                        temp_img = temp_img = uint8(item_temp)
                        hist = self.hog.compute(temp_img)
                        labelhatpatch = self.newELM.recall(hist)
                        label_val = distCal(labelhatpatch)
                        self.labelmatrix[x-halfSize, y-halfSize] = label_val
                        if label_val != -1:
                            stepsize = 2
                            for i in range(-stepsize+1, stepsize):
                                    item_sec = testcanvas[x-halfSize+i:x+halfSize+i, y-halfSize:y+halfSize]
                                    temp_sec = uint8(item_sec)
                                    hist_sec = self.hog.compute(temp_sec)
                                    labelhatpatch_sec = self.newELM.recall(hist_sec)
                                    self.labelmatrix[x-halfSize+i, y-halfSize] = distCal(labelhatpatch_sec)
                        else:
                            if stepsize <10:
                                stepsize += 2
                            else:
                                stepsize = 10
                
        return self.labelmatrix, self.testcanvas
        
#############################################################################
        
    def visualize(self):
        plt.figure(1)
        plt.imshow(self.testcanvas)
        plt.figure(2)
        plt.imshow(self.labelmatrix)

        '''
        canvasSize = [self.testcanvas.shape[0], self.testcanvas.shape[1]]
        self.labelmatrix = zeros((math.ceil(canvasSize[0]/float(self.patcharm)), math.ceil(canvasSize[1]/float(self.patcharm)) ))
        self.frag_testcanvas = fragImg(self.testcanvas, (self.patcharm, self.patcharm))
        
        for i in range(len(self.frag_testcanvas)):
            labelpatch = self.newELM.recall( self.frag_testcanvas[i,:] )
            self.labelmatrix[ i/math.ceil(canvasSize[1]/float(self.patcharm)), i%math.ceil(canvasSize[1]/float(self.patcharm))] = labelpatch.argmax()
        
        return self.labelmatrix, self.frag_testcanvas
        
        '''