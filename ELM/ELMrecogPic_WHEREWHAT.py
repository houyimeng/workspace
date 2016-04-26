from ELM import ELM
import matplotlib.pylab as plt 
from numpy import array, uint8, ones, where, vstack, zeros, sqrt, absolute, hstack
from zeropadding import zeropadding
from distCal import distCal, distCal_bin, distCal_normal
from codetable import codetable
from massflt import massflt
import cv2
import os

def find(lst, tar):
    return [i for i, x in enumerate(lst) if x == tar]

class ELMrecogPic_real_where(object):
    
    def __init__(self):
        
        self.patcharm = [576, 900, 1296] #1296 #900
        self.Outsize = [34, 2]
        self.dim = [20, 24, 28]
        
        self.newELM_20 = ELM(self.patcharm[0], self.patcharm[0]*10, self.Outsize[0])
        self.newELM_24 = ELM(self.patcharm[1], self.patcharm[1]*10, self.Outsize[0])        
        self.newELM_28 = ELM(self.patcharm[2], self.patcharm[2]*10, self.Outsize[0])
        self.newELM_20_binary = ELM(self.patcharm[0], self.patcharm[0]*10, self.Outsize[1])
        self.newELM_24_binary = ELM(self.patcharm[1], self.patcharm[1]*10, self.Outsize[1])
        self.newELM_28_binary = ELM(self.patcharm[2], self.patcharm[2]*10, self.Outsize[1])
        
        self.newELM_20.load('/home/houyimeng/dataspace/weights/harbour34_20')  
        self.newELM_24.load('/home/houyimeng/dataspace/weights/harbour34_24')    
        self.newELM_28.load('/home/houyimeng/dataspace/weights/harbour34_28')
        self.newELM_20_binary.load('/home/houyimeng/dataspace/weights/harbour2_20')  
        self.newELM_24_binary.load('/home/houyimeng/dataspace/weights/harbour2_24')    
        self.newELM_28_binary.load('/home/houyimeng/dataspace/weights/harbour2_28')
                
        self.hog_20 = cv2.HOGDescriptor((self.dim[0], self.dim[0]), (8,8), (4,4), (4,4), 9)
        self.hog_24 = cv2.HOGDescriptor((self.dim[1], self.dim[1]), (8,8), (4,4), (4,4), 9)
        self.hog_28 = cv2.HOGDescriptor((self.dim[2], self.dim[2]), (8,8), (4,4), (4,4), 9)
   
    def runtest(self):
        
        # load tested image        
        crop_rg = [0.5, 0.6, 0.25, 0.8]
        scaleFactor = 0.6               
        path = '/home/houyimeng/dataspace/images/canvas/'
        dirs = os.listdir(path)
        for item in dirs: 
            img = cv2.imread(path+item, 0)
            img = img[int(img.shape[0]*crop_rg[2]):int(img.shape[0]*crop_rg[3]), int(img.shape[1]*crop_rg[0]):int(img.shape[1]*crop_rg[1])]
            self.testcanvas = cv2.resize(img, (0,0), fx=scaleFactor, fy=scaleFactor)
            print "The size of the testcanvas:", self.testcanvas.shape
            cv2.imwrite('greyscale.png', img)
        
        stepsize = 2
###############################################################################
        
        fltSize_20 = self.dim[0]
        halfSize_20 = self.dim[0]/2
        
        ## zeropadding
        testcanvas = zeropadding(self.testcanvas, halfSize_20, halfSize_20)    
        canvasSize = array(self.testcanvas.shape) + fltSize_20   
        ## convolving part 1
        tot_count = (canvasSize[0]-fltSize_20)*(canvasSize[1]-fltSize_20)
        labelmatrix_20_raw = -1*ones(( canvasSize[0]-fltSize_20, canvasSize[1]-fltSize_20))
        
        n_cout = 0
        print "> Convolving part 1 in progress ..., WindowSize = (20,20)"
        for y in range(halfSize_20, canvasSize[0]-halfSize_20):
            for x in range(halfSize_20, canvasSize[1]-halfSize_20):
                n_cout += 1
                if n_cout%2000 == 0:
                    print "> Current convolving index is %d ( %d in total )..."%(n_cout, tot_count)
                    
                if x%stepsize == 0 and y%stepsize == 0:  
                    item_temp = testcanvas[y-halfSize_20:y+halfSize_20, x-halfSize_20:x+halfSize_20]
                    temp_img = uint8(item_temp)
                    hist = self.hog_20.compute(temp_img)
                    labelhatpatch = self.newELM_20_binary.recall(hist)
                    label_val = distCal_bin(labelhatpatch)
                    if label_val == 1:
                        labelhatpatch_34 = self.newELM_20.recall(hist)
                        label_val_34 = distCal_normal(labelhatpatch_34)
                        labelmatrix_20_raw[y-halfSize_20, x-halfSize_20] = label_val_34
                    else:
                        labelmatrix_20_raw[y-halfSize_20, x-halfSize_20] = label_val

###############################################################################
        ## convolving part 2
        fltSize_24 = self.dim[1]
        halfSize_24 = self.dim[1]/2
        
        ## zeropadding
        testcanvas = zeropadding(self.testcanvas, halfSize_24, halfSize_24)    
        canvasSize = array(self.testcanvas.shape) + fltSize_24   
        ## convolving part 1
        tot_count = (canvasSize[0]-fltSize_24)*(canvasSize[1]-fltSize_24)
        labelmatrix_24_raw = -1*ones(( canvasSize[0]-fltSize_24, canvasSize[1]-fltSize_24))
        n_cout = 0
        print "> Convolving part 2 in progress ..., WindowSize = (24,24)"
        for y in range(halfSize_24, canvasSize[0]-halfSize_24):
            for x in range(halfSize_24, canvasSize[1]-halfSize_24):
                n_cout += 1
                if n_cout%2000 == 0:
                    print "> Current convolving index is %d ( %d in total )..."%(n_cout, tot_count)
                    
                if x%stepsize == 0 and y%stepsize == 0:  
                    item_temp = testcanvas[y-halfSize_24:y+halfSize_24, x-halfSize_24:x+halfSize_24]
                    temp_img = uint8(item_temp)
                    hist = self.hog_24.compute(temp_img)
                    labelhatpatch = self.newELM_24_binary.recall(hist)
                    label_val = distCal_bin(labelhatpatch)
                    if label_val == 1:
                        labelhatpatch_34 = self.newELM_24.recall(hist)
                        label_val_34 = distCal_normal(labelhatpatch_34)
                        labelmatrix_24_raw[y-halfSize_24, x-halfSize_24] = label_val_34
                    else:
                        labelmatrix_24_raw[y-halfSize_24, x-halfSize_24] = label_val                    
                                     
###############################################################################
        ## convolving part 3
        fltSize_28 = self.dim[2]
        halfSize_28 = self.dim[2]/2
        
        ## zeropadding
        testcanvas = zeropadding(self.testcanvas, halfSize_28, halfSize_28)    
        canvasSize = array(self.testcanvas.shape) + fltSize_28   
        ## convolving part 1
        tot_count = (canvasSize[0]-fltSize_28)*(canvasSize[1]-fltSize_28)
        labelmatrix_28_raw = -1*ones(( canvasSize[0]-fltSize_28, canvasSize[1]-fltSize_28))
        n_cout = 0
        print "> Convolving part 3 in progress ..., WindowSize = (28,28)"
        for y in range(halfSize_28, canvasSize[0]-halfSize_28):
            for x in range(halfSize_28, canvasSize[1]-halfSize_28):
                n_cout += 1
                if n_cout%2000 == 0:
                    print "> Current convolving index is %d ( %d in total )..."%(n_cout, tot_count)
                    
                if x%stepsize == 0 and y%stepsize == 0:  
                    item_temp = testcanvas[y-halfSize_28:y+halfSize_28, x-halfSize_28:x+halfSize_28]
                    temp_img = uint8(item_temp)
                    hist = self.hog_28.compute(temp_img)
                    labelhatpatch = self.newELM_28_binary.recall(hist)
                    label_val = distCal_bin(labelhatpatch)
                    if label_val == 1:
                        labelhatpatch_34 = self.newELM_28.recall(hist)
                        label_val_34 = distCal_normal(labelhatpatch_34)
                        labelmatrix_28_raw[y-halfSize_28, x-halfSize_28] = label_val_34
                    else:
                        labelmatrix_28_raw[y-halfSize_28, x-halfSize_28] = label_val

################################################################################
        # get several possible centers
        self.labelmatrix_20 = massflt(labelmatrix_20_raw, 23)
        self.labelmatrix_24 = massflt(labelmatrix_24_raw, 23)
        self.labelmatrix_28 = massflt(labelmatrix_28_raw, 23)       
                               
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