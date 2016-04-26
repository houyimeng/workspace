
from ELM import ELM
import matplotlib.pylab as plt 
from numpy import array, uint8, ones, where, vstack, zeros, sqrt, hstack, argsort, random, std
from zeropadding import zeropadding
from distCal import *
from codetable import codetable
import cv2
import os

def find(lst, tar):
    return [i for i, x in enumerate(lst) if x == tar]

class ELMrecogPic_v2(object):
    
    def __init__(self):
        
        self.patcharm = [576, 900, 1296]
        self.Outsize = 34
        self.dim = [20, 24, 28]
        self.newELM_20 = ELM(self.patcharm[0], self.patcharm[0]*10, self.Outsize)
        self.newELM_24 = ELM(self.patcharm[1], self.patcharm[1]*10, self.Outsize)        
        self.newELM_28 = ELM(self.patcharm[2], self.patcharm[2]*10, self.Outsize)
        self.newELM_20.load('C:\\dataspace\\weights\\harbour34_20')  
        self.newELM_24.load('C:\\dataspace\\weights\\harbour34_24')   
        self.newELM_28.load('C:\\dataspace\\weights\\harbour34_28')                
        self.hog_20 = cv2.HOGDescriptor((self.dim[0], self.dim[0]), (8,8), (4,4), (4,4), 9)
        self.hog_24 = cv2.HOGDescriptor((self.dim[1], self.dim[1]), (8,8), (4,4), (4,4), 9)
        self.hog_28 = cv2.HOGDescriptor((self.dim[2], self.dim[2]), (8,8), (4,4), (4,4), 9)
        self.stepsize = 4
        self.centroidindex = None
   
    def runtest(self, path = 'C:\\dataspace\\harbour\\canvas\\'):

###############################################################################
        # load image & prep           
        #crop_rg = [0.05, 0.95, 0.05, 0.5]
        scaleFactor = 0.4       
        dirs = os.listdir(path)
        rand_idx = random.randint(len(dirs))
        img = cv2.imread(path+dirs[rand_idx], 0)
        #img = img[int(img.shape[0]*crop_rg[2]):int(img.shape[0]*crop_rg[3]), int(img.shape[1]*crop_rg[0]):int(img.shape[1]*crop_rg[1])]
        self.testcanvas = cv2.resize(img, (0,0), fx=scaleFactor, fy=scaleFactor)
        print "> Grabing a random canvas for testing ... "
        print '> Image SIZE:', self.testcanvas.shape
        print '> Step SIZE:', self.stepsize   
        
###############################################################################
        ## convolving part 1        
        fltSize_20 = self.dim[0]
        halfSize_20 = self.dim[0]/2
        
        ## zeropadding
        testcanvas = zeropadding(self.testcanvas, halfSize_20, halfSize_20)    
        canvasSize = array(self.testcanvas.shape) + fltSize_20   

        self.labelmatrix_20 = -1*ones(( canvasSize[0]-fltSize_20, canvasSize[1]-fltSize_20))
        self.confidence_20 = -1*ones(( canvasSize[0]-fltSize_20, canvasSize[1]-fltSize_20))        
        print "> Convolving part 1 in progress ..., WindowSize = (20,20)"
        for y in range(halfSize_20, canvasSize[0]-halfSize_20):
            for x in range(halfSize_20, canvasSize[1]-halfSize_20):
                    
                if x%self.stepsize == 0 and y%self.stepsize == 0:  
                    item_temp = testcanvas[y-halfSize_20:y+halfSize_20, x-halfSize_20:x+halfSize_20]
                    temp_img = uint8(item_temp)
                    hist = self.hog_20.compute(temp_img)
                    labelhatpatch = self.newELM_20.recall(hist)
                    label_val, conf = distCal(labelhatpatch)
                    self.confidence_20[y-halfSize_20, x-halfSize_20] = conf
                    self.labelmatrix_20[y-halfSize_20, x-halfSize_20] = label_val

###############################################################################
        ## convolving part 2
        fltSize_24 = self.dim[1]
        halfSize_24 = self.dim[1]/2
        
        ## zeropadding
        testcanvas = zeropadding(self.testcanvas, halfSize_24, halfSize_24)    
        canvasSize = array(self.testcanvas.shape) + fltSize_24   
        ## convolving part 1
        self.labelmatrix_24 = -1*ones(( canvasSize[0]-fltSize_24, canvasSize[1]-fltSize_24))
        self.confidence_24 = -1*ones(( canvasSize[0]-fltSize_24, canvasSize[1]-fltSize_24))         
        print "> Convolving part 2 in progress ..., WindowSize = (24,24)"
        for y in range(halfSize_24, canvasSize[0]-halfSize_24):
            for x in range(halfSize_24, canvasSize[1]-halfSize_24):
                    
                if x%self.stepsize == 0 and y%self.stepsize == 0:  
                    item_temp = testcanvas[y-halfSize_24:y+halfSize_24, x-halfSize_24:x+halfSize_24]
                    temp_img = uint8(item_temp)
                    hist = self.hog_24.compute(temp_img)
                    labelhatpatch = self.newELM_24.recall(hist)
                    label_val, conf = distCal(labelhatpatch)
                    self.confidence_24[y-halfSize_24, x-halfSize_24] = conf
                    self.labelmatrix_24[y-halfSize_24, x-halfSize_24] = label_val
                            
###############################################################################
        ## convolving part 3
        fltSize_28 = self.dim[2]
        halfSize_28 = self.dim[2]/2
        
        ## zeropadding
        testcanvas = zeropadding(self.testcanvas, halfSize_28, halfSize_28)    
        canvasSize = array(self.testcanvas.shape) + fltSize_28   
        ## convolving part 1
        self.labelmatrix_28 = -1*ones(( canvasSize[0]-fltSize_28, canvasSize[1]-fltSize_28))
        self.confidence_28 = -1*ones(( canvasSize[0]-fltSize_28, canvasSize[1]-fltSize_28))         
        print "> Convolving part 3 in progress ..., WindowSize = (28,28)"
        for y in range(halfSize_28, canvasSize[0]-halfSize_28):
            for x in range(halfSize_28, canvasSize[1]-halfSize_28):
                    
                if x%self.stepsize == 0 and y%self.stepsize == 0:  
                    item_temp = testcanvas[y-halfSize_28:y+halfSize_28, x-halfSize_28:x+halfSize_28]
                    temp_img = uint8(item_temp)
                    hist = self.hog_28.compute(temp_img)
                    labelhatpatch = self.newELM_28.recall(hist)
                    label_val, conf = distCal(labelhatpatch)
                    self.confidence_28[y-halfSize_28, x-halfSize_28] = conf
                    self.labelmatrix_28[y-halfSize_28, x-halfSize_28] = label_val
        
################################################################################
        # get several possible centers    
                        
        clust_temp_20 = where(self.labelmatrix_20 != -1)
        clust_temp_24 = where(self.labelmatrix_24 != -1)
        clust_temp_28 = where(self.labelmatrix_28 != -1)
        len_clust = array([len(clust_temp_20), len(clust_temp_24), len(clust_temp_28)])
        self.main_kernel = len_clust.argmax()
        
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
            if min(dist) <= halfSize_28:
                center[i] =  center[dist.argmin()]
            else:
                center[i] = numDigits
                numDigits += 1       
        
##############################################################################
        # make decision         
        self.labelhat = zeros(numDigits-1)
        self.centroids = zeros((2, numDigits-1))
        n_cout = 0
        
        for i in set(center):            
            clusters_idx = find(center, i)
            
            k_val = 10*ones(len(center))
            
            for k in clusters_idx:
                if k < len(clust_temp_20[0]):
                    k_val[k] = self.confidence_20[data_coor[0, k], data_coor[1, k]]
                elif len(clust_temp_20[0]) <= k < len(clust_temp_20[0]) + len(clust_temp_24[0]):
                    k_val[k] = self.confidence_24[data_coor[0, k], data_coor[1, k]] 
                else:
                    k_val[k] = self.confidence_28[data_coor[0, k], data_coor[1, k]]
                    
            self.centroids[0, n_cout] = data_coor[0, k_val.argmin()]
            self.centroids[1, n_cout] = data_coor[1, k_val.argmin()] 
            
            if k_val.argmin() < len(clust_temp_20[0]):
                self.labelhat[n_cout] = self.labelmatrix_20[self.centroids[0, n_cout], self.centroids[1, n_cout]]
            elif len(clust_temp_20[0]) <= k_val.argmin() < len(clust_temp_20[0]) + len(clust_temp_24[0]):
                self.labelhat[n_cout] = self.labelmatrix_24[self.centroids[0, n_cout], self.centroids[1, n_cout]] 
            else:
                self.labelhat[n_cout] = self.labelmatrix_28[self.centroids[0, n_cout], self.centroids[1, n_cout]]                 
        
            n_cout += 1
            
        data_var1 = std(self.centroids[0,:])
        data_var2 = std(self.centroids[1,:])

        if data_var1 > data_var2:
            self.centroidindex = argsort(self.centroids[0,:])
            self.lane_flg = 'v'
        else:
            self.centroidindex = argsort(self.centroids[1,:])
            self.lane_flg = 'h'

        n_start = 1
        for i in self.centroidindex:
            print "Prediction", n_start ,", Coordinates=", "(", int(self.centroids[0, i]), ',', int(self.centroids[1, i]), \
                    "), Label ->", codetable(int(self.labelhat[i]))         
            n_start += 1      
        
        return self.labelhat

    def savePATCH(self):
        dir_name = 'C:\\dataspace\\Recognition\\'
        n_start = 1
        kearm = self.dim[self.main_kernel]/2
        for ii in self.centroidindex:
            x_cor, y_cor = self.centroids[0,ii], self.centroids[1,ii]           
            if x_cor < kearm:
                img_coor = self.testcanvas[:int(x_cor+kearm), \
                                        int(y_cor-kearm):int(y_cor+kearm) ]
            elif y_cor < kearm:
                img_coor = self.testcanvas[int(x_cor-kearm):int(x_cor+kearm),\
                                           :int(y_cor+kearm) ]
            elif x_cor > self.testcanvas.shape[0] + kearm:
                img_coor = self.testcanvas[int(x_cor-kearm):, \
                                           int(y_cor-kearm):int(y_cor+kearm) ] 
            elif y_cor > self.testcanvas.shape[1] + kearm:
                img_coor = self.testcanvas[int(x_cor-kearm):int(x_cor+kearm), \
                                           :int(y_cor+kearm) ] 
            elif x_cor > self.testcanvas.shape[0] + kearm and y_cor > self.testcanvas.shape[1] + kearm:
                img_coor = self.testcanvas[int(x_cor-kearm):, int(y_cor-kearm):] 
                
            elif x_cor < kearm and y_cor < kearm:
                img_coor = self.testcanvas[:int(x_cor+kearm), :int(y_cor+kearm)] 
                
            else:                
                img_coor = self.testcanvas[int(x_cor-kearm):int(x_cor+kearm), \
                                        int(y_cor-kearm):int(y_cor+kearm) ]           
            file_name = 'IMG_'+str(n_start)+'_LABEL_'+str(codetable(self.labelhat[ii]))+'.jpg'
            cv2.imwrite( os.path.join(dir_name, file_name), img_coor)
            n_start += 1
        
    def visualize(self):
            
        fig = plt.figure()
        
        if self.lane_flg == 'v':
            fig.add_subplot(1,4,1)
            plt.imshow(self.testcanvas)
            fig.add_subplot(1,4,2)
            plt.imshow(self.labelmatrix_20)  
            fig.add_subplot(1,4,3)
            plt.imshow(self.labelmatrix_24)  
            fig.add_subplot(1,4,4)
            plt.imshow(self.labelmatrix_28)
        else:
            fig.add_subplot(4,1,1)
            plt.imshow(self.testcanvas)
            fig.add_subplot(4,1,2)
            plt.imshow(self.labelmatrix_20)  
            fig.add_subplot(4,1,3)
            plt.imshow(self.labelmatrix_24)  
            fig.add_subplot(4,1,4)
            plt.imshow(self.labelmatrix_28) 
        

