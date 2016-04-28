from ELM import ELM
import matplotlib.pylab as plt 
from numpy import array, uint8, ones, where, vstack, zeros, sqrt, hstack, argsort, random, std, absolute
from zeropadding import zeropadding
from distCal import *
from codetable import codetable
from showNumLet import showNumLet
import cv2
import os

def find(lst, tar):
    return [i for i, x in enumerate(lst) if x == tar]

class ELMrecogPic_v2_new(object):
    
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
        scaleFactor = 0.5   
        dirs = os.listdir(path)
        rand_idx = random.randint(len(dirs))
        img = cv2.imread(path+dirs[14], 0)
        self.testcanvas = cv2.resize(img, (0,0), fx=scaleFactor, fy=scaleFactor)
        print "> Grabing a random canvas", dirs[rand_idx] , "for testing ... "
        print '> Image SIZE:', self.testcanvas.shape
        print '> Step SIZE:', self.stepsize   
        
###############################################################################
        ## convolving part 1        
        fltSize_20 = self.dim[0]
        halfSize_20 = self.dim[0]/2
        boundry_thresh = 14
        ## zeropadding
        testcanvas = zeropadding(self.testcanvas, halfSize_20, halfSize_20)    
        canvasSize = array(self.testcanvas.shape) + fltSize_20   

        self.labelmatrix_20 = -1*ones(( canvasSize[0]-fltSize_20, canvasSize[1]-fltSize_20))
        self.confidence_20 = 10*ones(( canvasSize[0]-fltSize_20, canvasSize[1]-fltSize_20))        
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
        self.confidence_24 = 10*ones(( canvasSize[0]-fltSize_24, canvasSize[1]-fltSize_24))         
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
        self.confidence_28 = 10*ones(( canvasSize[0]-fltSize_28, canvasSize[1]-fltSize_28))         
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
        print "Start post-convolving process ..."
                
        clust_temp_20 = where(self.labelmatrix_20 != -1)
        clust_temp_24 = where(self.labelmatrix_24 != -1)
        clust_temp_28 = where(self.labelmatrix_28 != -1)        
        clust_temp_x = hstack((clust_temp_20[0], clust_temp_24[0], clust_temp_28[0]))
        clust_temp_y = hstack((clust_temp_20[1], clust_temp_24[1], clust_temp_28[1]))
        data_coor = vstack((clust_temp_x, clust_temp_y))
            
        center = ones(data_coor.shape[1])
        num_center0 = zeros(data_coor.shape[1]) # to be cut    
        numDigits = 0  
        mean_center = []

        for i in range(0, data_coor.shape[1]):
            # get coordinates
            coor = array([data_coor[0,i], data_coor[1,i]]) 
            
            if i  == 0:
                mean_center.append( coor )
                center[i] = numDigits
                num_center0[numDigits] += 1
                numDigits += 1

            elif i > 0 :                   
                dist = zeros(numDigits)        
                for j in range(0, numDigits):
                    mean_center_tmp = mean_center[j]
                    tmp_coor_x = mean_center_tmp[0]
                    tmp_coor_y = mean_center_tmp[1]
                    
                    dist[j] = max(absolute(coor[0] - tmp_coor_x) , absolute(coor[1] - tmp_coor_y))
                     
                if min(dist) <= boundry_thresh:
                    min_idx = dist.argmin() 
                    center[i] =  center[min_idx]
                    mean_tmp_new = ( mean_center[min_idx] * num_center0[min_idx] + coor )/(num_center0[min_idx]+1)
                    mean_center[min_idx] = mean_tmp_new
                    num_center0[min_idx] += 1
                    
                else:
                    center[i] = numDigits
                    mean_center.append( coor )
                    num_center0[numDigits] += 1
                    numDigits += 1 
                    
        #center_last_idx = find(num_center0, 0)            
        #num_center = num_center0[:center_last_idx[0]]

##############################################################################
        # make decision         
        self.labelhat = zeros(numDigits)
        self.centroids = zeros((2, numDigits))
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

        # show mean center 
        self.centermatrix = zeros(self.testcanvas.shape)
        self.centermatrix[0,0] = 2
        
        n_start = 1
        for i in self.centroidindex:
            print "Prediction", n_start ,", Coordinates=", "(", int(self.centroids[0, i]), ',', int(self.centroids[1, i]), \
                    "), Label ->", codetable(int(self.labelhat[i]))         
            
            center_coor = mean_center[i]
            img_show = showNumLet( str(int(self.labelhat[i])) )
            self.centermatrix[center_coor[0]-2:center_coor[0]+3, \
                              center_coor[1]-1:center_coor[1]+2] = img_show
            n_start += 1  
                 
        return self.labelhat

    def savePATCH(self):
        dir_name = 'C:\\dataspace\\Recognition\\'
        n_start = 1
        kearm1 = 28/2
        kearm = kearm1*1.3
        for ii in self.centroidindex:
            x_cor, y_cor = self.centroids[0,ii], self.centroids[1,ii]           
            if x_cor < kearm:
                img_coor = self.testcanvas[:int(x_cor+kearm), \
                                        int(y_cor-kearm1):int(y_cor+kearm1) ]
            elif y_cor < kearm1:
                img_coor = self.testcanvas[int(x_cor-kearm):int(x_cor+kearm),\
                                           :int(y_cor+kearm1) ]
            elif x_cor > self.testcanvas.shape[0] + kearm:
                img_coor = self.testcanvas[int(x_cor-kearm):, \
                                           int(y_cor-kearm1):int(y_cor+kearm1) ] 
            elif y_cor > self.testcanvas.shape[1] + kearm1:
                img_coor = self.testcanvas[int(x_cor-kearm):int(x_cor+kearm), \
                                           :int(y_cor+kearm1) ] 
            elif x_cor > self.testcanvas.shape[0] + kearm and y_cor > self.testcanvas.shape[1] + kearm1:
                img_coor = self.testcanvas[int(x_cor-kearm):, int(y_cor-kearm1):] 
                
            elif x_cor < kearm and y_cor < kearm1:
                img_coor = self.testcanvas[:int(x_cor+kearm), :int(y_cor+kearm1)] 
                
            else:                
                img_coor = self.testcanvas[int(x_cor-kearm):int(x_cor+kearm), \
                                        int(y_cor-kearm1):int(y_cor+kearm1) ]           
            file_name = 'IMG_'+str(n_start)+'_LABEL_'+str(codetable(self.labelhat[ii]))+'.jpg'
            cv2.imwrite( os.path.join(dir_name, file_name), img_coor)
            n_start += 1
        
    def visualize(self):
            
        fig = plt.figure(0)
        
        if self.lane_flg == 'v':
            fig.add_subplot(1,2,1)
            plt.imshow(self.testcanvas)
            fig.add_subplot(1,2,2)
            plt.imshow(self.centermatrix)  

        else:
            fig.add_subplot(2,1,1)
            plt.imshow(self.testcanvas)
            fig.add_subplot(2,1,2)
            plt.imshow(self.centermatrix)  

        