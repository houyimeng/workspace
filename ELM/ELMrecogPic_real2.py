
from ELM import ELM
import matplotlib.pylab as plt 
from numpy import array, uint8, ones, where, vstack, zeros, sqrt, absolute, argsort
from zeropadding import zeropadding
from distCal import distCal
from codetable import codetable
import cv2
import os

def find(lst, tar):
    return [i for i, x in enumerate(lst) if x == tar]

class ELMrecogPic_real2(object):
    
    def __init__(self):
        
        self.patcharm = 1296 #1296 #900
        self.Outsize = 35
        self.dim = 28
        self.newELM = ELM(self.patcharm, self.patcharm*10, self.Outsize)
        self.newELM.load('C:\\dataspace\\weights\\harbour35_28')          
        self.hog = cv2.HOGDescriptor((self.dim, self.dim), (8,8), (4,4), (4,4), 9)
   
    def runtest(self):
        
        # load tested image        
        crop_rg = [0.4, 0.6, 0.05, 0.7] 
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

        fltSize = self.dim
        halfSize = self.dim/2
        
        ## zeropadding
        testcanvas = zeropadding(self.testcanvas, halfSize, halfSize)    
        canvasSize = array(self.testcanvas.shape) + fltSize        

        ## convolving part
        tot_count = (canvasSize[0]-fltSize)*(canvasSize[1]-fltSize)
        self.labelmatrix = -1*ones(( canvasSize[0]-fltSize, canvasSize[1]-fltSize ))
        
        print "> Convolving in progress ..."
        for y in range(halfSize, canvasSize[0]-halfSize):
            stepsize = 2
            for x in range(halfSize, canvasSize[1]-halfSize):
                #print x,y
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
                        stepsize = 3
                        for i in range(-stepsize+1, 0):
                            for j in range(-stepsize+1, 0):
                                item_sec = testcanvas[y-halfSize+i:y+halfSize+i, x-halfSize+j:x+halfSize+j]
                                temp_sec = uint8(item_sec)
                                hist_sec = self.hog.compute(temp_sec)
                                labelhatpatch_sec = self.newELM.recall(hist_sec)
                                self.labelmatrix[y-halfSize+i, x-halfSize+j] = distCal(labelhatpatch_sec)
                    else:
                        if stepsize < 6:
                            stepsize += 2
                        else:
                            stepsize = 6
                            
        clust_temp = where(self.labelmatrix != -1) 
        data_coor = vstack((clust_temp[0], clust_temp[1]))        
        center = ones(data_coor.shape[1])
        
        numDigits = 2  
        for i in range(1, data_coor.shape[1]):
            coor = [data_coor[0,i], data_coor[1,i]]
            dist = zeros(i)
        
            for j in range(0, i):
                dist[j] = sqrt((coor[0] - data_coor[0, j])**2 + (coor[1] - data_coor[1, j])**2)
            if min(dist) <= halfSize:
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
            countnum = zeros(self.Outsize)
            # decicsion making   
            for k in clusters_idx:
                distfactor = absolute(data_coor[0, k]-centroids[i-1, 0]) + absolute(data_coor[1, k]-centroids[i-1, 1]) + 0.01 # euclidean distance
                countnum[self.labelmatrix[data_coor[0, k], data_coor[1, k]]] += 1/distfactor # equalization
                
            labelhat[i] = countnum.argmax()
        
            print "Number/letter :", i ,", Coordinates =", "(", int(centroids[i-1, 0]), ',', int(centroids[i-1, 1]), "), Prediction ->", int(labelhat[i])  

        
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
###################################################################################
#                Con, Iter = True, 0
#        separateWidth = halfSize       
#        
#        while Con and Iter < (numDigits * 10):
#            print "> Clustering entry No.",Iter+1, "..."
#            
#            clusters, centroids = kmean(datamatrix.T, numDigits)
#            Con = False
#            
#            for i in range(len(centroids)):
#                for j in range(len(centroids)):
#                    
#                    # bound detection 
#                    if i != j and absolute(centroids[i][0] - centroids[j][0]) <= separateWidth or  \
#                                  absolute(centroids[i][1] - centroids[j][1]) <= separateWidth :
#                        Con = True
#
#            Iter += 1
#        if Iter < numDigits * 10:        
#            print "Clustering Succeed!"
#        else:
#            print "Clustering failed, latest entry picked"
#        
#        centroids = array(centroids)
#        labelhat0 = zeros(numDigits)
#        countnum = zeros((self.Outsize, numDigits))
#
#        # decicsion making   
#        if clusters != []:
#            for i in range(len(clusters)):
#                for j in range(len(clusters[i])):
#                    temp = clusters[i][j]
#                    distfactor = absolute(temp[0]-centroids[i,0]) + absolute(temp[1]-centroids[i,1]) + 0.01 # euclidean distance
#                    countnum [self.labelmatrix[temp[0],temp[1]],i] += 1/distfactor # equalization
#
#                labelhat0[i] = countnum[:,i].argmax()
#                print "CLUSTER:",i ," Coordinate =", "(", int(centroids[i,0]), ",", int(centroids[i,1]), "), label:", int(labelhat0[i])  
#        else:
#            print "No data generated"
#            labelhat0 = -1