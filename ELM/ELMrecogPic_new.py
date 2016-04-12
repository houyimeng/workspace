
from ELM import ELM
from MNISTCanvasImage import MNISTCanvasImage
import matplotlib.pylab as plt 
from numpy import zeros, where, vstack, array, reshape, ones, argsort, absolute, random, mean
from kmean import kmean
from ImageToolbox.zeropadding import zeropadding
from lineMNIST import lineMNIST
from lineIMAGE import lineIMAGE
from distCal import distCal
from PIL import Image
import os

class ELMrecogPic_new(object):
    
    def __init__(self):
        
        self.patcharm = 28
        self.Outsize = 10 
        self.newELM = ELM(self.patcharm**2, (self.patcharm**2)*10, self.Outsize)
        self.newELM.load('SYNdata')          
        
        self.truecolormatrix =  None
        self.hatcolormatrix =  None
        self.testcanvas0 = None
        self.labelmatrix = None
   
    def runtest(self, numDigits, method = 'line', canvasSize0 = [], paddingMode = 0, edgeSize = 0):
        
        fltSize = 28
        halfSize = 14
        displaysize = 3
        Lane = 'h'
        DigitLane = 20
        
        if method == 'rand':
            
            imgcanvas = MNISTCanvasImage('MNIST', canvasSize0)    
            self.testcanvas0, labeltrue = imgcanvas.generateImage(numDigits)            
            
        elif method == 'line':   
            
            if self.datasource == 'MNIST':
                testcanvas_temp, labeltrue, canvasSize_info = lineMNIST(numDigits, edgeSize, DigitLane, Lane)
            elif self.datasource == 'NINGBO':
                testcanvas_temp, labeltrue, canvasSize_info = lineIMAGE(numDigits, edgeSize, DigitLane, Lane)
                
            if paddingMode == 1:                
                self.testcanvas0 = zeros(canvasSize0)
                idx1 = random.randint(0, canvasSize0[0]-canvasSize_info[0]-1)
                idx2 = random.randint(0, canvasSize0[1]-canvasSize_info[1]-1)
                self.testcanvas0[idx1:idx1+canvasSize_info[0], idx2:idx2+canvasSize_info[1]] = testcanvas_temp
            
            else:
                self.testcanvas0 = testcanvas_temp
                canvasSize0 = canvasSize_info
                
        elif method == 'ext':
            path = '/home/houyimeng/testcanvas/'
            dirs = os.listdir(path)
            extcanvas = []
            for item in dirs:
                temp_img = Image.open( path+item )
                temp_img = temp_img.crop(( int(temp_img.size[0]/2), 1, temp_img.size[0]-1, temp_img.size[1]-1))
                SIZE = ( int(temp_img.size[0]*1.6), int(temp_img.size[1]*1.6) )
                print SIZE
                temp_img = temp_img.resize( SIZE, Image.BILINEAR)
                extcanvas.append( array( (temp_img.getdata()-mean(temp_img.getdata()))>0 , dtype='int')) 
            
            print extcanvas[0].shape
            self.testcanvas0 = reshape( extcanvas[0], (SIZE[1], SIZE[0]))

            canvasSize0 = [self.testcanvas0.shape[0], self.testcanvas0.shape[1]]
            labeltrue = zeros(numDigits)
            
        ## zeropadding
        testcanvas = zeropadding(self.testcanvas0, halfSize, halfSize)    
        canvasSize = array(canvasSize0) + fltSize        
        
        #return self.testcanvas0        
        
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
                
        if Lane == 'h':
            centroidindex = argsort(centroids[:,1])
        elif Lane == 'v':
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
        
