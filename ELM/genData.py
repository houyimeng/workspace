# -*- coding: utf-8 -*-
"""
Created on Fri Mar 25 13:25:02 2016

@author: ThinkPad
"""

from MNISTDataset import MNISTDataset
from numpy import zeros, array, random, reshape, vstack, ones, hstack
from zeropadding import zeropadding
from loadImgData import loadImgData
import math
import pickle

class genData(object):
    
    def __init__(self, dtsource = 'MNIST'):
        
        self.inputsize = 18
        self.halfsize = self.inputsize/2
        self.dtsource = dtsource
        self.argu_flg = 0
        self.numLabel = 10
        
        if self.dtsource == 'MNIST':
            self.numtr = 60000
            self.numte = 10000
            self.num11 = int((self.numtr+self.numte)/10) 
            
            if self.argu_flg == 1:
                self.numArgu = 30000       
                self.numSum = self.numtr+self.numte+self.numArgu+self.num11
            else:
                self.numSum = self.numtr+self.numte+self.num11
                
            self.dataset = zeros((self.numSum, self.inputsize**2))
            self.labels = zeros((self.numSum, self.numLabel))
            
            self.genMNISTData()
        else:           
            itemTr, labelTr, itemTe, labelTe = loadImgData(self.dtsource)
            self.numtr = itemTr.shape[0]
            self.numte = itemTe.shape[0] 
            self.num11 = int((self.numtr+self.numte))
            self.numSum = self.numtr+self.numte+self.num11
            self.dataset = zeros((self.numSum, self.inputsize**2))
            self.labels = zeros((self.numSum, self.numLabel))
           
            self.dataset[:self.numtr+self.numte,:] = vstack(( itemTr, itemTe ))
            self.labels[:self.numtr+self.numte,0:10] = vstack(( labelTr, labelTe ))

        self.make11Data()
        
        shuffle_idx = random.permutation(self.numSum)
        self.dataset = self.dataset[shuffle_idx]
        self.labels = self.labels[shuffle_idx]
    
    @property
    def getdata(self):
        return self.dataset, self.labels
         
    def genMNISTData(self):  
        
        print "Now making MNIST dataset..."        
        MNISTdata = MNISTDataset("MNIST")        

        for i in range(self.numtr):
                
            label_tr, img_tr = MNISTdata.getTrainingItem(i)
            self.dataset[i,:] = img_tr
            self.labels[i,0:10] = label_tr[0:10].flatten()
            
        for i in range(self.numte):
                
            label_te, img_te = MNISTdata.getTestingItem(i)
            self.dataset[self.numtr+i,:] = img_te
            self.labels[self.numtr+i,0:10] = label_te[0:10].flatten()
        
        if self.argu_flg == 1:
            datasetArgu, labelsetArgu = self.genArgumentMNIST()        
            for i in range(self.numArgu):
                self.dataset[self.numtr+self.numte+i,:] = datasetArgu[i,:]
                self.labels[self.numtr+self.numte+i,0:10] = labelsetArgu[i,0:10]
        else:
            pass
        
    def make11Data(self):

        num1 = math.ceil(self.num11/float(2))
        num2 = math.floor(self.num11/float(2))
        
        dataset_none1 = self.genNoneLabels0(num1)
        dataset_none2 = self.genNoneLabels1(num2)
        
        if self.numLabel == 11:
            label_none1 =  hstack(( zeros((num1,10)), ones((num1, 1)) ))
            label_none2 =  hstack(( zeros((num2,10)), ones((num2, 1)) ))
        else:
            label_none1 =  zeros((num1,10))
            label_none2 =  zeros((num2,10))
            
        dataset_none = vstack((dataset_none1, dataset_none2))
        label_none = vstack(( label_none1, label_none2 )) 

        self.dataset[-self.num11:,:] = dataset_none
        self.labels[-self.num11:,:] = label_none
        
    def genNoneLabels0(self, nums):
        
        print 'Generating None label type1 ...'
        dataset_none0 = zeros((nums, self.inputsize**2))
        rounds = 10
        factor = 4
        
        for iters in range(rounds):
            canvas, truelabelmatrix = self.lineIMAGE()
            print canvas.shape                
            
            container = []
            for i in range(self.halfsize, canvas.shape[0]-self.halfsize):
                for j in range(self.halfsize, canvas.shape[1]-self.halfsize):                    
                    if (canvas[i-self.halfsize:i+self.halfsize, j-self.halfsize:j+self.halfsize] == 0).tolist().count(True) == self.inputsize**2:
                        continue                    
                    elif self.halfsize-self.inputsize/factor <= i%self.inputsize <= self.halfsize+self.inputsize/factor or self.halfsize-self.inputsize/factor <= j%self.inputsize <= self.halfsize+self.inputsize/factor:
                        continue
                    else:
                        poe = canvas[i-self.halfsize:i+self.halfsize, j-self.halfsize:j+self.halfsize]
                        print poe.shape
                        temp_img = reshape(canvas[i-self.halfsize:i+self.halfsize, j-self.halfsize:j+self.halfsize], self.inputsize**2)
                        container.append(temp_img)
                        
        idx0 = random.choice(len(container), nums, replace=False)
        n0 = 0
        for sample in idx0:            
            dataset_none0[n0,:] = array(container[sample])
            n0 += 1
                
        return dataset_none0
    
    def genNoneLabels1(self, nums):
        
        print 'Generating None label type2 ...'
        dataset_none1 = zeros((nums, self.inputsize**2))
        
        rand_idx_num = random.choice(self.numtr+self.numte, nums*2, replace=False)
        factor = 4
        
        n = 0
        for i in rand_idx_num: 
            if n == nums-1:
                break
            
            temp = reshape( self.dataset[i,:], (self.inputsize, self.inputsize))
            padimg = zeropadding(temp, self.inputsize, self.inputsize)
            
            if random.randint(0,2) == 0:
                rand_img_X = random.randint(self.halfsize, self.inputsize+self.inputsize/factor+1)
            else:
                rand_img_X = random.randint(self.halfsize*3+self.inputsize/factor,self.inputsize*3-self.halfsize+1)                
            if random.randint(0,2) == 0:
                rand_img_Y = random.randint(self.halfsize, self.inputsize+self.inputsize/factor+1)
            else:
                rand_img_Y = random.randint(self.halfsize*3+self.inputsize/factor,self.inputsize*3-self.halfsize+1)
            
            temp_img = padimg[rand_img_X-self.halfsize:rand_img_X+self.halfsize, rand_img_Y-self.halfsize:rand_img_Y+self.halfsize]
            
            if (temp_img == 0).tolist().count(True) == self.inputsize**2:
                continue
            else:
                dataset_none1[n,:] = temp_img.flatten()
                n += 1

        return dataset_none1
        
    def lineIMAGE(self, numDigits = 100, edgesize = 0):
        
        num_row = int((numDigits-1)/10)+1         
        rowpixel = num_row*self.inputsize+ (num_row-1)*edgesize
        
        if numDigits > 10:
            colpixel = 10*self.inputsize +edgesize*9
        else:
            colpixel = numDigits*self.inputsize + (numDigits-1)*edgesize
        
        canvas = zeros((rowpixel, colpixel))
        truelabelmatrix = zeros((numDigits))        
        idxvector = random.choice(self.numte+self.numtr, self.numte+self.numtr, replace=False)
    
        for j in range(numDigits):
            
                img = self.dataset[idxvector[j],:]
                truelabelmatrix[j] = self.labels[idxvector[j]].argmax()
                
                temp = reshape(img, (self.inputsize, self.inputsize))
                canvas[(self.inputsize+edgesize)*int(j/10):(self.inputsize+edgesize)*int(j/10)+self.inputsize , \
                       (self.inputsize+edgesize)*(j%10):(self.inputsize+edgesize)*(j%10)+self.inputsize] = temp
    
        return canvas, array(truelabelmatrix, dtype='int')
    
    def genArgumentMNIST(self):
        
        print "generating Argument MNIST image in progress..."
        dataset = MNISTDataset("MNIST")
        gendataset = []
        genlabelset = []
        stepsizeX = array([0, 2, 6, 8])
        stepsizeY = array([0, 2, 6, 8])
        
        rd_idx0 = random.choice(60000, 10000, replace=False)
        
        for i in rd_idx0:
            labelTr, itemTr = dataset.getTrainingItem(i)
            temp_tr = reshape( itemTr, (28,28))
            itemTr_pad = zeropadding(temp_tr, 4, 4)
            
            for x in stepsizeX:
                for y in stepsizeY:
                    TEMP = itemTr_pad[x:x+28, y:y+28].flatten()
                    gendataset.append( TEMP )
                    genlabelset.append( labelTr.flatten() )
                    
        print "Argument MNIST image complete!"
        rd_idx1 = random.choice(len(gendataset), self.numArgu, replace=False)
        
        popdatamatrix = zeros((self.numArgu, 28*28))
        poplabelmatrix = zeros((self.numArgu, 10))  
        
        for i in range(self.numArgu):
            popdatamatrix[i,:] = gendataset[rd_idx1[i]]
            poplabelmatrix[i,:] = genlabelset[rd_idx1[i]]
            
        return popdatamatrix, poplabelmatrix
    
    def save(self, Filename):

        Filename = 'MNIST_dataset'
        dataDict = {'datacontent':self.dataset, 'label':self.labels}                    
        with open(Filename, 'wb') as f:
            pickle.dump(dataDict , f)
            
    def load(self, Filename):
        
        with open(Filename, 'rb') as f:
            data_dict = pickle.load(f)

        self.dataset = data_dict['datacontent']
        self.labels = data_dict['label']

  
    
    
    