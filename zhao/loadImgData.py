
from numpy import zeros, array, concatenate, random
import cv2
import os

def loadImgData(folder, ratio=0.9):
    print ">>> Loading Data<<<"    
    
    train_data = []
    train_l = []
    test_data = []
    test_l = []
    
    if folder == 'negative':
        path =  'C:\\comingdata\\'+folder+'\\'
        dirs = os.listdir(path)  
        for item in dirs: 
                img = cv2.imread(path+item,0)
                
                trainOrtest = random.binomial(1, ratio)               
                if trainOrtest:
                    train_data.append( img.flatten() )
                else:
                    test_data.append( img.flatten() )
    else:
        for i in range(10):
            path = 'C:\\comingdata\\'+folder+'\\'+str(i)+'\\'
            dirs = os.listdir(path)
            for item in dirs: 
                img = cv2.imread(path+item,0)
                
                trainOrtest = random.binomial(1, ratio)
                if trainOrtest:
                    train_data.append( img.flatten() )
                    train_l.append(i)
                else:
                    test_data.append( img.flatten() )
                    test_l.append(i)
            print 10*(i+1), "% data loaded ... "
                
    train_data = array(train_data)
    N, M = train_data.shape
    train_l = array(train_l)
    if folder == 'negative':
        train_label = zeros((N,1))
    else:
        train_label = zeros((N, 10))
        train_label[range(N), train_l] = 1
    data = concatenate((train_data, train_label), axis=1)
    random.shuffle(data)
    train_data = data[:,0:M]
    train_label = data[:,M:]
    
    test_data = array(test_data)
    N, M = test_data.shape    
    if folder == 'negative':
        test_label = zeros((N,1))
    else:
        test_l = array(test_l)
        test_label = zeros((N, 10))
        test_label[range(N),test_l] = 1        
    data = concatenate((test_data, test_label), axis=1)
    random.shuffle(data)
    test_data = data[:,0:M]
    test_label = data[:,M:]
    
    print ">>> Loading data successful"
    print "Training data shape: ", train_data.shape
    print "Testing data shape: ", test_data.shape
    
    return train_data, train_label, test_data, test_label, data
    
train_data, train_label, test_data, test_label, data = loadImgData('negative')