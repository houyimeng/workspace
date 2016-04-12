
"""
Created on Wed Apr 06 11:34:17 2016

@author: ThinkPad
"""
from ELM.ELM import ELM
from ELM.loadImgData import loadImgData
from ELM.normalize import normalize
import matplotlib.pylab as plt
from numpy import reshape, random
import time

tic = time.time()
#load real-world cropped data
train_data0, train_label0, test_data0, test_label0 = loadImgData('binary')
#load synthesized data
#train_data1, train_label1, test_data1, test_label1 = loadImgData('synthesis')

train_data = normalize(train_data0)
test_data = normalize(test_data0)
train_label = train_label0
test_label = test_label0

numTr = train_data.shape[0]
numTe = test_data.shape[0]

result = []
ELMobj = ELM(28*28, 28*28*10, 10)

for i in range(numTr):
    if i%1000 == 0:
        print "Current training iteration is", i+1, "of", numTr
    ELMobj.train(train_data[i,:], reshape(train_label[i,:], (10,1)) )

for j in range(numTe):
    if j%1000 == 0:
        print "Current training iteration is", j+1, "of", numTe
    labelhat0 = ELMobj.recall(test_data[j,:])
    labeltrue = test_label[j,:].argmax()
    labelhat = labelhat0.argmax()    
    result.append(labeltrue == labelhat)
        
Acc = result.count(True)/ float(numTe)

toc = time.time()
print "Time Elapsed =", toc-tic
print "The classification accuracy is:", Acc
#ELMobj.save('C:\\ELMframework\\w8\\koutu')


def display():
    plt.figure(1)
    plt.imshow( reshape( train_data0[random.randint(1000),:], (28,28) ))
    plt.figure(2)
    plt.imshow( reshape( train_data1[random.randint(1000),:], (28,28) ))
    return None
