

# this is a script which could run multiple simulation

from MNISTdataset import MNISTdataset
from ELM import *
import numpy
import time

tic = time.time()

# init
size_input = 14*14
size_output = 10
Factor = 10

N_train = 6000
N_test = 1000
genWeights_type = 'bin' # 'bin' for binary weights and 'dec' for float weights
OPIUM_type = 'lite' #'basic' or 'lite'

elist1 = [] # list containing the index of error classification testing example
elist2 = []
elist3 = []
elist4 = []

N_sim = 2
class_errors = zeros((N_sim, 4))
Accuracy = zeros((N_sim, 4))

MNIST = MNISTdataset("data")

for ii in range(N_sim):
    
    print "#### Current No. of Simulation: %d of %d ####"%(ii+1, N_sim)
    print "..."
    
    # init class
    q1 = ELM(size_input, size_output, Factor, OPIUM_type, genWeights_type)
    q2 = ELM(size_input, size_output, Factor, OPIUM_type, genWeights_type)
    q3 = ELM(size_input, size_output, Factor, OPIUM_type, genWeights_type)
    q4 = ELM(size_input, size_output, Factor, OPIUM_type, genWeights_type)

    for i in range(N_train):
        label_train, item_train = MNIST.GetTrainingItem(i)
        item_train = reshape(item_train,(28,28))
        
        img00 = reshape(item_train[0:14,0:14], (14*14,1))
        img01 = reshape(item_train[0:14,14:28], (14*14,1))
        img10 = reshape(item_train[14:28,0:14], (14*14,1))
        img11 = reshape(item_train[14:28,14:28], (14*14,1))
    
        outq1 = q1.train(img00, label_train)
        outq2 = q2.train(img01, label_train)
        outq3 = q3.train(img10, label_train)
        outq4 = q4.train(img11, label_train)
        
        if (i+1)%100 == 0:        
            print "Current # of training iter is %d of %d"%(i+1, N_train)
            
    print ">>> Training complete <<<"
            
    for i in range(N_test):
                
        label_test, item_test = MNIST.GetTestingItem(i)
        item_test = reshape(item_test,(28,28))
                
        img00 = reshape(item_test[0:14,0:14], (14*14,1))
        img01 = reshape(item_test[0:14,14:28], (14*14,1))
        img10 = reshape(item_test[14:28,0:14], (14*14,1))
        img11 = reshape(item_test[14:28,14:28], (14*14,1))
                
        output_hat1 = q1.recall(img00)
        output_hat2 = q2.recall(img01)
        output_hat3 = q3.recall(img10)
        output_hat4 = q4.recall(img11)
        
        output_hat_max1 = output_hat1.argmax()
        output_hat_max2 = output_hat2.argmax()
        output_hat_max3 = output_hat3.argmax()
        output_hat_max4 = output_hat4.argmax()
    
        output_max = label_test.argmax()
        
        if (output_hat_max1!= output_max):
            class_errors[ii][0] +=1
            elist1.append(i)
            
        if (output_hat_max2!= output_max):
            class_errors[ii][1] +=1
            elist2.append(i)
                
        if (output_hat_max3!= output_max):
            class_errors[ii][2] +=1
            elist3.append(i)
        
        if (output_hat_max4!= output_max):
            class_errors[ii][3] +=1
            elist4.append(i)
        
        if (j+1)%100 == 0:
            print "Current # of testing iter is %d of %d"%(i+1, N_test)
            
    elist1.append(-1)
    elist2.append(-1)
    elist3.append(-1)
    elist4.append(-1)
    
    print ">>> Testing complete <<<"

for l in range(N_sim):
    Accuracy[l][:] = 1-class_errors[l][:]/float(N_test)

toc = time.time()
timeElapsed = toc-tic

mean_acc = Accuracy.mean(0)

print ""
print "TimeElapsed = %6.2f seconds "%timeElapsed
print "The Accuracy is :",'\n', Accuracy 
print "The mean Accuracy is :", mean_acc  

#q2.save('q2.pickle')
#qq = ELM(size_input, size_output, Factor, OPIUM_type, genWeights_type)
#qq.load('q2.pickle')
