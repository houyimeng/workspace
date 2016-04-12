# hierarchical ELM

from MNISTDataset import MNISTDataset
from ELM import *
import numpy
import time

tic = time.time()

# init
size_input1, size_input2 = 14*14, 10*4
size_output, Factor = 10, 10
N_train, N_test = 6000, 1000
genWeights_type, OPIUM_type = 'bin', 'basic'

#
N_sim = 1
elist = []
class_errors = zeros((N_sim, 1))
Accuracy = []

# load dataset
MNIST = MNISTdataset("data")

for ii in range(N_sim):
    
    print "#### Current No. of Simulation: %d of %d ####"%(ii+1, N_sim)
    print "..."
    
    # init class
    q1 = ELM(size_input1, size_output, Factor, OPIUM_type, genWeights_type)
    q2 = ELM(size_input1, size_output, Factor, OPIUM_type, genWeights_type)
    q3 = ELM(size_input1, size_output, Factor, OPIUM_type, genWeights_type)
    q4 = ELM(size_input1, size_output, Factor, OPIUM_type, genWeights_type)
    q =  ELM(size_input2, size_output, Factor, OPIUM_type, genWeights_type)
    
  
    for i in range(N_train):
        
        # stage 1  
        label_train, item_train = MNIST.GetTrainingItem(i)
        item_train = reshape(item_train,(28,28))
        
        img00 = reshape(item_train[0:14,0:14], (14*14,1))
        img01 = reshape(item_train[0:14,14:28], (14*14,1))
        img10 = reshape(item_train[14:28,0:14], (14*14,1))
        img11 = reshape(item_train[14:28,14:28], (14*14,1))
    
        in_q1 = q1.train(img00, label_train)
        in_q2 = q2.train(img01, label_train)
        in_q3 = q3.train(img10, label_train)
        in_q4 = q4.train(img11, label_train)
        
        # stage 2    
        input_cat = concatenate((in_q1, in_q2, in_q3, in_q4))
        q.train(input_cat, label_train)

        if (i+1)%100 == 0:        
            print "Current # of training iter is %d of %d"%(i+1, N_train)
            
    print ">>> Training complete <<<"
            
    for i in range(N_test):
        
        # stage 1
        label_test, item_test = MNIST.GetTestingItem(i)
        item_test = reshape(item_test,(28,28))
                
        img00 = reshape(item_test[0:14,0:14], (14*14,1))
        img01 = reshape(item_test[0:14,14:28], (14*14,1))
        img10 = reshape(item_test[14:28,0:14], (14*14,1))
        img11 = reshape(item_test[14:28,14:28], (14*14,1))
                
        mid_hat1 = q1.recall(img00)
        mid_hat2 = q2.recall(img01)
        mid_hat3 = q3.recall(img10)
        mid_hat4 = q4.recall(img11)
        
        # stage 2
        mid_hat = concatenate((mid_hat1, mid_hat2, mid_hat3, mid_hat4))
        out_hat = q.recall(mid_hat)
                
        out_hat_max = out_hat.argmax()
        output_max = label_test.argmax()
        
        if (out_hat_max!= output_max):
            class_errors[ii] +=1
            elist.append(i)
        
        if (i+1)%100 == 0:
            print "Current # of testing iter is %d of %d"%(i+1, N_test)
            
    elist.append(-1)
    
    print ">>> Testing complete <<<"
        
for l in range(N_sim):
     Accuracy.append( 1-class_errors[l]/float(N_test) )

toc = time.time()
timeElapsed = toc-tic
mean_acc = mean(Accuracy)

print ""
print "TimeElapsed = %6.2f seconds "%timeElapsed
print "The Accuracy is :", Accuracy 
print "The mean Accuracy is :", mean_acc  
        






