# -*- coding: utf-8 -*-
"""
Created on Tue Mar  8 14:06:00 2016

@author: houyimeng
"""

from multiprocessing import Pool
from ELMrecogPic import ELMrecogPic
from numpy import zeros
import time

tic = time.time()
def ELMtask(inputarg):
    print inputarg
    anELM = ELMrecogPic()
    ans = anELM.runtest(6, 'line')
    return ans

worker = 4
sim = 4
if __name__ == '__main__':
    p = Pool(worker)
    ans = p.map(ELMtask, zeros(sim).tolist())
    
toc = time.time()

print "timeElapsed =", toc-tic