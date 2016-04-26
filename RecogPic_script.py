# -*- coding: utf-8 -*-
"""
Created on Fri Mar 18 09:36:27 2016

@author: yimeng
"""

from ELM.ELMrecogPic_v1 import ELMrecogPic_v1
from ELM.ELMrecogPic_v2 import ELMrecogPic_v2
import time


tic = time.time()
hellokitty1 = ELMrecogPic_v2()
labelhat = hellokitty1.runtest()
hellokitty1.savePATCH()
hellokitty1.visualize()
toc = time.time()

timeElapsed1 = toc-tic
print "TIme Elapsed = ", timeElapsed1


