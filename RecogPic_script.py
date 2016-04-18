# -*- coding: utf-8 -*-
"""
Created on Fri Mar 18 09:36:27 2016

@author: yimeng
"""

from ELM.ELMrecogPic_real3 import ELMrecogPic_real3
import matplotlib.pylab as plt

newELMrecogOBJ = ELMrecogPic_real3()
labelmatrix_20, labelmatrix_24, labelmatrix_28, testcanvas = newELMrecogOBJ.runtest()
newELMrecogOBJ.visualize()











