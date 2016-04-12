# -*- coding: utf-8 -*-
"""
Created on Fri Mar 18 09:36:27 2016

@author: yimeng
"""

from ELM.ELMrecogPic_beta3 import ELMrecogPic_beta3

'''
from numpy import reshape, arange, array

aaa = array([[8,7,5], [-1,1,-1],[-1,2,1]])
print aaa
bbb = massflt(aaa)
print bbb
'''

newELMrecogOBJ = ELMrecogPic_beta3()
labelmatrix, frag_testcanvas = newELMrecogOBJ.runtest()



