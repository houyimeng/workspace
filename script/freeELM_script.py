# -*- coding: utf-8 -*-
"""
Created on Thu Mar 10 13:45:11 2016

@author: Yimeng
"""

from freeELM import freeELM


layout = [4, 1]
oneELM = freeELM(layout)

oneELM.trainLoop()
accuracy = oneELM.testLoop()




