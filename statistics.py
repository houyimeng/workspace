# -*- coding: utf-8 -*-
"""
Created on Mon Apr 25 16:12:04 2016

@author: ThinkPad
"""

import os
import numpy as np
import matplotlib.pylab as plt

path = "C:\\dataspace\\harbour\\colorful_merge\\"

dirs = os.listdir(path)

nums = np.zeros(len(dirs)-1)
for i in range(len(dirs)-1):
    fullpath = path+str(i)+"\\"
    dirs2 = os.listdir(fullpath)
    nums[i] = len(dirs2)

plt.figure()
plt.plot(nums,'-x')

np.savetxt('statistics.txt', nums)