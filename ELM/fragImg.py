# -*- coding: utf-8 -*-
"""
Created on Mon Apr 11 14:29:32 2016

@author: ThinkPad
"""
import math
from numpy import pad, zeros
import matplotlib.pylab as plt

def zeropadding(matrix, leftPad,rightPad,topPad,bottomPad, constant_val = 0):
    pads = ((leftPad,rightPad),(topPad,bottomPad))
    return pad(matrix, pads, 'constant', constant_values = constant_val)

def fragImg(inputImg, wind_size):
    
    a, b = inputImg.shape
    a_pad = wind_size[0] - a%wind_size[0]
    b_pad = wind_size[1] - b%wind_size[1]
    a_num = math.ceil(a/float(wind_size[0]))
    b_num = math.ceil(b/float(wind_size[1]))
    
    pad_upper = b_pad/2
    pad_lower = b_pad-pad_upper
    pad_left = a_pad/2
    pad_right = a_pad-pad_left
    
    inputImg_aft = zeropadding(inputImg, pad_left, pad_right, pad_upper, pad_lower, constant_val = 0)

    tot_img = int(a_num)*int(b_num)
    img_patches = zeros((tot_img, wind_size[0]*wind_size[1]))

    # segment images
    for i in range(int(a_num)):
        for j in range(int(b_num)):
            
            temp = inputImg_aft[i*wind_size[0]:(i+1)*wind_size[0], j*wind_size[1]:(j+1)*wind_size[1]]
            img_patches[int(b_num)*i+j,:] = temp.flatten()
            
    return img_patches

'''
path = "C:\\comingdata\\others\\snapshot.jpg"
IMG = cv2.imread(path, 0)

img_flt = fragImg(IMG)

plt.figure()
plt.imshow(IMG)
plt.figure()
plt.imshow(img_flt[0])
'''



    