# -*- coding: utf-8 -*-

from numpy import  zeros, random

# generate a matrix of random bits
def randbits(n,m):
    output = zeros((n,m))
    for row in range(n):
        one_row = random.choice([-1, 1], m)
        output[row,:] = one_row
    return output
    
# concatenate array into single string
def catbits(inputarray):
    output = ''
    for i in range (0,len(inputarray)):
        output = output + str(inputarray[i])
    return output

# separate a string of bits into array
def sepbits(inputarray):
    output = zeros(len(inputarray))
    for i in range (0,len(inputarray)):
        output[i] = int(inputarray[i])
    return output
    
    


