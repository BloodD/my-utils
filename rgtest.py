#!/usr/bin/env python
"""
This code is fot testing the region growing.
"""
import os
import sys
import time
import nibabel as nib
import region_growing as rg
import matplotlib.pyplot as plt 
import numpy as np

img = nib.load("zstat1.nii.gz")
data = img.get_data()
#test coor [36,60,28] [21,39,30] [23,38,30]
coor = [23,38,30]
num = 10000

size_list = []
st = time.time()

for t in range(1,50):
    t = t/10.0
    print t
    region_img,size = rg.region_growing(data,coor,float(t),num,6)
    print "Totoal time is :%s"%(time.time()-st)
    size_list.append([t,size])

print size_list
size_list = np.array(size_list)

plt.plot(size_list[:,0],size_list[:,1],'ro')
plt.show()

result = img
result._data = region_img
nib.save(result,"region.nii.gz")

