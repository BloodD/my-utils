# emacs: -*- mode: python-mode; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##

import os 
import sys
import time as time
import numpy as np
import scipy as sp
import pylab as pl

#sys.path.remove('/usr/local/neurosoft/epd-7.2.1/lib/python2.7/site-packages/nibabel-1.3.0-py2.7.egg')
import matplotlib.cm as cm
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt

from mvpa2.base.hdf5 import *
from mvpa2.datasets.base import Dataset
from mvpa2.datasets.mri import map2nifti

from sklearn.feature_extraction import image
from sklearn.cluster.spectral import SpectralClustering
from pylab import imread, imshow, gray, mean
from scipy.spatial import distance as ds
import nibabel as ni

print ni.__version__
#print sys.path

def create_similarity_mat(mat,fa,g,tag):
    """
    First,compute the connectivity distance of each pair voxels in seeds mask.
    Second,compute the space distance
    Third,generate the similarity graph for spectral clustering.
    """
    case = tag
    voxel = np.array(fa.values())
   # print "seed voxel number:"
    print voxel[0]
    v = voxel[0]
    spacedist = ds.cdist(v,v,'euclidean') 
    #print "voxel spacial distance:"
    #print spacedist

    tmp = mat
    alpha = 1
    gama=g
    o = 10
    print "compute distance..."
    distance = ds.pdist(tmp,'euclidean')
    dist = ds.squareform(distance,force='no',checks=True)
    
    if case == 1:
        com = dist  
    elif case == 2: 
        com = dist*spacedist 
        print "dist*sp acedistance"
    else:
        return 0
    print "total distance :"
    print com

    similarity = [[np.exp(-gama*j/(o*o)) for j in i] for i in com]
    return similarity 

def open_conn_mat(filename):
    hfilename =  filename
    conn_profile = h5load(hfilename)
    #print conn_profile.samples
    return conn_profile

def add_spatial_constrain():
    pass 

def create_mask(conn,th,thn):
    mat = conn 
    shape = mat.shape
    n = shape[0]
    m = shape[1]
    i = 0
    ##### mask = [Falise,Ture,False,...,False] 
    mask = np.zeros(n)
    mask = np.array([False for x in mask])
    print mask.shape,n,m,th,thn

    while i < n:
        tmp = np.array(mat[i])
  #     print mat[i]
        zero = tmp.nonzero()
        non = np.array(zero)
        num = non.size
        n = float(n)
        thd = (n - num)/n
  #      print n,num,thd
        if  num >= thn:
            mask[i]= True
        #print thd 
        i+= 1  
    return  mask 

def mask_feature(mat,mask):
    map = mat 
    mak = mask
    map = map[mak]
    print map.shape
    return map
####
def mean_feature():
    #bmap=map[::,::3]+map[::,1::3]+map[::,2::3]
    #amap=bmap[::,::3]+bmap[::,1::3]+bmap[::,2::3]

    #print amap.shape
    return 0

def plot_simi_map():
    pass

def save_label():
    pass

def save_result():
    pass
          
# Apply spectral clustering (this step goes much faster if you have pyamg installed)

def main():
    '''
    Spectral clustering...
    '''
    st =  time.time()
    tmpset = Dataset([])
   # hfilename = "/nfs/j3/userhome/dangxiaobin/workingdir/cutROI/%s/fdt_matrix2_targets_sc.T.hdf5"%(id)
    hfilename = 'fdt_matrix2.T.hdf5'
    print hfilename
    #load connectivity profile of seed mask voxels  
    conn = open_conn_mat(hfilename) 
    tmpset.a = conn.a
    print conn.shape,conn.a
    #remove some features
    mask = create_mask(conn.samples,0.5,1)
   # print mask,mask.shape
    conn_m = mask_feature(conn.samples,mask)
   # print  conn_m
    map = conn_m.T
    print "map:"
    print map.shape,map.max(),map.min()
    
    voxel = np.array(conn.fa.values())
    print voxel[0]
    v = voxel[0]
    spacedist = ds.cdist(v,v,'euclidean') 
    print spacedist

    """
    similar_mat = create_similarity_mat(map,conn.fa,0.1,2)
    X = np.array(similar_mat)
    print "similarity matrix: shape:",X.shape
    print X
    """
    
    corr = np.corrcoef(map)
    corr = np.abs(corr)
    corr = 0.1*corr + 0.9/(spacedist+1)
    
    print "Elaspsed time: ", time.time() - st
    print corr.shape,corr
    plt.imshow(corr,interpolation='nearest',cmap=cm.jet)
    cb = plt.colorbar() 
    pl.xticks(())
    pl.yticks(())
    pl.show()
    
    cnum = 3
    near = 100
    sc = SpectralClustering(cnum,'arpack',None,100,1,'precomputed',near,None,True)
    #sc.fit(map)
    sc.fit_predict(corr)
    '''
    cnum = 3
    near = 100
    sc = SpectralClustering(cnum,'arpack',None,100,1,'nearest_neighbors',near,None,True)
    sc.fit(map)
   # sc.fit_predict(X)
   # param = sc.get_params(deep=True)
    '''
    tmpset.samples = sc.labels_+1
   # print sc.affinity_matrix_
    #print list(sc.labels_)
    print "Elaspsed time: ", time.time() - st
    print "Number of voxels: ", sc.labels_.size
    print "Number  of clusters: ", np.unique(sc.labels_).size

    result = map2nifti(tmpset)
    result.to_filename("fg_parcel_S0006.nii.gz")
    print ".....The end........"

if __name__ == '__main__':
    main()
